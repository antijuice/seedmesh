"""Verifying real requests, from a real client, while it is being used.

The verification layer has worked for a long time and has never run outside
`tools/backend_demo.py`, which drives it by hand. That is the gap between "the trust layer
exists" and "the trust layer runs": a client measured latency and success, which any
monitoring tool does, and never checked whether the numbers coming back were *correct*.

This closes it. `seedmesh chat` now samples its own requests, re-runs a sampled one on a
second server in a different network cluster, and compares fingerprints.

**Only prefill steps are verifiable, and getting this wrong would manufacture fraud.**

Petals runs inference as a stateful session: each decode step's output depends on the KV
cache accumulated from every previous token. Capturing one decode step's input and replaying
it against another server compares a computation-with-history to a computation-without, and
the outputs differ for entirely honest reasons. A sampler that did that would report constant
MISMATCHes against innocent peers — the exact failure this project keeps having to design
against.

The prefill step is the exception, and it is a real one: it runs with an empty cache, so it
is *definitionally* equivalent to a stateless forward over the same tokens. That is the only
step captured here (`session.position == 0`), and it is also the most valuable one, since it
carries the whole prompt rather than a single token.

**Verification happens off the inference path.** Samples are captured cheaply during a
request and replayed between prompts, alongside the gossip round. Double-routing inline would
double the user's latency to check a small fraction of requests.

**The Petals call is injected**, so the sampling policy, the comparison and the bookkeeping
are all testable without a backend, a GPU or a swarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np

from seedmesh.core.types import BlockRange, PeerId, ServerInfo, Verdict
from seedmesh.verification.compare import compare_sketches
from seedmesh.verification.sketch import SketchSpec, derive_seed, sketch_hidden_state

# Bounded because each entry holds a prompt-sized activation tensor. Verification samples the
# tail of what a session did; it does not need a transcript of it.
MAX_PENDING = 8


@dataclass
class CapturedStep:
    """One prefill step, kept so it can be replayed against a second server."""

    subject: PeerId
    block_range: BlockRange
    inputs: np.ndarray
    outputs: np.ndarray


@dataclass
class VerificationOutcome:
    subject: PeerId
    verifier: PeerId
    verdict: Verdict
    distance: float
    detail: str
    comparable: bool = True


class InlineVerifier:
    """Samples real requests and checks them against an independent second server."""

    def __init__(
        self,
        sampler,
        *,
        run_on: Callable[[ServerInfo, BlockRange, np.ndarray, bytes], Optional[np.ndarray]],
        discover: Callable[[], List[ServerInfo]],
        sketch_spec: Optional[SketchSpec] = None,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self.sampler = sampler
        self.run_on = run_on
        self.discover = discover
        self.sketch_spec = sketch_spec or SketchSpec()
        self.max_pending = max_pending

        self.pending: List[CapturedStep] = []
        self.outcomes: List[VerificationOutcome] = []
        self.checked = 0
        self.skipped_no_verifier = 0
        self.not_sampled = 0
        self.skipped_unlocated = 0
        """Skips where candidates DID host the blocks but none could be placed on a network.
        Counted apart because the two have different remedies: "nobody else hosts these
        blocks" needs another volunteer, while "nobody's address is knowable" is what
        happens when every server is reachable only through a relay."""

    # ---- capture (on the request path -- must stay cheap) ----------------------

    def capture(
        self, subject: PeerId, start: int, end: int, model_id: str, inputs, outputs
    ) -> bool:
        """Record a prefill step for later replay. Returns whether it was kept.

        Called from the observer's wrapper, so it runs inside a live inference request. It
        copies two tensors and returns; everything expensive happens later.
        """
        if end <= start or len(self.pending) >= self.max_pending:
            return False
        try:
            captured_in = _to_array(inputs)
            captured_out = _to_array(outputs)
        except Exception:
            return False
        if captured_in is None or captured_out is None:
            return False

        self.pending.append(
            CapturedStep(
                subject=subject,
                block_range=BlockRange(model_id, start, end),
                inputs=captured_in,
                outputs=captured_out,
            )
        )
        return True

    # ---- replay (between prompts -- may be slow) -------------------------------

    def run_pending(self, limit: int = 1) -> List[VerificationOutcome]:
        """Verify up to ``limit`` captured steps. Returns what was decided."""
        results: List[VerificationOutcome] = []
        if not self.pending:
            return results

        try:
            servers = self.discover()
        except Exception:
            self.pending.clear()
            return results

        by_id = {s.peer_id: s for s in servers}
        while self.pending and len(results) < limit:
            step = self.pending.pop(0)
            subject = by_id.get(step.subject)
            if subject is None:
                # The peer that served us is no longer announced. Not evidence of anything.
                continue
            outcome = self._verify(step, subject, servers)
            if outcome is not None:
                results.append(outcome)

        self.outcomes.extend(results)
        return results

    def _verify(
        self, step: CapturedStep, subject: ServerInfo, servers: List[ServerInfo]
    ) -> Optional[VerificationOutcome]:
        candidates = [s for s in servers if s.peer_id != subject.peer_id]

        # These are two different facts and `plan()` collapses them into one None.
        # Reporting both as "no independent verifier" sent a real debugging session chasing
        # firewalls and diversity rules for hours, when the swarm was eligible the whole
        # time and the sampler simply had not selected that request: an established peer
        # sits at base_rate=0.03, so seven prompts expect 0.2 verifications.
        if not self.sampler.should_verify(subject.peer_id):
            self.not_sampled += 1
            return None

        task = self.sampler.plan(subject, step.block_range, candidates, force=True)
        if task is None:
            # No independent, comparable partner. Recorded as a skip, never as a pass:
            # "could not check" and "checked and fine" must not look the same.
            self.skipped_no_verifier += 1
            covering = [
                c for c in candidates if c.block_range.covers(step.block_range)
            ]
            if covering and not any(c.address for c in covering):
                self.skipped_unlocated += 1
            return None

        replayed = None
        try:
            replayed = self.run_on(task.verifier, step.block_range, step.inputs, task.nonce)
        except Exception:
            replayed = None

        if replayed is None:
            # The verifier failed to answer. That is a reliability signal about the verifier,
            # which the observer already records -- it is not evidence about the subject.
            self.sampler.record_outcome(task, Verdict.INCONCLUSIVE, comparable=False)
            return VerificationOutcome(
                subject.peer_id, task.verifier.peer_id, Verdict.INCONCLUSIVE,
                float("inf"), "verifier did not answer", comparable=False,
            )

        seed = derive_seed(task.nonce, step.block_range.key(), 0)
        ours = sketch_hidden_state(_flatten(step.outputs), seed=seed, spec=self.sketch_spec)
        theirs = sketch_hidden_state(_flatten(replayed), seed=seed, spec=self.sketch_spec)
        comparison = compare_sketches(ours, theirs, task.tolerance)

        self.sampler.record_outcome(task, comparison.verdict, comparable=comparison.comparable)
        self.checked += 1
        return VerificationOutcome(
            subject.peer_id,
            task.verifier.peer_id,
            comparison.verdict,
            comparison.relative_distance,
            comparison.reason,
            comparable=comparison.comparable,
        )

    # ---- reporting -------------------------------------------------------------

    def describe(self) -> List[str]:
        if not self.checked and not self.skipped_no_verifier and not self.not_sampled:
            return []
        lines = [f"  verified    {self.checked} request(s) against a second server"]
        tally: dict[str, int] = {}
        for outcome in self.outcomes:
            tally[outcome.verdict.value] = tally.get(outcome.verdict.value, 0) + 1
        if tally:
            lines.append("    " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
        if self.not_sampled:
            # Not a problem, and must not be shown as one. Verification is sampled, not
            # exhaustive -- most requests are deliberately never checked.
            lines.append(
                f"    {self.not_sampled} not sampled (verification is a sample, not every "
                "request)"
            )
        if self.skipped_no_verifier:
            # Said out loud. A swarm that cannot verify must not look like one that verified
            # and found nothing wrong.
            lines.append(f"    {self.skipped_no_verifier} skipped: no independent verifier")
            if self.skipped_unlocated:
                lines.append(
                    "      other servers host these blocks, but none has a direct address --"
                )
                lines.append(
                    "      a relayed peer's address belongs to the relay, so it cannot be"
                )
                lines.append(
                    "      shown to be a different operator. Port-forwarded servers can verify."
                )
        return lines


def _to_array(tensor: Any) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32, copy=True)
    detach = getattr(tensor, "detach", None)
    if detach is None:
        return None
    return detach().cpu().float().numpy().copy()


def _flatten(array: np.ndarray) -> np.ndarray:
    """(1, tokens, dim) -> (tokens, dim); the sketch works on a 2-D activation."""
    return array[0] if array.ndim == 3 else array


def is_prefill(session: Any) -> bool:
    """Whether this step runs against an empty cache, and is therefore replayable.

    See the module docstring: replaying a decode step compares a computation that had history
    against one that did not, and reports honest servers as mismatched.
    """
    position = getattr(session, "position", None)
    return position == 0
