"""Turning a real inference client into a reputation observer.

The reputation stack could always *score* observations. Nothing produced them outside
`tools/backend_demo.py`, which drives `PetalsBackend` by hand. `seedmesh chat` -- the command
people actually run -- talks to `AutoDistributedModelForCausalLM` directly and measured
nothing, so the swarm's collective memory was fed by a demo script and by nobody else.

**Where the measurement comes from.** Petals routes a request through one
`_ServerInferenceSession` per span, and every per-server request in a generation is exactly
one call to that object's `step()`. Wrapping that single method yields, per request: the
server's transport peer id (`self.span.peer_id`), the blocks it covered
(`self.span.start:end`), a precise wall-clock duration, the token count, and the exception if
it failed. Nothing else in the client has all five in one place.

`RemoteSequenceManager.on_request_success` / `on_request_failure` look like the intended
hooks, and they are called on the same paths -- but they receive only a peer id. No timing,
no block range, no exception. Latency is most of what reputation is *for* here, so they are
the wrong seam.

**Why wrap rather than patch.** `tools/port_petals.py` is this project's usual answer to
"Petals does not do X", but a codemod changes what every volunteer installs and has to be
re-applied on every `seedmesh setup`. This is pure client-side observation of an unmodified
call, so it belongs in our process at runtime. Nothing about the request changes; a wrapper
that raised would be worse than no reputation at all, so every failure path here re-raises
the original exception untouched and swallows its own bookkeeping errors.

**Refusal is not failure.** A server at capacity raises `AllocationFailed`, which is correct,
honest behaviour -- and `Outcome.REFUSED` exists precisely so that it costs the operator
nothing. By the time the client sees it the exception has crossed a process boundary and been
flattened into a message string, so the only thing left to match on is the text. That
fragility points the wrong way: a rename upstream would silently downgrade honest
backpressure into `ERROR` and start punishing operators for behaving correctly. So
`tools/verify_petals_port.py` asserts the symbol still exists in the installed checkout,
which turns a silent reputational bug into a loud install-time one.

**Not wired, on purpose: this does not steer routing.** Petals picks spans by throughput, and
overriding that with reputation is a policy change that needs its own measurement first --
a bias strong enough to avoid a faulty server is also strong enough to strand an honest slow
one, which is the failure this project keeps having to un-make. Observation and gossip first;
routing influence when there is evidence for a rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from seedmesh.core.clock import Clock, SystemClock
from seedmesh.core.types import BlockRange, Observation, Outcome, PeerId
from seedmesh.reputation.scorer import ReputationScorer

# The server-side exception class name, as it survives serialisation into the client's error
# text. Kept as a constant so `tools/verify_petals_port.py` can check for the same string.
REFUSAL_MARKER = "AllocationFailed"


def classify(exc: BaseException) -> Outcome:
    """Map a client-side exception onto an outcome the scorer can act on.

    The three-way split matters: TIMEOUT is churn (a laptop closed its lid), REFUSED is
    honest backpressure, and only ERROR suggests the server did something wrong. Collapsing
    them would make an over-subscribed but well-behaved server look faulty.
    """
    import asyncio

    text = f"{type(exc).__name__}: {exc}"
    if REFUSAL_MARKER in text:
        return Outcome.REFUSED
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return Outcome.TIMEOUT
    # Petals wraps a remote timeout in its own exception type, so the class check above is
    # not sufficient on its own.
    if "TimeoutError" in text or "timed out" in text.lower():
        return Outcome.TIMEOUT
    return Outcome.ERROR


@dataclass
class ObserverStats:
    requests: int = 0
    ok: int = 0
    timeout: int = 0
    error: int = 0
    refused: int = 0
    tokens: int = 0

    def note(self, outcome: Outcome, tokens: int) -> None:
        self.requests += 1
        self.tokens += tokens
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)


class InferenceObserver:
    """Records first-hand experience of the servers this client actually used."""

    def __init__(
        self,
        observer_id: PeerId,
        scorer: ReputationScorer,
        model_id: str,
        *,
        clock: Optional[Clock] = None,
    ) -> None:
        self.observer_id = observer_id
        self.scorer = scorer
        self.model_id = model_id
        self.clock: Clock = clock or SystemClock()
        self.stats = ObserverStats()
        self.subjects: set[PeerId] = set()

    def note(
        self,
        subject: PeerId,
        start: int,
        end: int,
        outcome: Outcome,
        latency_ms: float,
        *,
        tokens: int = 0,
    ) -> None:
        # A zero-length span is not a measurement of anything, and BlockRange rightly
        # refuses to represent one. Drop it rather than let it raise into the request path.
        if end <= start:
            return
        self.scorer.record(
            Observation(
                observer=self.observer_id,
                subject=subject,
                block_range=BlockRange(self.model_id, start, end),
                outcome=outcome,
                latency_ms=max(0.0, latency_ms),
                at=self.clock.now(),
                tokens=tokens,
            )
        )
        self.subjects.add(subject)
        self.stats.note(outcome, tokens)

    def describe(self) -> list[str]:
        """A short human summary, for printing when a chat session ends."""
        stats = self.stats
        if not stats.requests:
            return ["  no servers measured this session"]
        lines = [
            f"  measured {stats.requests} request(s) across {len(self.subjects)} server(s): "
            f"{stats.ok} ok, {stats.timeout} timeout, {stats.error} error, "
            f"{stats.refused} refused"
        ]
        for breakdown in self.scorer.ranked(sorted(self.subjects)):
            lines.append(
                f"    ...{breakdown.peer_id[-8:]}  score {breakdown.score:.3f}  "
                f"reliability {breakdown.reliability:.3f}  p50 {breakdown.p50_ms:.0f} ms"
            )
        return lines


def _span_of(session: Any) -> Optional[tuple[PeerId, int, int]]:
    span = getattr(session, "span", None)
    if span is None:
        return None
    peer_id = getattr(span, "peer_id", None)
    start = getattr(span, "start", None)
    end = getattr(span, "end", None)
    if peer_id is None or start is None or end is None:
        return None
    return str(peer_id), int(start), int(end)


def _token_count(args: tuple, kwargs: dict) -> int:
    inputs = args[0] if args else kwargs.get("inputs")
    shape = getattr(inputs, "shape", None)
    if shape is not None and len(shape) >= 2:
        try:
            return int(shape[1])
        except (TypeError, ValueError):
            return 0
    return 0


def attach(session_class: type, observer: InferenceObserver, *, method: str = "step") -> Callable[[], None]:
    """Wrap ``session_class.step`` so every per-server request is measured.

    Returns a callable that restores the original method. Takes the class rather than
    importing Petals so the whole mechanism is testable on any platform -- including
    Windows, where the backend does not run at all.

    Idempotent: attaching twice would double-count every request, so a second attach on an
    already-wrapped class is refused with a no-op detach.
    """
    original = getattr(session_class, method)
    if getattr(original, "_seedmesh_observed", False):
        return lambda: None

    def wrapper(self, *args, **kwargs):
        span = _span_of(self)
        if span is None:
            # Not something we can attribute to a server; measuring it would put latency
            # against an unknown peer id, which is worse than not measuring it.
            return original(self, *args, **kwargs)
        subject, start, end = span
        tokens = _token_count(args, kwargs)
        began = time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            elapsed_ms = (time.perf_counter() - began) * 1000.0
            _safely(observer.note, subject, start, end, classify(exc), elapsed_ms, tokens=tokens)
            raise
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        _safely(observer.note, subject, start, end, Outcome.OK, elapsed_ms, tokens=tokens)
        return result

    wrapper._seedmesh_observed = True  # type: ignore[attr-defined]
    setattr(session_class, method, wrapper)

    def detach() -> None:
        setattr(session_class, method, original)

    return detach


def _safely(fn: Callable, *args, **kwargs) -> None:
    """Run bookkeeping without ever letting it break the request it is measuring."""
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def observe_petals(observer: InferenceObserver) -> Callable[[], None]:
    """Attach to Petals' real per-server session class. Returns a detach callable."""
    from petals.client.inference_session import _ServerInferenceSession

    return attach(_ServerInferenceSession, observer)
