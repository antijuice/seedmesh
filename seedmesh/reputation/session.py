"""The client-side reputation stack, assembled.

Every piece of this existed before and none of it was connected to the command people run.
`seedmesh chat` talked to Petals and measured nothing, so the property the design is *for* --
a client arriving with the swarm's collective experience rather than a blank slate -- was
true only in `tools/backend_demo.py`.

One object, so `_cmd_chat` stays a chat loop:

    identity    persistent Ed25519 key       who signs what this node reports
    scorer      persisted per swarm          what this node measured itself
    observer    wraps Petals' session.step   how measurement happens at all
    aggregator  in memory, per session       what other observers reported
    gossip      publish + fetch over the DHT the exchange

**Ordering matters on both ends.** Load before attaching, so the first request is judged
against everything learned in previous sessions rather than a cold prior. Save before
shutting the DHT down, so a Ctrl-C -- the way a chat session normally ends -- keeps its
evidence.

**Unlocatable observers share one bucket, on purpose.** The aggregator caps how much weight
any one network cluster can contribute, which is what stops a hundred sybils from
out-shouting three honest peers. A client has no address for another *client*: clients
announce no blocks and are invisible to discovery. Filing each unlocatable observer under its
own cluster would hand an attacker exactly the amplification the cap exists to prevent, so
they all land in `UNKNOWN_CLUSTER`, whose budget the aggregator already sets lower than a
normal cluster's. An attacker gains nothing by being unlocatable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from seedmesh.core.clock import Clock, SystemClock
from seedmesh.reputation.aggregate import AggregationConfig, ReputationAggregator
from seedmesh.reputation.diversity import UNKNOWN_CLUSTER
from seedmesh.reputation.gossip import ReputationGossip
from seedmesh.reputation.observer import InferenceObserver
from seedmesh.reputation.scorer import ReputationScorer

DEFAULT_SYNC_INTERVAL_S = 300.0


class ClientReputation:
    """Measure the servers this client uses, remember it, and trade it with other clients."""

    def __init__(
        self,
        *,
        identity,
        model_id: str,
        state_path: Optional[Path],
        publish_fn: Callable,
        fetch_fn: Callable,
        transport_id: str,
        prefix: str = "seedmesh",
        clock: Optional[Clock] = None,
        sync_interval_s: float = DEFAULT_SYNC_INTERVAL_S,
    ) -> None:
        self.clock: Clock = clock or SystemClock()
        self.identity = identity
        self.state_path = Path(state_path) if state_path is not None else None
        self.sync_interval_s = sync_interval_s

        self.scorer = ReputationScorer(clock=self.clock)
        self.loaded = self.scorer.load(self.state_path) if self.state_path else 0

        self.observer = InferenceObserver(
            identity.peer_id, self.scorer, model_id, clock=self.clock
        )

        # See the module docstring: one shared bucket for everything we cannot place.
        self._clusters: dict[str, str] = {}
        self.aggregator = ReputationAggregator(self._cluster_of, AggregationConfig())

        self.gossip = ReputationGossip(
            identity,
            self.scorer,
            self.aggregator,
            publish_fn=publish_fn,
            fetch_fn=fetch_fn,
            transport_id=transport_id,
            prefix=prefix,
            clock=self.clock,
            # No observer_locator. `ReputationGossip` offers one so a caller can attribute an
            # observer to a network cluster, and `tools/backend_demo.py` can: it knows the
            # addresses of the *servers* it discovered. A chat client cannot, because the
            # peers it fetches from are other *clients*, which announce no blocks and are
            # invisible to discovery. Supplying the transport id as a stand-in would look
            # like attribution while carrying no information about which network the peer is
            # on -- and a wrong cluster label is worse than none, because the diversity check
            # would then treat colluding peers as independent. So they stay unlocated, in the
            # bucket sized for exactly that.
        )

        self._last_sync = 0.0
        self._detach: Optional[Callable[[], None]] = None
        self.verifier = None
        self.gate = None

    def _cluster_of(self, peer_id: str) -> str:
        return self._clusters.get(peer_id, UNKNOWN_CLUSTER)

    def enable_verification(self, *, run_on, discover, clusters=None, config=None) -> None:
        """Turn on inline verification: sample real requests, re-check them elsewhere.

        Kept separate from the constructor because it is the one part that needs a live
        Petals client. Reputation without it is still worth having -- latency and success
        are real signals -- but only this answers whether the numbers were *correct*, which
        is the property the whole project is for.
        """
        from seedmesh.reputation.diversity import ClusterIndex
        from seedmesh.verification.inline import InlineVerifier
        from seedmesh.verification.sampler import SamplerConfig, VerificationSampler

        sampler = VerificationSampler(
            self.scorer,
            clusters or ClusterIndex(),
            config or SamplerConfig(),
        )
        self.verifier = InlineVerifier(sampler, run_on=run_on, discover=discover)
        self.observer.verifier = self.verifier

    def enable_routing_gate(self, sequence_manager) -> None:
        """Stop routing to peers with corroborated evidence of incorrect work.

        The last step of the loop: measure, share, and finally *act*. Deliberately a veto on
        proven-wrong peers rather than a preference for high scorers -- see
        `seedmesh/reputation/routing.py` for why ranking by score starves honest slow
        volunteers.
        """
        from seedmesh.reputation.routing import make_gate

        self.gate = make_gate(sequence_manager, self.scorer, self.aggregator)

    # ---- lifecycle ---------------------------------------------------------------

    def attach(self, session_class: Optional[type] = None) -> None:
        """Start measuring. With no argument, attaches to Petals' real session class."""
        from seedmesh.reputation.observer import attach, observe_petals

        self._detach = (
            attach(session_class, self.observer)
            if session_class is not None
            else observe_petals(self.observer)
        )

    def detach(self) -> None:
        if self._detach is not None:
            self._detach()
            self._detach = None

    def save(self) -> bool:
        if self.state_path is None:
            return False
        try:
            self.scorer.save(self.state_path)
            return True
        except Exception:
            # Losing a session's evidence is bad; crashing on the way out of a chat the user
            # already ended is worse.
            return False

    def close(self) -> tuple[bool, int]:
        """Final publish and save. Returns ``(saved, accepted)``."""
        self.detach()
        self.run_verification()
        accepted = 0
        try:
            _, accepted = self.gossip.sync()
        except Exception:
            accepted = 0
        return self.save(), accepted

    # ---- periodic work -----------------------------------------------------------

    def due_for_sync(self) -> bool:
        return (self.clock.now() - self._last_sync) >= self.sync_interval_s

    def maybe_sync(self, force: bool = False) -> Optional[int]:
        """One gossip round if enough time has passed. Returns records accepted, or None.

        Called between prompts rather than on a timer: a DHT round-trip during generation
        would compete with the request it is supposed to be measuring, and a background
        thread touching hivemind mid-inference is a concurrency problem nobody needs. The
        user is typing anyway.
        """
        if not force and not self.due_for_sync():
            return None
        self._last_sync = self.clock.now()
        # Verification before gossip, so a verdict reached this round is published this
        # round rather than sitting locally until the next one.
        self.run_verification()
        # ...and the gate after it, so a mismatch confirmed this round stops receiving work
        # immediately rather than at the next prompt.
        self.refresh_gate()
        try:
            _, accepted = self.gossip.sync()
        except Exception:
            return None
        self.save()
        return accepted

    def refresh_gate(self) -> set:
        """Re-apply routing exclusions. Never raises into the caller."""
        if self.gate is None:
            return set()
        try:
            newly = self.gate.refresh()
        except Exception:
            return set()
        for peer_id in newly:
            print(f"  routing: excluding ...{peer_id[-8:]} (corroborated incorrect work)")
        return newly

    def run_verification(self, limit: int = 1) -> list:
        """Replay a sampled request against a second server. Never raises into the caller."""
        if self.verifier is None:
            return []
        try:
            return self.verifier.run_pending(limit=limit)
        except Exception:
            return []

    # ---- reporting ---------------------------------------------------------------

    def describe(self) -> list[str]:
        lines: list[str] = []
        if self.loaded:
            lines.append(f"  recalled {self.loaded} peer(s) from previous sessions")
        lines.extend(self.observer.describe())
        if self.verifier is not None:
            lines.extend(self.verifier.describe())
        if self.gate is not None:
            lines.extend(self.gate.describe())
        lines.extend(self.gossip.describe())
        heard = sorted(self.aggregator.subjects())
        if heard:
            lines.append(f"  other observers reported on {len(heard)} server(s):")
            for subject in heard[:10]:
                score = self.aggregator.aggregate(subject)
                lines.append(
                    f"    ...{subject[-8:]}  reliability {score.reliability:.3f} "
                    f"from {score.observer_count} observer(s) in "
                    f"{score.distinct_clusters} cluster(s)"
                )
        return lines


def for_dht(
    dht: Any,
    *,
    identity,
    model_id: str,
    state_path: Optional[Path],
    prefix: str = "seedmesh",
    sync_interval_s: float = DEFAULT_SYNC_INTERVAL_S,
) -> ClientReputation:
    """Build a `ClientReputation` bound to a live hivemind DHT."""
    from seedmesh.reputation.dht import transport_for

    publish, fetch = transport_for(dht)
    return ClientReputation(
        identity=identity,
        model_id=model_id,
        state_path=state_path,
        publish_fn=publish,
        fetch_fn=fetch,
        transport_id=str(dht.peer_id),
        prefix=prefix,
        sync_interval_s=sync_interval_s,
    )
