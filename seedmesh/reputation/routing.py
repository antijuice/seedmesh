"""Letting measured trust actually change where requests go.

The loop was measure -> share -> *nothing*. A client collected evidence, published it, and
then routed exactly as it would have with no trust layer at all. This is the step that makes
the evidence matter.

**A veto, not a preference ordering — and that distinction is the whole design.**

The tempting version ranks servers by reputation score and prefers the best. It is also the
version that keeps going wrong here: a score blends reliability, latency and integrity, so
preferring high scores means preferring *fast* servers, and a volunteer on a slow home
connection gets quietly starved of traffic for being honest and slow. That is the defect this
project has had to undo repeatedly, and a routing rule is the worst place to reintroduce it.

So this excludes peers with **corroborated evidence of incorrect work** and does nothing else.
Slow is not excluded. Unreliable is not excluded. Unproven is not excluded. Only quarantine —
which `ScorerConfig` already gates behind agreement from at least two *distinct* disputing
partners, precisely so one peer's accusation cannot evict another.

**Latency stays Petals' job, correctly.** A prior version of this reasoning (mine, repeated in
several commit messages) claimed Petals routes on self-reported throughput. That is only
partly true, and the difference matters: inference uses `min_latency`, a Dijkstra path whose
client-to-server leg is round-trip time the client *measured itself*. What is self-reported is
the compute term (`inference_rps`), the server-to-server pings, and remaining cache. So Petals
already measures the thing it is best placed to measure, and the honest division of labour is:
it decides *how fast*, we decide *whether the answer can be trusted*. A server cannot
self-report its way out of a mismatch that two independent verifiers confirmed.

**Implemented through a supported mechanism, not a patch.** `ClientConfig.blocked_servers` is
filtered inside `RemoteSequenceManager._update()`, so an excluded peer never enters the
routing graph at all — no wrapper, no monkey-patch, no codemod entry. The set is recomputed
from the scorer rather than accumulated, so a peer that is later cleared is un-blocked
automatically.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

from seedmesh.core.types import PeerId
from seedmesh.reputation.routing_bias import Router


class ReputationGate:
    """Keeps a Petals client from routing to peers proven to compute incorrectly."""

    def __init__(self, sequence_manager: Any, router: Router) -> None:
        self.sequence_manager = sequence_manager
        self.router = router
        self.blocked: Set[PeerId] = set()
        self.ever_blocked: Set[PeerId] = set()

    def excluded_among(self, peer_ids: List[PeerId]) -> Set[PeerId]:
        """Which of these peers are barred. Delegates to the Router's policy.

        Deliberately one policy, not two: `Router.is_excluded` already encodes "corroborated
        quarantine, never a bare low score", and a second copy here would drift from it.
        """
        return {peer_id for peer_id in peer_ids if self.router.is_excluded(peer_id)}

    def known_peers(self) -> List[PeerId]:
        """Every peer the client currently knows about, from its own routing table."""
        try:
            infos = self.sequence_manager.state.sequence_info.block_infos
        except Exception:
            return []
        peers: Set[PeerId] = set()
        for info in infos or ():
            for peer_id in (getattr(info, "servers", None) or {}):
                peers.add(str(peer_id))
        return sorted(peers)

    def refresh(self) -> Set[PeerId]:
        """Recompute the blocked set and apply it. Returns peers newly blocked.

        Recomputed from scratch every time rather than accumulated, so clearing a peer's
        quarantine restores it to routing without anyone having to remember to un-block it.
        """
        blocked = self.excluded_among(self.known_peers())
        newly = blocked - self.blocked
        if blocked == self.blocked:
            return set()

        self.blocked = blocked
        self.ever_blocked |= blocked
        self._apply(blocked)
        return newly

    def _apply(self, blocked: Set[PeerId]) -> None:
        """Push the set into Petals' own filter.

        `blocked_servers` is read from the config once, at construction, into
        `sequence_manager.blocked_servers` -- so setting it on the config after the fact
        does nothing. The manager's own attribute is what `_update()` consults.
        """
        try:
            from hivemind.p2p import PeerID

            as_peer_ids = {PeerID.from_base58(peer_id) for peer_id in blocked}
        except Exception:
            as_peer_ids = set(blocked)

        self.sequence_manager.blocked_servers = as_peer_ids or None
        # Without this the change waits for the next periodic refresh (60s by default),
        # which is a long time to keep sending work to a peer known to compute it wrongly.
        try:
            self.sequence_manager.update(wait=False)
        except Exception:
            pass

    def describe(self) -> List[str]:
        if not self.ever_blocked:
            return []
        lines = [
            f"  routing     excluded {len(self.ever_blocked)} peer(s) with corroborated "
            "evidence of incorrect work"
        ]
        for peer_id in sorted(self.ever_blocked):
            breakdown = self.router.scorer.breakdown(peer_id)
            lines.append(
                f"    ...{peer_id[-8:]}  integrity {breakdown.integrity:.3f}, "
                f"{breakdown.disputing_partners} disputing partner(s)"
            )
        return lines


def make_gate(sequence_manager: Any, scorer, aggregator=None) -> Optional[ReputationGate]:
    """Build a gate for a live client, or None if the client cannot be gated."""
    if sequence_manager is None:
        return None
    return ReputationGate(sequence_manager, Router(scorer, aggregator=aggregator))
