"""Reputation-biased pipeline routing.

A client must pick one server per contiguous block range so the chosen servers together
cover the whole model. The spec asks for two things that pull against each other: prefer
*fewer hops* (each hop is a network round-trip) and prefer *higher reputation*. Greedy
per-range selection cannot trade those off, so routing is posed as a shortest-path problem
over block positions and solved exactly.

Two details that a naive implementation gets wrong:

**Cold-start starvation.** If routing always picks the best-known server, a new volunteer
never receives traffic, so it never accumulates history, so it stays unknown forever --
the network cannot absorb new capacity. Routing therefore scores on an optimistic upper
bound (UCB-style), giving unproven peers a bonus that decays as evidence arrives. This is
also what spreads load: a peer that is being hammered is not automatically preferred over
an equally good idle one.

**Where the optimum can sit.** Servers host overlapping ranges and a server may be used for
part of its range, so segment boundaries are not given. Because hop cost is linear in the
blocks a hop covers, an optimal segmentation always exists with every breakpoint at some
server's start or end -- a linear objective over an interval attains its minimum at an
endpoint. The search therefore runs over that finite boundary set and is exact, not a
heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from seedmesh.core.types import BlockRange, PeerId, ServerInfo
from seedmesh.reputation.aggregate import AggregateScore, ReputationAggregator, blend
from seedmesh.reputation.scorer import ReputationScorer


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    hop_penalty: float = 0.6
    """Cost of one extra hop, in block-equivalents. An extra hop is worth taking when it
    improves average per-block quality by more than ``hop_penalty / n_blocks``."""

    quality_weight: float = 1.0
    exploration_coefficient: float = 0.25
    """Strength of the optimism bonus for unproven peers. Zero disables exploration and
    reintroduces cold-start starvation."""

    load_weight: float = 4.0
    """How strongly recent load discourages reusing a server, in block-equivalents.

    Without this the swarm collapses onto whichever few servers offer the shortest path,
    because the cheapest route never changes once scores converge. For a volunteer network
    that is a failure mode rather than an efficiency: it burns out the handful of people
    the network depends on, wastes donated capacity, and concentrates the swarm onto nodes
    whose loss would take it down.

    It also feeds verification. A server that is never routed to is never checked, so
    spreading load is what gives redundant sampling anything to sample.

    **This costs latency, and the trade is real.** Measured over the ``mixed_threat``
    preset, 4 seeds x 400 requests (reproduce with ``seedmesh simulate``):

        load_weight   load Gini   cheats caught   mean latency
             0.0        0.829          33%           1646 ms
             1.5        0.780          67%           1999 ms
             4.0        0.723         100%           2287 ms
             9.0        0.645         100%           2956 ms

    4.0 is the knee: full detection at +39% latency. Past it, latency keeps degrading and
    detection has nowhere left to go. Deployments that care more about interactive latency
    than about donor burnout should lower it deliberately, knowing what it buys.

    These numbers come from a simulated topology and are not a measurement of any real
    swarm. Re-derive them once one exists."""

    load_half_life_s: float = 600.0
    max_alternates: int = 3

    def __post_init__(self) -> None:
        if self.hop_penalty < 0 or self.quality_weight <= 0:
            raise ValueError("hop_penalty must be >= 0 and quality_weight > 0")
        if self.load_weight < 0:
            raise ValueError("load_weight must be >= 0")


@dataclass(frozen=True, slots=True)
class Hop:
    server: ServerInfo
    covers: BlockRange
    score: float
    cost: float


@dataclass(frozen=True, slots=True)
class Route:
    model_id: str
    hops: tuple[Hop, ...]
    total_cost: float

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def peers(self) -> tuple[PeerId, ...]:
        return tuple(hop.server.peer_id for hop in self.hops)

    def covers(self, n_blocks: int) -> bool:
        """Whether the hops tile ``[0, n_blocks)`` exactly, in order and without gaps."""
        position = 0
        for hop in self.hops:
            if hop.covers.start != position:
                return False
            position = hop.covers.end
        return position == n_blocks

    def mean_score(self) -> float:
        total_blocks = sum(len(hop.covers) for hop in self.hops)
        if total_blocks == 0:
            return 0.0
        return sum(hop.score * len(hop.covers) for hop in self.hops) / total_blocks


class Router:
    """Plans inference pipelines, biased by reputation.

    Holds no network state of its own: candidates come from the DHT lookup, scores from the
    local scorer plus (optionally) aggregated swarm belief.
    """

    def __init__(
        self,
        scorer: ReputationScorer,
        config: Optional[RoutingConfig] = None,
        *,
        aggregator: Optional[ReputationAggregator] = None,
    ) -> None:
        self.scorer = scorer
        self.config = config or RoutingConfig()
        self.aggregator = aggregator
        self._load: dict[PeerId, float] = {}
        self._load_updated: float = scorer.clock.now()

    # ---- load tracking ---------------------------------------------------------

    def _decay_load(self) -> None:
        now = self.scorer.clock.now()
        elapsed = now - self._load_updated
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self.config.load_half_life_s)
        for peer_id in list(self._load):
            decayed = self._load[peer_id] * factor
            if decayed < 1e-3:
                del self._load[peer_id]
            else:
                self._load[peer_id] = decayed
        self._load_updated = now

    def note_assignment(self, peer_id: PeerId, blocks: int = 1) -> None:
        """Record that work was sent to a peer, so routing can spread the next request."""
        self._decay_load()
        self._load[peer_id] = self._load.get(peer_id, 0.0) + float(blocks)

    def recent_load(self, peer_id: PeerId) -> float:
        self._decay_load()
        return self._load.get(peer_id, 0.0)

    def _load_share(self, peer_id: PeerId) -> float:
        """This peer's share of recent work, in [0, 1]."""
        self._decay_load()
        total = sum(self._load.values())
        if total <= 0:
            return 0.0
        return self._load.get(peer_id, 0.0) / total

    # ---- scoring ---------------------------------------------------------------

    def _remote(self, peer_id: PeerId) -> Optional[AggregateScore]:
        return self.aggregator.aggregate(peer_id) if self.aggregator is not None else None

    def is_excluded(self, peer_id: PeerId) -> bool:
        """Whether a peer is barred from routing.

        Exclusion tracks the scorer's corroborated quarantine, never a bare low score --
        one peer's accusation must not be able to evict another. See
        :attr:`ScorerConfig.single_dispute_weight`.
        """
        if self.scorer.breakdown(peer_id).is_quarantined:
            return True
        remote = self._remote(peer_id)
        return bool(remote is not None and remote.mismatch_confirmed)

    def effective_score(self, peer_id: PeerId) -> float:
        """Optimistic score used for routing: belief plus an uncertainty bonus."""
        breakdown = self.scorer.breakdown(peer_id)
        base = blend(breakdown, self._remote(peer_id))
        bonus = self.scorer.exploration_bonus(
            peer_id, coefficient=self.config.exploration_coefficient
        )
        return max(0.0, min(1.0, base + bonus))

    # ---- planning --------------------------------------------------------------

    def plan(
        self,
        model_id: str,
        n_blocks: int,
        candidates: Iterable[ServerInfo],
        *,
        exclude: Optional[set[PeerId]] = None,
    ) -> Optional[Route]:
        """Find the minimum-cost pipeline covering ``[0, n_blocks)``.

        Returns ``None`` when the available servers cannot cover the model -- a real and
        expected condition in a churny volunteer swarm, not an error. Callers should
        surface it as honest unavailability rather than retrying blindly.
        """
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")

        exclude = exclude or set()
        usable = [
            server
            for server in candidates
            if server.model_id == model_id
            and server.peer_id not in exclude
            and not self.is_excluded(server.peer_id)
        ]
        if not usable:
            return None

        scores = {server.peer_id: self.effective_score(server.peer_id) for server in usable}
        congestion = {
            server.peer_id: self.config.load_weight * self._load_share(server.peer_id)
            for server in usable
        }

        boundaries = {0, n_blocks}
        for server in usable:
            start = max(0, min(server.block_range.start, n_blocks))
            end = max(0, min(server.block_range.end, n_blocks))
            boundaries.add(start)
            boundaries.add(end)
        positions = sorted(boundaries)
        index_of = {position: i for i, position in enumerate(positions)}

        infinity = float("inf")
        best: list[float] = [infinity] * len(positions)
        back: list[Optional[tuple[int, ServerInfo, float, float]]] = [None] * len(positions)
        best[index_of[0]] = 0.0

        # Deterministic iteration: equal-cost routes must resolve identically on every
        # client, otherwise identical inputs produce different pipelines run to run.
        usable.sort(key=lambda s: (s.block_range.start, s.block_range.end, s.peer_id))

        for i, position in enumerate(positions):
            if best[i] == infinity or position >= n_blocks:
                continue
            for server in usable:
                block_range = server.block_range
                if block_range.start > position or block_range.end <= position:
                    continue
                reach = min(block_range.end, n_blocks)
                score = scores[server.peer_id]
                per_block = self.config.quality_weight * (1.0 - score)
                hop_cost = self.config.hop_penalty + congestion[server.peer_id]
                for j in range(i + 1, len(positions)):
                    target = positions[j]
                    if target > reach:
                        break
                    cost = hop_cost + (target - position) * per_block
                    total = best[i] + cost
                    if total < best[j] - 1e-12:
                        best[j] = total
                        back[j] = (i, server, score, cost)

        end_index = index_of[n_blocks]
        if best[end_index] == infinity:
            return None

        hops: list[Hop] = []
        cursor = end_index
        while cursor != index_of[0]:
            entry = back[cursor]
            if entry is None:  # pragma: no cover - unreachable when best[] is finite
                return None
            previous, server, score, cost = entry
            hops.append(
                Hop(
                    server=server,
                    covers=BlockRange(model_id, positions[previous], positions[cursor]),
                    score=score,
                    cost=cost,
                )
            )
            cursor = previous
        hops.reverse()

        return Route(model_id=model_id, hops=tuple(hops), total_cost=best[end_index])

    def alternates(
        self,
        needed: BlockRange,
        candidates: Iterable[ServerInfo],
        *,
        exclude: Optional[set[PeerId]] = None,
    ) -> list[ServerInfo]:
        """Replacement servers covering ``needed``, best first.

        Used for mid-generation failover when a volunteer closes their laptop -- the churn
        case the whole design assumes is normal.
        """
        exclude = exclude or set()
        matches = [
            server
            for server in candidates
            if server.block_range.covers(needed)
            and server.peer_id not in exclude
            and not self.is_excluded(server.peer_id)
        ]
        matches.sort(key=lambda s: (-self.effective_score(s.peer_id), s.peer_id))
        return matches[: self.config.max_alternates]

    def rank(self, candidates: Sequence[ServerInfo]) -> list[tuple[ServerInfo, float]]:
        ranked = [
            (server, self.effective_score(server.peer_id))
            for server in candidates
            if not self.is_excluded(server.peer_id)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].peer_id))
        return ranked
