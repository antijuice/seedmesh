"""Turning many strangers' claims into one number, without letting strangers control it.

Sharing reputation is what makes the swarm's experience collective instead of every client
relearning from scratch. It is also the attack surface: once you accept input from peers
you have never met, you have built a voting system, and voting systems get stuffed.

The defences, in the order they apply:

1. **One observer, one vote.** Influence is capped per observer regardless of how many
   reports it files. Filing ten thousand reports buys nothing over filing one.
2. **Per-cluster caps.** Since peer ids are free but network diversity is not, weight is
   capped per network cluster (see :mod:`seedmesh.reputation.diversity`). A thousand
   sybils in one ASN carry one cluster's worth of influence between them.
3. **Weighted median, not weighted mean.** A mean can be dragged anywhere by extreme
   values; a median has a 50% breakdown point, so an attacker must control a majority of
   *weight* -- which rule 2 has already made expensive -- before they can move the result.
4. **Collusion discount.** An observer vouching for a subject in its own network cluster is
   discounted heavily: that is the shape of self-dealing.
5. **Corroboration for accusations.** A mismatch claim is only believed when it comes from
   several independent clusters. Otherwise defaming an honest competitor would be as cheap
   as praising yourself, which is strictly worse -- it gives attackers a way to evict good
   nodes.
6. **First-hand experience wins.** :func:`blend` always lets what you measured yourself
   dominate what you were told, in proportion to how much you have measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from seedmesh.core.types import PeerId
from seedmesh.reputation.diversity import UNKNOWN_CLUSTER
from seedmesh.reputation.records import ObservationBatch
from seedmesh.reputation.scorer import ScoreBreakdown

ClusterLookup = Callable[[PeerId], str]


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    max_weight_per_observer: float = 1.0
    max_weight_per_cluster: float = 3.0
    unknown_cluster_weight: float = 0.5
    """Total weight shared by every peer whose network location is unknown. Lower than a
    normal cluster because the bucket cannot be checked for diversity at all."""

    collusion_discount: float = 0.1
    min_clusters_for_confidence: int = 3
    min_clusters_for_mismatch: int = 2
    prior_strength: float = 4.0
    prior_success_rate: float = 0.80

    def __post_init__(self) -> None:
        if self.max_weight_per_cluster < self.max_weight_per_observer:
            raise ValueError("per-cluster cap must be at least the per-observer cap")
        if not 0.0 <= self.collusion_discount <= 1.0:
            raise ValueError("collusion_discount must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AggregateScore:
    """Swarm-wide belief about one subject, with the evidence that produced it."""

    subject: PeerId
    reliability: float
    total_weight: float
    distinct_clusters: int
    observer_count: int
    mismatch_clusters: int
    mismatch_confirmed: bool
    confidence: float

    @property
    def is_trustworthy_signal(self) -> bool:
        """Whether this aggregate rests on enough independent evidence to act on alone."""
        return self.distinct_clusters >= 2 and self.total_weight > 0


def weighted_median(pairs: Sequence[tuple[float, float]]) -> float:
    """Median of ``(value, weight)`` pairs. Returns 0.0 for empty input.

    Chosen over a mean for its 50% breakdown point -- see module docstring, rule 3.
    """
    usable = [(v, w) for v, w in pairs if w > 0]
    if not usable:
        return 0.0
    usable.sort(key=lambda item: item[0])
    total = sum(w for _, w in usable)
    target = total / 2.0
    cumulative = 0.0
    for value, weight in usable:
        cumulative += weight
        if cumulative >= target:
            return value
    return usable[-1][0]


@dataclass(slots=True)
class _Claim:
    observer: PeerId
    cluster: str
    estimate: float
    weight: float
    claims_mismatch: bool


class ReputationAggregator:
    """Combines verified :class:`ObservationBatch` records into per-subject scores.

    Input must already be signature-verified -- pass batches through
    :func:`seedmesh.reputation.records.verify_batch` first. This class assumes
    attribution is settled and worries only about weighting.
    """

    def __init__(
        self,
        cluster_lookup: ClusterLookup,
        config: Optional[AggregationConfig] = None,
        *,
        observer_weight: Optional[Callable[[PeerId], float]] = None,
    ) -> None:
        self.config = config or AggregationConfig()
        self._cluster_of = cluster_lookup
        self._observer_weight = observer_weight or (lambda _peer: 1.0)
        self._batches: dict[PeerId, ObservationBatch] = {}

    def ingest(self, batch: ObservationBatch) -> None:
        """Accept a verified batch, superseding any earlier batch from the same observer.

        Keeping only the newest batch per observer is itself a defence: an observer cannot
        accumulate influence by publishing many batches, only refresh the one it has.
        """
        existing = self._batches.get(batch.observer)
        if existing is None or batch.epoch > existing.epoch:
            self._batches[batch.observer] = batch

    def ingest_many(self, batches: Iterable[ObservationBatch]) -> None:
        for batch in batches:
            self.ingest(batch)

    def drop_expired(self, now: float) -> int:
        stale = [obs for obs, batch in self._batches.items() if batch.is_expired(now)]
        for observer in stale:
            del self._batches[observer]
        return len(stale)

    def subjects(self) -> set[PeerId]:
        return {report.subject for batch in self._batches.values() for report in batch.reports}

    def _collect_claims(self, subject: PeerId) -> list[_Claim]:
        cfg = self.config
        subject_cluster = self._cluster_of(subject)
        claims: list[_Claim] = []

        for observer, batch in self._batches.items():
            if observer == subject:
                continue  # defence in depth; verify_batch already rejects self-reports
            report = batch.report_for(subject)
            if report is None:
                continue

            attempts = report.attempts
            if attempts <= 0 and report.mismatches <= 0:
                continue

            estimate = (cfg.prior_strength * cfg.prior_success_rate + report.ok) / (
                cfg.prior_strength + attempts
            )

            weight = min(cfg.max_weight_per_observer, max(0.0, self._observer_weight(observer)))
            observer_cluster = self._cluster_of(observer)
            if observer_cluster == subject_cluster and observer_cluster != UNKNOWN_CLUSTER:
                weight *= cfg.collusion_discount

            if weight <= 0:
                continue

            claims.append(
                _Claim(
                    observer=observer,
                    cluster=observer_cluster,
                    estimate=estimate,
                    weight=weight,
                    claims_mismatch=report.mismatches > 0,
                )
            )

        return self._apply_cluster_caps(claims)

    def _apply_cluster_caps(self, claims: list[_Claim]) -> list[_Claim]:
        """Scale down each cluster's claims so no cluster exceeds its weight budget."""
        cfg = self.config
        totals: dict[str, float] = {}
        for claim in claims:
            totals[claim.cluster] = totals.get(claim.cluster, 0.0) + claim.weight

        for claim in claims:
            budget = (
                cfg.unknown_cluster_weight
                if claim.cluster == UNKNOWN_CLUSTER
                else cfg.max_weight_per_cluster
            )
            total = totals[claim.cluster]
            if total > budget:
                claim.weight *= budget / total

        return [claim for claim in claims if claim.weight > 0]

    def aggregate(self, subject: PeerId) -> AggregateScore:
        cfg = self.config
        claims = self._collect_claims(subject)

        if not claims:
            return AggregateScore(
                subject=subject,
                reliability=cfg.prior_success_rate,
                total_weight=0.0,
                distinct_clusters=0,
                observer_count=0,
                mismatch_clusters=0,
                mismatch_confirmed=False,
                confidence=0.0,
            )

        reliability = weighted_median([(c.estimate, c.weight) for c in claims])
        total_weight = sum(c.weight for c in claims)
        clusters = {c.cluster for c in claims}
        mismatch_clusters = {c.cluster for c in claims if c.claims_mismatch}

        confidence = min(1.0, len(clusters) / cfg.min_clusters_for_confidence) * min(
            1.0, total_weight / cfg.max_weight_per_observer
        )

        return AggregateScore(
            subject=subject,
            reliability=reliability,
            total_weight=total_weight,
            distinct_clusters=len(clusters),
            observer_count=len(claims),
            mismatch_clusters=len(mismatch_clusters),
            # An accusation needs independent corroboration; see module docstring, rule 5.
            mismatch_confirmed=len(mismatch_clusters) >= cfg.min_clusters_for_mismatch,
            confidence=confidence,
        )

    def aggregate_all(self) -> dict[PeerId, AggregateScore]:
        return {subject: self.aggregate(subject) for subject in self.subjects()}


def blend(
    local: ScoreBreakdown,
    remote: Optional[AggregateScore],
    *,
    max_remote_influence: float = 0.5,
) -> float:
    """Combine first-hand measurement with swarm hearsay.

    Weighting is driven by *local* confidence: with no first-hand history the swarm's view
    is worth listening to, and as your own evidence accumulates it takes over. Hearsay is
    capped at ``max_remote_influence`` even at zero local confidence, so a client that has
    just joined cannot be fully steered by whatever the network tells it.

    A confirmed mismatch is handled outside this arithmetic. Integrity is a gate, not a
    term to be averaged away by a chorus of favourable reports.
    """
    if remote is None or not remote.is_trustworthy_signal:
        return local.score

    remote_share = max_remote_influence * (1.0 - local.confidence) * remote.confidence
    blended = (1.0 - remote_share) * local.score + remote_share * remote.reliability

    if remote.mismatch_confirmed:
        blended = min(blended, 0.1)

    return max(0.0, min(1.0, blended))
