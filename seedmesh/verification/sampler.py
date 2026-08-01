"""Choosing which requests to double-check, and who checks them.

Spec section 3.2 says to verify a flat 2-5% of requests. Section 9 then flags the hole in
its own proposal: picking "any two servers hosting this block range" is trivially defeated
when one operator controls both. This module implements the sampling and closes that hole.

**Sampling is adaptive, not flat.** A flat rate spends the same effort on a peer with ten
thousand clean results as on one that joined a minute ago, which is backwards. Rate rises
when confidence is low (a new peer earns scrutiny until it has a record) and when integrity
is already shaky, and decays toward the base rate as evidence accumulates. Same total
overhead, aimed where it discriminates.

**The verifier must be independent of the subject.** Selection requires a different network
cluster and a first-seen gap, so an operator cannot verify themselves by spinning up a
second node. See :mod:`seedmesh.reputation.diversity`.

**Pairings rotate.** Always choosing the highest-scoring verifier would produce stable,
predictable pairs -- an attacker's dream, since a stable pair is a collusion relationship
worth buying. Selection prefers under-used pairings among eligible candidates.

One residual risk is documented rather than solved here: a colluding pair with a side
channel can notice they received the same nonce and agree on a fabricated answer. The
defence is that the diversity constraint makes forming such a pair expensive, not that it
is impossible. See ``docs/threat-model.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from seedmesh.core.types import BlockRange, PeerId, ServerInfo, Verdict
from seedmesh.reputation.diversity import ClusterIndex
from seedmesh.reputation.scorer import ReputationScorer
from seedmesh.verification.calibrate import ToleranceTable
from seedmesh.verification.compare import Tolerance


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    base_rate: float = 0.03
    """Steady-state verification rate for well-established peers (spec: 2-5%)."""

    unproven_rate: float = 0.25
    """Rate applied to peers with essentially no history."""

    suspicion_rate: float = 0.60
    """Rate for peers with damaged integrity -- confirm or clear them quickly."""

    suspicion_integrity: float = 0.9
    max_rate: float = 0.75
    min_first_seen_gap_s: float = 900.0
    """Peers that first appeared within this window of each other are treated as related.
    Simultaneous arrival is the signature of one operator starting a fleet from a script."""

    require_distinct_cluster: bool = True

    def __post_init__(self) -> None:
        for name in ("base_rate", "unproven_rate", "suspicion_rate", "max_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1]")


@dataclass
class PairHistory:
    """How often each (subject, verifier) pairing has been used, and how it went.

    Serves two purposes: rotating pairings so no relationship becomes predictable, and
    exposing pairs that agree suspiciously often relative to the subject's record with
    everyone else.
    """

    counts: dict[tuple[PeerId, PeerId], int] = field(default_factory=dict)
    agreements: dict[tuple[PeerId, PeerId], int] = field(default_factory=dict)

    @staticmethod
    def _key(subject: PeerId, verifier: PeerId) -> tuple[PeerId, PeerId]:
        return (subject, verifier) if subject <= verifier else (verifier, subject)

    def record_pairing(self, subject: PeerId, verifier: PeerId) -> None:
        key = self._key(subject, verifier)
        self.counts[key] = self.counts.get(key, 0) + 1

    def record_verdict(self, subject: PeerId, verifier: PeerId, verdict: Verdict) -> None:
        if verdict is Verdict.MATCH:
            key = self._key(subject, verifier)
            self.agreements[key] = self.agreements.get(key, 0) + 1

    def pairing_count(self, subject: PeerId, verifier: PeerId) -> int:
        return self.counts.get(self._key(subject, verifier), 0)

    def partners_of(self, peer: PeerId) -> set[PeerId]:
        partners: set[PeerId] = set()
        for a, b in self.counts:
            if a == peer:
                partners.add(b)
            elif b == peer:
                partners.add(a)
        return partners


@dataclass(frozen=True, slots=True)
class VerificationTask:
    """A decision to double-route one sub-computation.

    Carries both parties' network clusters, resolved at planning time, because that is what
    the scorer attributes a disagreement to -- see
    :meth:`ReputationScorer.record_verification`.
    """

    subject: PeerId
    verifier: ServerInfo
    block_range: BlockRange
    nonce: bytes
    subject_cluster: str
    verifier_cluster: str
    tolerance: Optional[Tolerance] = None
    """Tolerance for *this specific pair of compute profiles*. Never a global constant --
    see :class:`~seedmesh.verification.calibrate.ToleranceTable`."""


class VerificationSampler:
    """Decides when to verify and picks an independent verifier."""

    def __init__(
        self,
        scorer: ReputationScorer,
        clusters: ClusterIndex,
        config: Optional[SamplerConfig] = None,
        *,
        rng: Optional[random.Random] = None,
        history: Optional[PairHistory] = None,
        tolerances: Optional[ToleranceTable] = None,
    ) -> None:
        self.scorer = scorer
        self.clusters = clusters
        self.config = config or SamplerConfig()
        self.rng = rng or random.Random()
        self.history = history or PairHistory()
        self.tolerances = tolerances

    def _tolerance_for(self, subject: ServerInfo, verifier: ServerInfo) -> Optional[Tolerance]:
        """Tolerance for this pair, or ``None`` if the pair cannot be judged.

        With no table configured, every pair is comparable and the caller supplies a global
        tolerance -- the pre-measurement behaviour, kept so existing callers still work.
        """
        if self.tolerances is None:
            return None
        return self.tolerances.get(subject.profile.key, verifier.profile.key)

    def verification_rate(self, peer_id: PeerId) -> float:
        """Probability that a given request to this peer gets double-checked."""
        cfg = self.config
        breakdown = self.scorer.breakdown(peer_id)

        if breakdown.integrity < cfg.suspicion_integrity:
            return min(cfg.max_rate, cfg.suspicion_rate)

        # Interpolate from the unproven rate down to the base rate as confidence grows.
        rate = cfg.unproven_rate + (cfg.base_rate - cfg.unproven_rate) * breakdown.confidence
        return max(0.0, min(cfg.max_rate, rate))

    def should_verify(self, peer_id: PeerId) -> bool:
        return self.rng.random() < self.verification_rate(peer_id)

    def eligible_verifiers(
        self,
        subject: ServerInfo,
        needed: BlockRange,
        candidates: Iterable[ServerInfo],
    ) -> list[ServerInfo]:
        """Candidates that can check ``subject`` on ``needed`` and are plausibly unrelated."""
        cfg = self.config
        eligible: list[ServerInfo] = []
        for candidate in candidates:
            if candidate.peer_id == subject.peer_id:
                continue
            if not candidate.block_range.covers(needed):
                continue
            if self.scorer.breakdown(candidate.peer_id).is_quarantined:
                continue
            if cfg.require_distinct_cluster and not self.clusters.are_independent(
                subject, candidate, min_first_seen_gap_s=cfg.min_first_seen_gap_s
            ):
                continue
            if self.tolerances is not None and not self.tolerances.can_compare(
                subject.profile.key, candidate.profile.key
            ):
                # No calibration for this pair of compute profiles, so any verdict would be
                # judged against a made-up threshold. Refusing is the same call the sampler
                # already makes when no independent verifier exists.
                continue
            eligible.append(candidate)
        return eligible

    def select_verifier(
        self,
        subject: ServerInfo,
        needed: BlockRange,
        candidates: Iterable[ServerInfo],
    ) -> Optional[ServerInfo]:
        """Pick a verifier, preferring under-used pairings among capable independent peers.

        Returns ``None`` when no independent verifier exists. That is a real outcome, not a
        failure to be worked around: verifying against a peer that might be the subject's
        own sock puppet is worse than not verifying, because it manufactures evidence of
        integrity that the reputation layer will then trust.
        """
        eligible = self.eligible_verifiers(subject, needed, candidates)
        if not eligible:
            return None

        def rank(candidate: ServerInfo) -> tuple[int, int, float, str]:
            # Same-profile pairs first: their honest disagreement is ~10x tighter, so they
            # discriminate a cheat from noise far more sharply than a cross-profile pair
            # whose tolerance has to be wide enough to accommodate NF4.
            cross_profile = int(candidate.profile.key != subject.profile.key)
            pairings = self.history.pairing_count(subject.peer_id, candidate.peer_id)
            score = self.scorer.breakdown(candidate.peer_id).score
            return (cross_profile, pairings, -score, candidate.peer_id)

        eligible.sort(key=rank)
        # Break ties randomly among the equally-preferred front so pairings stay
        # unpredictable even when scores are identical.
        best_cross, best_pairings = rank(eligible[0])[:2]
        front = [
            c
            for c in eligible
            if int(c.profile.key != subject.profile.key) == best_cross
            and self.history.pairing_count(subject.peer_id, c.peer_id) == best_pairings
        ]
        return self.rng.choice(front) if len(front) > 1 else eligible[0]

    def plan(
        self,
        subject: ServerInfo,
        needed: BlockRange,
        candidates: Iterable[ServerInfo],
        *,
        force: bool = False,
    ) -> Optional[VerificationTask]:
        """Decide whether to verify this request, and against whom."""
        if not force and not self.should_verify(subject.peer_id):
            return None
        verifier = self.select_verifier(subject, needed, candidates)
        if verifier is None:
            return None
        self.history.record_pairing(subject.peer_id, verifier.peer_id)
        nonce = self.rng.getrandbits(128).to_bytes(16, "big")
        return VerificationTask(
            subject=subject.peer_id,
            verifier=verifier,
            block_range=needed,
            nonce=nonce,
            subject_cluster=self.clusters.coarse_of(subject),
            verifier_cluster=self.clusters.coarse_of(verifier),
            tolerance=self._tolerance_for(subject, verifier),
        )

    def record_outcome(
        self,
        task: VerificationTask,
        verdict: Verdict,
        *,
        comparable: bool = True,
    ) -> None:
        """Feed a verdict back into reputation and pairing history.

        A MISMATCH proves that *one of the two* peers is wrong -- it does not say which.
        Charging only the subject would let a malicious verifier defame honest servers at
        will, so the disagreement is recorded against both -- but *attributed to the
        partner it happened with*, so neither is convicted on one accusation.

        That attribution is what separates the cheat from the honest peer that caught it:
        a cheat disagrees with verifiers from every independent cluster it meets, while its
        accuser disagrees only with the cheat's cluster and matches everyone else.
        Attribution is by *cluster*, so a sybil fleet cannot manufacture accusers -- see
        :meth:`ReputationScorer.record_verification`.
        """
        self.history.record_verdict(task.subject, task.verifier.peer_id, verdict)

        if verdict is Verdict.INCONCLUSIVE:
            # An ambiguous-band result is weak evidence about both peers and accumulates.
            # A structural failure to compare is evidence about neither.
            if comparable:
                self.scorer.record_ambiguous(task.subject, task.verifier_cluster)
                self.scorer.record_ambiguous(task.verifier.peer_id, task.subject_cluster)
            return

        self.scorer.record_verification(
            task.subject, verdict, partner_cluster=task.verifier_cluster
        )
        self.scorer.record_verification(
            task.verifier.peer_id, verdict, partner_cluster=task.subject_cluster
        )
