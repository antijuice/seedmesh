"""Per-peer reliability scoring with time decay.

Upstream Petals ranks servers by *self-reported* throughput. That is the gap this module
closes: a score built only from what the observer measured itself.

Three design choices worth stating outright.

**Reliability is a Beta posterior, not a success ratio.** A naive ``ok / (ok + fail)``
gives a server with one lucky success a perfect 1.0, ranking it above a server with 999/1000.
Scoring against a prior means a new peer starts near the prior and earns its way to the
extremes, so confidence and score are separate quantities.

**Integrity is separate from reliability and decays far more slowly.** Being slow or flaky
is forgivable and often not the operator's fault; returning results that fail verification
is not. Folding both into one number would let a caught node launder a proven mismatch by
serving a burst of cheap successful requests. Integrity multiplies the score, so it acts
as a gate rather than a term.

**Latency modulates but never dominates.** The latency factor is bounded to
``[1 - latency_weight, 1]``, which encodes the priority ordering that matters for a
trust layer: a fast liar must always rank below a slow honest node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from seedmesh.core.clock import Clock, SystemClock
from seedmesh.core.types import BlockRange, Observation, PeerId, Verdict


@dataclass(frozen=True, slots=True)
class ScorerConfig:
    """Tunables for :class:`ReputationScorer`.

    Defaults are starting points, not measurements. The half-lives in particular should be
    re-derived from real churn data once the swarm has any (spec milestone M4).
    """

    reliability_half_life_s: float = 6 * 3600.0
    """How fast ordinary success/failure history fades. Six hours means a node that fixes
    its connection is competitive again within a day, without forgetting a bad afternoon."""

    integrity_half_life_s: float = 30 * 24 * 3600.0
    """Verification history fades ~120x slower than reliability. A proven mismatch should
    still be visible a month later."""

    prior_strength: float = 4.0
    """Pseudo-observations of the prior. Higher = newcomers regress harder to the mean."""

    prior_success_rate: float = 0.80
    """What we assume of a peer we have never used. Deliberately below 1.0 so an unknown
    peer does not outrank a proven one, and above 0.5 so newcomers still get tried."""

    integrity_prior_strength: float = 2.0

    mismatch_penalty: float = 40.0
    """Weight of a fully corroborated mismatch, in units of verification failures. One
    proven bad result costs roughly as much as 40 clean ones earn."""

    single_dispute_weight: float = 0.15
    """Weight of a disagreement seen with only *one* partner, as a fraction of
    ``mismatch_penalty``.

    A pairwise mismatch proves that one of the two peers is wrong. It does not say which.
    Charging both at full weight -- the obvious implementation -- evicts the honest peer
    that just caught a cheat, which makes verifying dangerous and is strictly worse than
    not verifying at all. So an uncorroborated dispute only dents the score; guilt is
    established by disagreeing with *several independent partners*, which is a pattern an
    honest peer does not produce."""

    corroborated_dispute_weight: float = 1.0
    strong_dispute_weight: float = 1.5

    ambiguous_dispute_weight: float = 0.25
    """Dispute weight contributed by one *ambiguous-band* comparison, once the rate gate
    below has opened."""

    ambiguous_rate_threshold: float = 0.25
    min_ambiguous_for_evidence: float = 2.0
    """Ambiguous results only become evidence when they are *disproportionate*, not merely
    numerous.

    This closes the gap a threshold alone leaves open: an attacker's best play is to
    corrupt results by just under ``mismatch_distance``, which evades any single-sample
    test by construction. Accumulating weak evidence defeats that -- but accumulating it in
    absolute terms creates a worse problem, which simulation demonstrated. Sybil disputes
    dent an honest node's integrity, its verification rate jumps in response, the extra
    sampling turns up the rare honest-vs-honest ambiguous result, and the node convicts
    itself. Scrutiny became self-fulfilling.

    Gating on *rate* breaks that loop. Calibration puts an honest peer's ambiguous rate
    near the target false-positive rate (~0.1%), so a peer sitting above 25% is not having
    bad luck no matter how many times it has been sampled -- and sampling an honest peer
    more often no longer pushes it toward conviction, because the denominator grows too.

    Swept over the ``healthy``, ``mixed_threat`` and ``sybil`` presets, 5 seeds x 400
    requests each. ``min_ambiguous_for_evidence`` is the binding constraint -- raising it
    to 3 starves detection of the sybil fleet, because 12 identities split the sampling
    budget for one block range between them:

        weight  min  rate | sybils caught  false quarantines (all presets)
          0.25  3.0  0.15        80%                 0
          0.15  2.0  0.15        80%                 0
          0.25  2.0  0.25       100%                 0
          0.34  2.0  0.30       100%                 0

    The chosen row is the most conservative gate that still reaches full detection."""

    min_disputing_partners: int = 2
    """Distinct partners that must disagree with a peer before it can be quarantined."""

    quarantine_integrity: float = 0.5

    latency_reference_ms: float = 500.0
    """p95 latency at which the latency factor sits halfway."""

    latency_weight: float = 0.25
    """Maximum fraction of the score latency may remove. Bounds the fast-liar problem."""

    max_latency_samples: int = 256

    def __post_init__(self) -> None:
        if not 0.0 <= self.latency_weight < 1.0:
            raise ValueError("latency_weight must be in [0, 1)")
        if self.prior_strength <= 0 or self.integrity_prior_strength <= 0:
            raise ValueError("prior strengths must be positive")
        if self.reliability_half_life_s <= 0 or self.integrity_half_life_s <= 0:
            raise ValueError("half-lives must be positive")


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """An explainable score.

    Volunteers will ask "why is my node ranked below that one". A single opaque float is a
    bad answer for a community-run network, so every component is surfaced and the monitor
    page renders them.
    """

    peer_id: PeerId
    score: float
    reliability: float
    integrity: float
    latency_factor: float
    confidence: float
    observations: float
    p50_ms: float
    p95_ms: float
    confirmed_mismatches: float
    disputing_partners: int = 0
    """Distinct independent peers that have disagreed with this one."""

    direct_faults: float = 0.0
    """Unilaterally proven faults, e.g. returning the input unchanged. Needs no
    corroboration: the client witnessed it directly rather than inferring it from a
    disagreement between two parties."""

    quarantined: bool = False

    @property
    def is_quarantined(self) -> bool:
        """Whether this peer should be excluded from routing outright.

        Requires corroboration, not just a low score. See
        :attr:`ScorerConfig.single_dispute_weight` for why a lone disagreement must never
        be enough to evict anyone.
        """
        return self.quarantined


class _DecayedQuantiles:
    """Bounded, exponentially-decayed sample of latencies.

    A plain rolling window would either keep unbounded memory per peer (a DoS vector when
    peers are free to mint) or forget too abruptly. This keeps at most ``capacity``
    weighted samples, decays their weights with the same half-life as reliability, and
    evicts the lowest-weight sample when full -- so old measurements fade rather than
    falling off a cliff.
    """

    __slots__ = ("_samples", "_capacity", "_half_life", "_last_decay")

    def __init__(self, capacity: int, half_life_s: float, now: float) -> None:
        self._samples: list[list[float]] = []  # [value, weight]
        self._capacity = capacity
        self._half_life = half_life_s
        self._last_decay = now

    def _decay_to(self, now: float) -> None:
        elapsed = now - self._last_decay
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self._half_life)
        for sample in self._samples:
            sample[1] *= factor
        self._last_decay = now
        # Drop samples that have decayed into irrelevance rather than carrying them forever.
        self._samples = [s for s in self._samples if s[1] > 1e-6]

    def add(self, value: float, now: float) -> None:
        self._decay_to(now)
        self._samples.append([float(value), 1.0])
        if len(self._samples) > self._capacity:
            weakest = min(range(len(self._samples)), key=lambda i: self._samples[i][1])
            self._samples.pop(weakest)

    def quantile(self, q: float, now: float) -> float:
        self._decay_to(now)
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples, key=lambda s: s[0])
        total = sum(s[1] for s in ordered)
        if total <= 0:
            return ordered[len(ordered) // 2][0]
        target = q * total
        cumulative = 0.0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= target:
                return value
        return ordered[-1][0]

    def __len__(self) -> int:
        return len(self._samples)


@dataclass
class _PeerState:
    success: float = 0.0
    failure: float = 0.0
    integrity_pass: float = 0.0
    direct_faults: float = 0.0
    confirmed_mismatches: float = 0.0
    observations: float = 0.0
    last_reliability_decay: float = 0.0
    last_integrity_decay: float = 0.0
    latencies: Optional[_DecayedQuantiles] = None
    ranges: set[BlockRange] = field(default_factory=set)
    ambiguous: dict[str, float] = field(default_factory=dict)
    """Ambiguous-band results, keyed by partner cluster. Held apart from ``disputes``
    because they only become evidence once the rate gate opens."""

    comparisons: float = 0.0
    """Total comparable verifications, the denominator for the ambiguous rate."""

    disputes: dict[str, float] = field(default_factory=dict)
    """Decayed disagreement counts, keyed by the *network cluster* of the partner.

    Keying by cluster rather than by peer id is not a detail -- it is what makes the
    corroboration rule survive a sybil fleet. Keyed by peer id, an operator running twelve
    identities manufactures twelve distinct accusers, so the honest node that catches them
    looks like a serial disagreer and gets convicted while each individual sybil looks
    merely unlucky. Keyed by cluster, that whole fleet counts once, and the honest node's
    accusers collapse to the single cluster they actually came from. Simulation confirmed
    both directions: see ``docs/threat-model.md``."""


class ReputationScorer:
    """Accumulates first-hand observations into per-peer scores.

    This class holds *local* belief only -- what this node measured itself. Merging other
    peers' beliefs is :mod:`seedmesh.reputation.aggregate`'s job, and is kept separate
    because the two have completely different trust assumptions: everything recorded here
    was witnessed directly, everything there was asserted by a stranger.
    """

    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.config = config or ScorerConfig()
        self.clock: Clock = clock or SystemClock()
        self._peers: dict[PeerId, _PeerState] = {}

    # ---- recording -------------------------------------------------------------

    def _state(self, peer_id: PeerId) -> _PeerState:
        state = self._peers.get(peer_id)
        if state is None:
            now = self.clock.now()
            state = _PeerState(
                last_reliability_decay=now,
                last_integrity_decay=now,
                latencies=_DecayedQuantiles(
                    self.config.max_latency_samples,
                    self.config.reliability_half_life_s,
                    now,
                ),
            )
            self._peers[peer_id] = state
        return state

    def _decay_reliability(self, state: _PeerState, now: float) -> None:
        elapsed = now - state.last_reliability_decay
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self.config.reliability_half_life_s)
        state.success *= factor
        state.failure *= factor
        state.observations *= factor
        state.last_reliability_decay = now

    def _decay_integrity(self, state: _PeerState, now: float) -> None:
        elapsed = now - state.last_integrity_decay
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self.config.integrity_half_life_s)
        state.integrity_pass *= factor
        state.direct_faults *= factor
        state.confirmed_mismatches *= factor
        state.comparisons *= factor
        for bucket in (state.disputes, state.ambiguous):
            for partner in list(bucket):
                decayed = bucket[partner] * factor
                if decayed < 1e-4:
                    del bucket[partner]
                else:
                    bucket[partner] = decayed
        state.last_integrity_decay = now

    def _combined_disputes(self, state: _PeerState) -> dict[str, float]:
        """Hard disputes, plus ambiguous results if this peer's ambiguous rate is anomalous."""
        cfg = self.config
        combined = dict(state.disputes)

        ambiguous_total = sum(state.ambiguous.values())
        rate = ambiguous_total / state.comparisons if state.comparisons > 0 else 0.0
        if (
            ambiguous_total >= cfg.min_ambiguous_for_evidence
            and rate >= cfg.ambiguous_rate_threshold
        ):
            for cluster, count in state.ambiguous.items():
                combined[cluster] = (
                    combined.get(cluster, 0.0) + count * cfg.ambiguous_dispute_weight
                )
        return combined

    def _dispute_load(self, state: _PeerState) -> tuple[float, int]:
        """Effective failure weight from disputes, and how many distinct clusters raised them.

        The escalation ladder is the corroboration rule: one accusing cluster dents the
        score, two independent clusters establish guilt, three or more make it emphatic.
        """
        cfg = self.config
        combined = self._combined_disputes(state)
        total = sum(combined.values())
        distinct = sum(1 for weight in combined.values() if weight >= 0.5)

        if distinct == 0:
            escalation = 0.0
        elif distinct == 1:
            escalation = cfg.single_dispute_weight
        elif distinct < cfg.min_disputing_partners + 1:
            escalation = cfg.corroborated_dispute_weight
        else:
            escalation = cfg.strong_dispute_weight

        return total * escalation, distinct

    def record(self, observation: Observation) -> None:
        """Record one first-hand request outcome."""
        now = max(observation.at, self.clock.now())
        state = self._state(observation.subject)
        self._decay_reliability(state, now)
        state.ranges.add(observation.block_range)

        if observation.outcome.is_success:
            state.success += 1.0
            state.observations += 1.0
            state.latencies.add(observation.latency_ms, now)
        elif observation.outcome.counts_against_reliability:
            state.failure += 1.0
            state.observations += 1.0
        else:
            # REFUSED: honest backpressure. Neither rewarded nor punished; see
            # Outcome.counts_against_reliability for why punishing it is harmful.
            pass

    def record_many(self, observations: Iterable[Observation]) -> None:
        for observation in observations:
            self.record(observation)

    def record_verification(
        self,
        peer_id: PeerId,
        verdict: Verdict,
        *,
        partner_cluster: Optional[str] = None,
        at: Optional[float] = None,
    ) -> None:
        """Record the result of a redundant-execution check against a peer in ``partner_cluster``.

        ``INCONCLUSIVE`` is recorded as nothing at all. It means the check could not
        decide -- the comparison peer vanished, or the distance landed in the ambiguous
        band -- and inferring guilt from an inconclusive test is how honest nodes get
        driven off a volunteer network.

        A ``MISMATCH`` with no ``partner_cluster`` is a *direct* fault: something the client
        proved by itself, such as a server echoing its own input. That needs no corroboration.
        """
        now = at if at is not None else self.clock.now()
        state = self._state(peer_id)
        self._decay_integrity(state, now)

        if verdict is Verdict.MATCH:
            state.integrity_pass += 1.0
            state.comparisons += 1.0
        elif verdict is Verdict.MISMATCH:
            state.confirmed_mismatches += 1.0
            state.comparisons += 1.0
            if partner_cluster is None:
                state.direct_faults += 1.0
            else:
                state.disputes[partner_cluster] = (
                    state.disputes.get(partner_cluster, 0.0) + 1.0
                )

    def record_direct_fault(self, peer_id: PeerId, *, at: Optional[float] = None) -> None:
        """Record a fault the client witnessed itself, needing no second opinion."""
        self.record_verification(peer_id, Verdict.MISMATCH, partner_cluster=None, at=at)

    def record_ambiguous(
        self,
        peer_id: PeerId,
        partner_cluster: str,
        *,
        at: Optional[float] = None,
    ) -> None:
        """Record a comparison that landed in the ambiguous band.

        Contributes a fraction of a dispute rather than a whole one, so no single
        ambiguous result can convict, but a pattern of them across independent clusters
        eventually does. See :attr:`ScorerConfig.ambiguous_dispute_weight`.

        Callers must not route *structural* inconclusives here -- a dimension mismatch says
        nothing about the peer. See :attr:`seedmesh.verification.compare.Comparison.comparable`.
        """
        now = at if at is not None else self.clock.now()
        state = self._state(peer_id)
        self._decay_integrity(state, now)
        state.ambiguous[partner_cluster] = state.ambiguous.get(partner_cluster, 0.0) + 1.0
        state.comparisons += 1.0

    # ---- scoring ---------------------------------------------------------------

    def breakdown(self, peer_id: PeerId) -> ScoreBreakdown:
        """Full explainable score for one peer."""
        cfg = self.config
        now = self.clock.now()
        state = self._peers.get(peer_id)

        if state is None:
            return ScoreBreakdown(
                peer_id=peer_id,
                score=cfg.prior_success_rate,
                reliability=cfg.prior_success_rate,
                integrity=1.0,
                latency_factor=1.0,
                confidence=0.0,
                observations=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                confirmed_mismatches=0.0,
                disputing_partners=0,
                direct_faults=0.0,
                quarantined=False,
            )

        self._decay_reliability(state, now)
        self._decay_integrity(state, now)

        prior_mass = cfg.prior_strength
        reliability = (prior_mass * cfg.prior_success_rate + state.success) / (
            prior_mass + state.success + state.failure
        )

        dispute_load, disputing_partners = self._dispute_load(state)
        effective_fail = cfg.mismatch_penalty * (state.direct_faults + dispute_load)
        integrity = (cfg.integrity_prior_strength + state.integrity_pass) / (
            cfg.integrity_prior_strength + state.integrity_pass + effective_fail
        )

        # Exclusion needs corroboration from independent partners, or a fault the client
        # proved on its own. A low score alone is a routing preference, not an eviction.
        corroborated = (
            disputing_partners >= cfg.min_disputing_partners or state.direct_faults >= 1.0
        )
        quarantined = integrity < cfg.quarantine_integrity and corroborated

        p50 = state.latencies.quantile(0.50, now)
        p95 = state.latencies.quantile(0.95, now)
        raw_latency = cfg.latency_reference_ms / (cfg.latency_reference_ms + p95) if p95 > 0 else 1.0
        latency_factor = 1.0 - cfg.latency_weight * (1.0 - raw_latency)

        confidence = state.observations / (state.observations + cfg.prior_strength)
        score = integrity * reliability * latency_factor

        return ScoreBreakdown(
            peer_id=peer_id,
            score=score,
            reliability=reliability,
            integrity=integrity,
            latency_factor=latency_factor,
            confidence=confidence,
            observations=state.observations,
            p50_ms=p50,
            p95_ms=p95,
            confirmed_mismatches=state.confirmed_mismatches,
            disputing_partners=disputing_partners,
            direct_faults=state.direct_faults,
            quarantined=quarantined,
        )

    def score(self, peer_id: PeerId) -> float:
        return self.breakdown(peer_id).score

    def known_peers(self) -> list[PeerId]:
        return list(self._peers)

    def ranked(self, peer_ids: Optional[Iterable[PeerId]] = None) -> list[ScoreBreakdown]:
        """Peers ordered best-first. Ties break on peer id for determinism."""
        targets = list(peer_ids) if peer_ids is not None else self.known_peers()
        results = [self.breakdown(pid) for pid in targets]
        results.sort(key=lambda b: (-b.score, b.peer_id))
        return results

    def exploration_bonus(self, peer_id: PeerId, *, coefficient: float = 0.25) -> float:
        """Optimism-under-uncertainty term for a peer, used by routing.

        Without this the system deadlocks on cold start: routing prefers proven peers, so
        a new volunteer never receives traffic, so it never becomes proven. The bonus
        decays as ``1/sqrt(1 + n)``, so it fades once a peer has real history.
        """
        state = self._peers.get(peer_id)
        observations = state.observations if state else 0.0
        return coefficient / math.sqrt(1.0 + observations)

    # ---- persistence -----------------------------------------------------------

    def save(self, path) -> None:
        """Persist scores so a restart does not discard everything measured.

        Without this a client rejoining a swarm re-learns every peer from the prior, which
        wastes the requests that produced the evidence and briefly re-exposes it to peers it
        had already excluded. Integrity state matters most: a confirmed mismatch has a 30-day
        half-life precisely so it outlives a session.

        Decayed counters are stored raw alongside their last-decay timestamps, so reloading
        after an outage applies the elapsed decay rather than resuming as though no time
        passed.
        """
        import json
        from pathlib import Path

        payload = {
            "version": 1,
            "saved_at": self.clock.now(),
            "peers": {
                peer_id: {
                    "success": state.success,
                    "failure": state.failure,
                    "observations": state.observations,
                    "integrity_pass": state.integrity_pass,
                    "direct_faults": state.direct_faults,
                    "confirmed_mismatches": state.confirmed_mismatches,
                    "comparisons": state.comparisons,
                    "disputes": dict(state.disputes),
                    "ambiguous": dict(state.ambiguous),
                    "last_reliability_decay": state.last_reliability_decay,
                    "last_integrity_decay": state.last_integrity_decay,
                    "latencies": [[v, w] for v, w in (state.latencies._samples if state.latencies else [])],
                    "latency_last_decay": state.latencies._last_decay if state.latencies else 0.0,
                }
                for peer_id, state in self._peers.items()
            },
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-save must not leave a truncated file that silently
        # resets every score on next start.
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(target)

    def load(self, path) -> int:
        """Restore scores. Returns how many peers were loaded; 0 if there is nothing to load."""
        import json
        from pathlib import Path

        source = Path(path)
        if not source.exists():
            return 0
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return 0  # corrupt state is not worth crashing a client over
        if payload.get("version") != 1:
            return 0

        for peer_id, saved in payload.get("peers", {}).items():
            state = self._state(peer_id)
            state.success = float(saved.get("success", 0.0))
            state.failure = float(saved.get("failure", 0.0))
            state.observations = float(saved.get("observations", 0.0))
            state.integrity_pass = float(saved.get("integrity_pass", 0.0))
            state.direct_faults = float(saved.get("direct_faults", 0.0))
            state.confirmed_mismatches = float(saved.get("confirmed_mismatches", 0.0))
            state.comparisons = float(saved.get("comparisons", 0.0))
            state.disputes = {k: float(v) for k, v in (saved.get("disputes") or {}).items()}
            state.ambiguous = {k: float(v) for k, v in (saved.get("ambiguous") or {}).items()}
            state.last_reliability_decay = float(saved.get("last_reliability_decay", 0.0))
            state.last_integrity_decay = float(saved.get("last_integrity_decay", 0.0))
            if state.latencies is not None:
                state.latencies._samples = [
                    [float(v), float(w)] for v, w in saved.get("latencies", [])
                ]
                state.latencies._last_decay = float(saved.get("latency_last_decay", 0.0))
        return len(payload.get("peers", {}))

    def export_reports(self, *, since: float = 0.0) -> dict[PeerId, dict]:
        """Snapshot of local belief, shaped for publishing as signed observation reports."""
        now = self.clock.now()
        out: dict[PeerId, dict] = {}
        for peer_id, state in self._peers.items():
            self._decay_reliability(state, now)
            self._decay_integrity(state, now)
            if state.observations <= 0 and state.confirmed_mismatches <= 0:
                continue
            out[peer_id] = {
                "ok": state.success,
                "failed": state.failure,
                "mismatches": state.confirmed_mismatches,
                "p50_ms": state.latencies.quantile(0.50, now),
                "p95_ms": state.latencies.quantile(0.95, now),
            }
        return out
