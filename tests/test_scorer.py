"""Reputation scoring: decay, priors, and the corroboration rule.

The corroboration tests are regressions for two defects the simulator exposed, both of
which would have made the network worse than having no verification at all.
"""

from __future__ import annotations

import pytest

from seedmesh.core.clock import ManualClock
from seedmesh.core.types import BlockRange, Observation, Outcome, Verdict
from seedmesh.reputation.scorer import ReputationScorer, ScorerConfig

RANGE = BlockRange("m", 0, 4)


def _scorer(clock=None, **overrides):
    return ReputationScorer(ScorerConfig(**overrides), clock=clock or ManualClock())


def _observe(scorer, subject, outcome, clock, latency=50.0):
    scorer.record(
        Observation("smobserver", subject, RANGE, outcome, latency, clock.now())
    )


def test_unknown_peer_scores_at_the_prior():
    scorer = _scorer()
    breakdown = scorer.breakdown("smnever-seen")
    assert breakdown.reliability == pytest.approx(scorer.config.prior_success_rate)
    assert breakdown.confidence == 0.0
    assert not breakdown.is_quarantined


def test_single_success_does_not_produce_a_perfect_score():
    """A Beta prior is the point: one lucky result must not outrank a long clean record."""
    clock = ManualClock()
    scorer = _scorer(clock)
    _observe(scorer, "smlucky", Outcome.OK, clock)
    assert scorer.breakdown("smlucky").reliability < 0.95


def test_sustained_success_beats_a_single_success():
    clock = ManualClock()
    scorer = _scorer(clock)
    _observe(scorer, "smnew", Outcome.OK, clock)
    for _ in range(200):
        _observe(scorer, "smproven", Outcome.OK, clock)
    assert scorer.breakdown("smproven").reliability > scorer.breakdown("smnew").reliability


def test_failures_reduce_reliability():
    clock = ManualClock()
    scorer = _scorer(clock)
    for _ in range(20):
        _observe(scorer, "smbad", Outcome.ERROR, clock)
    assert scorer.breakdown("smbad").reliability < 0.2


def test_refused_is_neither_rewarded_nor_punished():
    """Honest backpressure must not be penalised, or servers learn to accept work they
    cannot do."""
    clock = ManualClock()
    scorer = _scorer(clock)
    before = scorer.breakdown("smbusy").reliability
    for _ in range(10):
        _observe(scorer, "smbusy", Outcome.REFUSED, clock)
    after = scorer.breakdown("smbusy")
    assert after.reliability == pytest.approx(before)
    assert after.observations == 0.0


def test_reliability_decays_toward_the_prior():
    clock = ManualClock()
    scorer = _scorer(clock, reliability_half_life_s=3600.0)
    for _ in range(50):
        _observe(scorer, "smflaky", Outcome.ERROR, clock)

    damaged = scorer.breakdown("smflaky").reliability
    clock.advance(3600.0 * 12)
    recovered = scorer.breakdown("smflaky").reliability

    assert recovered > damaged
    assert recovered == pytest.approx(scorer.config.prior_success_rate, abs=0.05)


def test_decay_halves_evidence_over_one_half_life():
    clock = ManualClock()
    scorer = _scorer(clock, reliability_half_life_s=100.0)
    for _ in range(8):
        _observe(scorer, "smpeer", Outcome.OK, clock)

    before = scorer.breakdown("smpeer").observations
    clock.advance(100.0)
    assert scorer.breakdown("smpeer").observations == pytest.approx(before / 2, rel=1e-6)


def test_latency_modulates_but_never_dominates():
    """A fast liar must always rank below a slow honest node."""
    clock = ManualClock()
    scorer = _scorer(clock)
    for _ in range(50):
        _observe(scorer, "smslow", Outcome.OK, clock, latency=5000.0)

    slow = scorer.breakdown("smslow")
    assert slow.latency_factor >= 1.0 - scorer.config.latency_weight
    assert slow.score > 0.6


def test_integrity_decays_far_more_slowly_than_reliability():
    clock = ManualClock()
    scorer = _scorer(clock, reliability_half_life_s=3600.0, integrity_half_life_s=30 * 86400.0)
    scorer.record_direct_fault("smcheat")

    clock.advance(86400.0)
    assert scorer.breakdown("smcheat").integrity < 0.2


# ---- the corroboration rule ---------------------------------------------------


def test_a_single_dispute_does_not_quarantine_either_party():
    """Regression: charging both sides of a mismatch at full weight evicted the honest
    peer that caught the cheat, which makes verifying actively dangerous."""
    clock = ManualClock()
    scorer = _scorer(clock)
    scorer.record_verification("smhonest", Verdict.MISMATCH, partner_cluster="asn:1")

    breakdown = scorer.breakdown("smhonest")
    assert breakdown.disputing_partners == 1
    assert not breakdown.is_quarantined


def test_two_independent_clusters_do_quarantine():
    clock = ManualClock()
    scorer = _scorer(clock)
    scorer.record_verification("smcheat", Verdict.MISMATCH, partner_cluster="asn:1")
    scorer.record_verification("smcheat", Verdict.MISMATCH, partner_cluster="asn:2")

    breakdown = scorer.breakdown("smcheat")
    assert breakdown.disputing_partners == 2
    assert breakdown.is_quarantined


def test_a_sybil_fleet_in_one_cluster_cannot_convict_an_honest_peer():
    """Regression: disputes keyed by peer id let a 12-identity fleet manufacture twelve
    distinct accusers and convict the node that caught them. Keying by network cluster
    collapses the whole fleet to one accuser."""
    clock = ManualClock()
    scorer = _scorer(clock)

    for _ in range(12):
        scorer.record_verification("smhonest", Verdict.MISMATCH, partner_cluster="asn:64999")

    breakdown = scorer.breakdown("smhonest")
    assert breakdown.disputing_partners == 1
    assert not breakdown.is_quarantined


def test_a_direct_fault_needs_no_corroboration():
    """Passthrough is witnessed by the client itself, not inferred from a disagreement."""
    clock = ManualClock()
    scorer = _scorer(clock)
    scorer.record_direct_fault("smlazy")

    breakdown = scorer.breakdown("smlazy")
    assert breakdown.direct_faults == 1.0
    assert breakdown.is_quarantined


def test_a_few_ambiguous_results_among_many_matches_prove_nothing():
    """Regression for a self-fulfilling scrutiny loop.

    Absolute accumulation of ambiguous results meant that sampling a peer more often
    pushed it toward conviction on its own: sybil disputes dented integrity, integrity
    raised the verification rate, and the extra sampling turned up the rare honest
    ambiguous result. Gating on rate makes extra sampling neutral, because the denominator
    grows with the numerator.
    """
    clock = ManualClock()
    scorer = _scorer(clock)

    for _ in range(100):
        scorer.record_verification("smhonest", Verdict.MATCH, partner_cluster="asn:1")
    for cluster in ("asn:2", "asn:3", "asn:4"):
        scorer.record_ambiguous("smhonest", cluster)

    assert not scorer.breakdown("smhonest").is_quarantined


def test_a_high_ambiguous_rate_across_independent_clusters_does_convict():
    """The attacker who tunes corruption just under the mismatch bound still loses."""
    clock = ManualClock()
    scorer = _scorer(clock)

    for cluster in ("asn:1", "asn:2"):
        for _ in range(4):
            scorer.record_ambiguous("smsubtle", cluster)

    assert scorer.breakdown("smsubtle").is_quarantined


def test_ambiguous_results_from_one_cluster_alone_do_not_convict():
    """An honest verifier repeatedly checking one cheating fleet must stay safe."""
    clock = ManualClock()
    scorer = _scorer(clock)
    for _ in range(20):
        scorer.record_ambiguous("smhonest", "asn:64999")

    breakdown = scorer.breakdown("smhonest")
    assert breakdown.disputing_partners == 1
    assert not breakdown.is_quarantined


def test_inconclusive_verdicts_record_nothing():
    clock = ManualClock()
    scorer = _scorer(clock)
    for _ in range(20):
        scorer.record_verification("smpeer", Verdict.INCONCLUSIVE, partner_cluster="asn:1")

    breakdown = scorer.breakdown("smpeer")
    assert breakdown.integrity == pytest.approx(1.0)
    assert not breakdown.is_quarantined


def test_matches_cannot_launder_a_corroborated_mismatch():
    """Integrity is a gate, not an average -- a burst of cheap successes must not clear it."""
    clock = ManualClock()
    scorer = _scorer(clock)
    scorer.record_verification("smcheat", Verdict.MISMATCH, partner_cluster="asn:1")
    scorer.record_verification("smcheat", Verdict.MISMATCH, partner_cluster="asn:2")
    for _ in range(30):
        scorer.record_verification("smcheat", Verdict.MATCH, partner_cluster="asn:3")

    assert scorer.breakdown("smcheat").is_quarantined


def test_exploration_bonus_decays_with_evidence():
    clock = ManualClock()
    scorer = _scorer(clock)
    fresh = scorer.exploration_bonus("smnew")
    for _ in range(100):
        _observe(scorer, "smknown", Outcome.OK, clock)
    assert scorer.exploration_bonus("smknown") < fresh / 3


def test_ranking_is_deterministic_under_ties():
    clock = ManualClock()
    scorer = _scorer(clock)
    for peer in ("smb", "sma", "smc"):
        _observe(scorer, peer, Outcome.OK, clock)
    first = [b.peer_id for b in scorer.ranked()]
    assert first == [b.peer_id for b in scorer.ranked()]


def test_export_reports_omits_peers_with_no_evidence():
    clock = ManualClock()
    scorer = _scorer(clock)
    scorer.breakdown("smuntouched")
    _observe(scorer, "smused", Outcome.OK, clock)
    assert set(scorer.export_reports()) == {"smused"}
