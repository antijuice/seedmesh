"""Aggregation of strangers' claims, and the diversity rules underneath it."""

from __future__ import annotations

import pytest

from seedmesh.core.clock import ManualClock
from seedmesh.core.identity import Identity
from seedmesh.core.types import BlockRange, ServerInfo
from seedmesh.reputation.aggregate import (
    AggregationConfig,
    ReputationAggregator,
    blend,
    weighted_median,
)
from seedmesh.reputation.diversity import UNKNOWN_CLUSTER, ClusterIndex, StaticAsnResolver
from seedmesh.reputation.records import PeerReport, build_batch, verify_batch
from seedmesh.reputation.scorer import ScoreBreakdown

TARGET = "smtarget"


def _breakdown(score=0.9, confidence=0.5):
    return ScoreBreakdown(
        peer_id=TARGET,
        score=score,
        reliability=score,
        integrity=1.0,
        latency_factor=1.0,
        confidence=confidence,
        observations=10.0,
        p50_ms=50.0,
        p95_ms=100.0,
        confirmed_mismatches=0.0,
    )


# ---- weighted median -----------------------------------------------------------


def test_weighted_median_ignores_zero_weight_claims():
    assert weighted_median([(0.0, 0.0), (1.0, 1.0)]) == 1.0


def test_weighted_median_resists_extremes():
    pairs = [(0.9, 1.0), (0.9, 1.0), (0.9, 1.0), (0.0, 1.0)]
    assert weighted_median(pairs) == pytest.approx(0.9)


def test_weighted_median_of_nothing_is_zero():
    assert weighted_median([]) == 0.0


# ---- sybil resistance ----------------------------------------------------------


def test_a_sybil_fleet_in_one_cluster_cannot_overturn_honest_consensus():
    clock = ManualClock()
    honest = {f"smh{i}": f"asn:{100 + i}" for i in range(4)}
    sybils = {f"sms{i}": "asn:64999" for i in range(40)}
    clusters = {**honest, **sybils, TARGET: "asn:999"}

    aggregator = ReputationAggregator(lambda p: clusters.get(p, UNKNOWN_CLUSTER))

    for peer_id in honest:
        identity = Identity.from_seed(peer_id.encode().ljust(32, b"\x00"))
        clusters[identity.peer_id] = clusters[peer_id]
        report = PeerReport(TARGET, 2.0, 38.0, 1.0, 200.0, 800.0)
        aggregator.ingest(
            verify_batch(build_batch(identity, [report], epoch=1, clock=clock), clock=clock)
        )

    for peer_id in sybils:
        identity = Identity.from_seed(peer_id.encode().ljust(32, b"\x00"))
        clusters[identity.peer_id] = clusters[peer_id]
        report = PeerReport(TARGET, 40.0, 0.0, 0.0, 5.0, 10.0)
        aggregator.ingest(
            verify_batch(build_batch(identity, [report], epoch=1, clock=clock), clock=clock)
        )

    score = aggregator.aggregate(TARGET)
    assert score.reliability < 0.4, "40 sybils outvoted 4 honest observers"
    assert score.mismatch_confirmed


def test_one_observer_gets_one_vote_regardless_of_republishing():
    """Only the newest batch per observer counts, so spamming buys no influence."""
    clock = ManualClock()
    aggregator = ReputationAggregator(lambda _p: "asn:1")
    identity = Identity.generate()

    for epoch in range(1, 6):
        report = PeerReport(TARGET, 40.0, 0.0, 0.0, 5.0, 10.0)
        aggregator.ingest(
            verify_batch(build_batch(identity, [report], epoch=epoch, clock=clock), clock=clock)
        )

    assert aggregator.aggregate(TARGET).observer_count == 1


def test_cluster_cap_bounds_a_single_networks_influence():
    clock = ManualClock()
    config = AggregationConfig(max_weight_per_cluster=3.0)
    # The target sits in its own cluster, so the collusion discount stays out of the way
    # and this measures the cap alone.
    aggregator = ReputationAggregator(
        lambda p: "asn:target" if p == TARGET else "asn:1", config
    )

    for index in range(20):
        identity = Identity.from_seed(bytes([index + 1]) * 32)
        report = PeerReport(TARGET, 40.0, 0.0, 0.0, 5.0, 10.0)
        aggregator.ingest(
            verify_batch(build_batch(identity, [report], epoch=1, clock=clock), clock=clock)
        )

    assert aggregator.aggregate(TARGET).total_weight == pytest.approx(3.0)


def test_unknown_cluster_gets_a_smaller_budget_than_a_known_one():
    clock = ManualClock()
    config = AggregationConfig(unknown_cluster_weight=0.5, max_weight_per_cluster=3.0)
    aggregator = ReputationAggregator(lambda _p: UNKNOWN_CLUSTER, config)

    for index in range(20):
        identity = Identity.from_seed(bytes([index + 1]) * 32)
        report = PeerReport(TARGET, 40.0, 0.0, 0.0, 5.0, 10.0)
        aggregator.ingest(
            verify_batch(build_batch(identity, [report], epoch=1, clock=clock), clock=clock)
        )

    assert aggregator.aggregate(TARGET).total_weight == pytest.approx(0.5)


def test_self_dealing_within_a_cluster_is_discounted():
    clock = ManualClock()
    config = AggregationConfig(collusion_discount=0.1)
    aggregator = ReputationAggregator(lambda _p: "asn:7", config)

    identity = Identity.generate()
    report = PeerReport(TARGET, 40.0, 0.0, 0.0, 5.0, 10.0)
    aggregator.ingest(
        verify_batch(build_batch(identity, [report], epoch=1, clock=clock), clock=clock)
    )

    assert aggregator.aggregate(TARGET).total_weight == pytest.approx(0.1)


def test_an_accusation_from_one_cluster_is_not_confirmed():
    """Defaming an honest competitor must be at least as hard as praising yourself."""
    clock = ManualClock()
    aggregator = ReputationAggregator(lambda _p: "asn:5")
    identity = Identity.generate()
    report = PeerReport(TARGET, 0.0, 10.0, 3.0, 200.0, 900.0)
    aggregator.ingest(
        verify_batch(build_batch(identity, [report], epoch=1, clock=clock), clock=clock)
    )

    assert not aggregator.aggregate(TARGET).mismatch_confirmed


def test_expired_batches_stop_counting():
    clock = ManualClock()
    aggregator = ReputationAggregator(lambda _p: "asn:1")
    identity = Identity.generate()
    report = PeerReport(TARGET, 10.0, 0.0, 0.0, 5.0, 10.0)
    aggregator.ingest(
        verify_batch(build_batch(identity, [report], epoch=1, clock=clock, ttl_s=60.0), clock=clock)
    )

    clock.advance(120.0)
    assert aggregator.drop_expired(clock.now()) == 1
    assert aggregator.aggregate(TARGET).total_weight == 0.0


def test_unknown_subject_falls_back_to_the_prior():
    aggregator = ReputationAggregator(lambda _p: "asn:1")
    score = aggregator.aggregate("smnobody")
    assert score.reliability == pytest.approx(aggregator.config.prior_success_rate)
    assert not score.is_trustworthy_signal


# ---- blending ------------------------------------------------------------------


def test_first_hand_experience_dominates_once_confident():
    from seedmesh.reputation.aggregate import AggregateScore

    remote = AggregateScore(TARGET, 0.1, 5.0, 4, 5, 0, False, 1.0)
    assert blend(_breakdown(0.95, confidence=0.99), remote) > 0.9


def test_hearsay_is_capped_even_with_no_local_evidence():
    from seedmesh.reputation.aggregate import AggregateScore

    remote = AggregateScore(TARGET, 0.0, 5.0, 4, 5, 0, False, 1.0)
    assert blend(_breakdown(0.9, confidence=0.0), remote, max_remote_influence=0.5) >= 0.45


def test_a_corroborated_mismatch_gates_the_score():
    from seedmesh.reputation.aggregate import AggregateScore

    remote = AggregateScore(TARGET, 0.99, 5.0, 4, 5, 3, True, 1.0)
    assert blend(_breakdown(0.99, confidence=0.9), remote) <= 0.1


def test_untrustworthy_remote_signal_is_ignored():
    from seedmesh.reputation.aggregate import AggregateScore

    local = _breakdown(0.9, confidence=0.2)
    remote = AggregateScore(TARGET, 0.0, 0.0, 0, 0, 0, False, 0.0)
    assert blend(local, remote) == local.score


# ---- diversity -----------------------------------------------------------------


def test_addressless_peers_share_one_bucket():
    """Otherwise omitting your address would mint unlimited clusters."""
    index = ClusterIndex()
    a = ServerInfo("sma", BlockRange("m", 0, 4), 0.0)
    b = ServerInfo("smb", BlockRange("m", 0, 4), 0.0)
    assert index.coarse_of(a) == index.coarse_of(b) == UNKNOWN_CLUSTER
    assert not index.are_independent(a, b)


def test_asn_is_preferred_over_address_prefix():
    index = ClusterIndex(StaticAsnResolver({"203.0.113.5": 64500}))
    server = ServerInfo("sma", BlockRange("m", 0, 4), 0.0, address="203.0.113.5")
    assert index.coarse_of(server) == "asn:64500"


def test_addresses_in_the_same_prefix_share_a_cluster():
    index = ClusterIndex()
    a = ServerInfo("sma", BlockRange("m", 0, 4), 0.0, address="198.51.10.1")
    b = ServerInfo("smb", BlockRange("m", 0, 4), 0.0, address="198.51.240.9")
    assert index.coarse_of(a) == index.coarse_of(b)


def test_distinct_networks_are_independent():
    index = ClusterIndex()
    a = ServerInfo("sma", BlockRange("m", 0, 4), 0.0, address="198.51.10.1", asn=1)
    b = ServerInfo("smb", BlockRange("m", 0, 4), 100_000.0, address="203.0.113.9", asn=2)
    assert index.are_independent(a, b)


def test_malformed_addresses_fall_back_to_the_unknown_bucket():
    index = ClusterIndex()
    server = ServerInfo("sma", BlockRange("m", 0, 4), 0.0, address="not-an-ip")
    assert index.coarse_of(server) == UNKNOWN_CLUSTER
