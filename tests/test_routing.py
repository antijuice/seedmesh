"""Pipeline planning: exactness, hop/quality trade-off, exclusion and load spreading."""

from __future__ import annotations

import pytest

from seedmesh.core.clock import ManualClock
from seedmesh.core.types import BlockRange, Observation, Outcome, ServerInfo, Verdict
from seedmesh.reputation.routing_bias import Router, RoutingConfig
from seedmesh.reputation.scorer import ReputationScorer, ScorerConfig

MODEL = "m"
N_BLOCKS = 32


def _server(peer_id, start, end, asn=1):
    return ServerInfo(
        peer_id=peer_id,
        block_range=BlockRange(MODEL, start, end),
        first_seen=0.0,
        address=f"198.51.{asn}.1",
        asn=asn,
    )


def _router(clock=None, **config):
    scorer = ReputationScorer(ScorerConfig(), clock=clock or ManualClock())
    return Router(scorer, RoutingConfig(**config)), scorer


def _train(scorer, clock, peer_id, successes, outcome=Outcome.OK, latency=50.0):
    for _ in range(successes):
        scorer.record(
            Observation("smobs", peer_id, BlockRange(MODEL, 0, 8), outcome, latency, clock.now())
        )


def test_plan_covers_the_whole_model_without_gaps():
    router, _ = _router(load_weight=0.0)
    servers = [_server(f"sm{i}", i * 8, (i + 1) * 8) for i in range(4)]
    route = router.plan(MODEL, N_BLOCKS, servers)

    assert route is not None
    assert route.covers(N_BLOCKS)
    assert route.hop_count == 4


def test_fewer_hops_are_preferred_when_quality_is_equal():
    router, _ = _router(load_weight=0.0)
    servers = [
        _server("smwide1", 0, 16),
        _server("smwide2", 16, 32),
        *[_server(f"smnarrow{i}", i * 8, (i + 1) * 8) for i in range(4)],
    ]
    assert router.plan(MODEL, N_BLOCKS, servers).hop_count == 2


def test_quality_can_outweigh_an_extra_hop():
    clock = ManualClock()
    router, scorer = _router(clock, load_weight=0.0, hop_penalty=0.2)
    servers = [
        _server("smwide", 0, 32),
        _server("smgood1", 0, 16),
        _server("smgood2", 16, 32),
    ]

    _train(scorer, clock, "smwide", 200, outcome=Outcome.ERROR)
    _train(scorer, clock, "smgood1", 200)
    _train(scorer, clock, "smgood2", 200)

    assert router.plan(MODEL, N_BLOCKS, servers).peers == ("smgood1", "smgood2")


def test_partial_use_of_an_overlapping_range_is_found():
    """The optimum can split a wide server where a better one takes over."""
    clock = ManualClock()
    router, scorer = _router(clock, load_weight=0.0, hop_penalty=0.05)
    servers = [_server("smwide", 0, 32), _server("smbetter", 16, 32)]

    _train(scorer, clock, "smwide", 200, outcome=Outcome.ERROR)
    _train(scorer, clock, "smbetter", 200)

    route = router.plan(MODEL, N_BLOCKS, servers)
    assert route.covers(N_BLOCKS)
    assert route.hops[-1].server.peer_id == "smbetter"
    assert route.hops[-1].covers == BlockRange(MODEL, 16, 32)


def test_uncoverable_model_returns_none_rather_than_a_partial_route():
    router, _ = _router()
    assert router.plan(MODEL, N_BLOCKS, [_server("sma", 0, 16)]) is None


def test_empty_candidate_set_returns_none():
    router, _ = _router()
    assert router.plan(MODEL, N_BLOCKS, []) is None


def test_servers_for_another_model_are_ignored():
    router, _ = _router()
    other = ServerInfo("smother", BlockRange("other", 0, 32), 0.0)
    assert router.plan(MODEL, N_BLOCKS, [other]) is None


def test_quarantined_servers_are_excluded():
    clock = ManualClock()
    router, scorer = _router(clock)
    scorer.record_direct_fault("smcheat")

    servers = [_server("smcheat", 0, 32), _server("smclean", 0, 32)]
    assert router.plan(MODEL, N_BLOCKS, servers).peers == ("smclean",)


def test_a_single_accusation_does_not_exclude():
    """Exclusion must track the corroborated quarantine, not a bare low score."""
    clock = ManualClock()
    router, scorer = _router(clock)
    scorer.record_verification("smaccused", Verdict.MISMATCH, partner_cluster="asn:1")

    assert not router.is_excluded("smaccused")


def test_explicitly_excluded_peers_are_skipped():
    router, _ = _router()
    servers = [_server("sma", 0, 32), _server("smb", 0, 32)]
    assert router.plan(MODEL, N_BLOCKS, servers, exclude={"sma"}).peers == ("smb",)


def test_exploration_lets_an_unproven_peer_compete():
    """Without this the swarm cannot absorb new capacity: unknown peers never get tried,
    so they never become known."""
    clock = ManualClock()
    router, scorer = _router(clock, exploration_coefficient=0.25)
    _train(scorer, clock, "smproven", 500, latency=400.0)

    assert router.effective_score("smfresh") > router.effective_score("smproven") - 0.3


def test_zero_exploration_reproduces_cold_start_starvation():
    clock = ManualClock()
    router, scorer = _router(clock, exploration_coefficient=0.0)
    _train(scorer, clock, "smproven", 500, latency=10.0)

    assert router.effective_score("smproven") > router.effective_score("smfresh")


def test_load_spreads_across_equally_good_servers():
    clock = ManualClock()
    router, _ = _router(clock, load_weight=4.0)
    servers = [_server("sma", 0, 32, asn=1), _server("smb", 0, 32, asn=2)]

    chosen = []
    for _ in range(20):
        route = router.plan(MODEL, N_BLOCKS, servers)
        peer = route.peers[0]
        chosen.append(peer)
        router.note_assignment(peer, N_BLOCKS)

    assert set(chosen) == {"sma", "smb"}
    assert 5 <= chosen.count("sma") <= 15


def test_zero_load_weight_fixates_on_one_server():
    clock = ManualClock()
    router, _ = _router(clock, load_weight=0.0)
    servers = [_server("sma", 0, 32, asn=1), _server("smb", 0, 32, asn=2)]

    chosen = set()
    for _ in range(20):
        peer = router.plan(MODEL, N_BLOCKS, servers).peers[0]
        chosen.add(peer)
        router.note_assignment(peer, N_BLOCKS)

    assert len(chosen) == 1


def test_recent_load_decays():
    clock = ManualClock()
    router, _ = _router(clock, load_half_life_s=100.0)
    router.note_assignment("sma", 10)

    before = router.recent_load("sma")
    clock.advance(100.0)
    assert router.recent_load("sma") == pytest.approx(before / 2, rel=1e-6)


def test_alternates_are_ranked_and_exclude_the_failed_peer():
    clock = ManualClock()
    router, scorer = _router(clock)
    needed = BlockRange(MODEL, 0, 8)
    servers = [_server("smgood", 0, 8), _server("smbad", 0, 8), _server("smdead", 0, 8)]

    _train(scorer, clock, "smgood", 100)
    _train(scorer, clock, "smbad", 100, outcome=Outcome.ERROR)

    alternates = router.alternates(needed, servers, exclude={"smdead"})
    assert [s.peer_id for s in alternates][0] == "smgood"
    assert "smdead" not in [s.peer_id for s in alternates]


def test_alternates_require_full_coverage_of_the_needed_range():
    router, _ = _router()
    needed = BlockRange(MODEL, 0, 8)
    assert router.alternates(needed, [_server("smpartial", 0, 4)]) == []


def test_planning_is_deterministic():
    router, _ = _router(load_weight=0.0)
    servers = [_server(f"sm{i}", i * 8, (i + 1) * 8) for i in range(4)]
    assert router.plan(MODEL, N_BLOCKS, servers).peers == router.plan(
        MODEL, N_BLOCKS, servers
    ).peers


def test_invalid_block_count_is_rejected():
    router, _ = _router()
    with pytest.raises(ValueError):
        router.plan(MODEL, 0, [])


def test_mean_score_weights_by_blocks_covered():
    router, _ = _router(load_weight=0.0)
    servers = [_server("sma", 0, 24), _server("smb", 24, 32)]
    route = router.plan(MODEL, N_BLOCKS, servers)
    assert 0.0 <= route.mean_score() <= 1.0
