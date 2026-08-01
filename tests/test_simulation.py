"""End-to-end behaviour of the whole trust layer against the simulated swarm.

These assert on outcomes the design is supposed to guarantee, so they fail if a future
change quietly trades away sybil resistance or starts convicting honest volunteers. They
are deliberately loose on exact numbers and strict on direction.
"""

from __future__ import annotations

import pytest

from seedmesh.sim import calibrate_world, run_gossip, run_swarm
from seedmesh.sim.presets import healthy_swarm, mixed_threat_swarm, sybil_swarm
from seedmesh.sim.world import Behaviour


@pytest.fixture(scope="module")
def tolerance():
    """Calibrate on an all-honest swarm.

    Fitting thresholds on a population that contains cheats would widen the tolerance
    until the cheats fit inside it, which is how a verification system quietly becomes a
    rubber stamp.
    """
    return calibrate_world(healthy_swarm(seed=7).build(), samples=150)


def test_calibration_produces_a_usable_band(tolerance):
    assert 0.0 < tolerance.match_distance < 0.1
    assert tolerance.mismatch_distance > tolerance.match_distance


def test_healthy_swarm_never_accuses_an_honest_server(tolerance):
    """The single most important property. Heterogeneous hardware is the premise of a
    volunteer network, so false accusations here make the design unusable in principle."""
    world = healthy_swarm(seed=7).build()
    metrics = run_swarm(world, requests=300, tolerance=tolerance, seed=7)

    assert metrics.verdict_mismatch == 0
    assert metrics.quarantined_honest == {}
    assert metrics.success_rate > 0.9


def test_healthy_swarm_spreads_load(tolerance):
    world = healthy_swarm(seed=7).build()
    metrics = run_swarm(world, requests=300, tolerance=tolerance, seed=7)

    idle = [peer for peer, count in metrics.load.items() if count == 0]
    assert not idle
    assert metrics.load_gini() < 0.7


def test_threats_are_detected_and_routed_around(tolerance):
    world = mixed_threat_swarm(seed=11).build()
    malicious = {p for p, s in world.servers.items() if s.is_malicious}
    metrics = run_swarm(world, requests=400, tolerance=tolerance, seed=11)

    assert len(metrics.quarantined_malicious) >= len(malicious) - 1
    assert metrics.quarantined_honest == {}
    assert metrics.malicious_hops_second_half <= metrics.malicious_hops_first_half


def test_a_sybil_fleet_is_caught_and_does_not_convict_its_accusers(tolerance):
    """Regression for the defect the simulator exposed: with disputes keyed by peer id, a
    12-identity fleet convicted four honest servers and none of its own members."""
    world = sybil_swarm(seed=13).build()
    fleet = {p for p, s in world.servers.items() if s.is_malicious}
    metrics = run_swarm(world, requests=400, tolerance=tolerance, seed=13)

    assert metrics.quarantined_honest == {}
    assert len(metrics.quarantined_malicious) >= len(fleet) * 0.8
    assert metrics.malicious_hops_second_half < metrics.malicious_hops_first_half


def test_lazy_servers_that_echo_their_input_are_caught(tolerance):
    """Passthrough is invisible to redundant sampling alone -- it looks like a fast success."""
    from seedmesh.sim.world import SwarmBuilder

    builder = SwarmBuilder(model_id="sim-model-32", n_blocks=32, seed=3)
    for start, end in [(0, 16), (16, 32)]:
        builder.add(Behaviour.HONEST, (start, end), hardware="a100_bf16", first_seen=0.0)
        builder.add(Behaviour.HONEST, (start, end), hardware="cpu_fp32", first_seen=50_000.0)
    builder.add(Behaviour.LAZY, (0, 16), hardware="rtx3090_fp16", first_seen=90_000.0)

    world = builder.build()
    metrics = run_swarm(world, requests=300, tolerance=tolerance, seed=3)

    assert metrics.passthrough_detections > 0 or metrics.verdict_mismatch > 0
    assert len(metrics.quarantined_malicious) == 1


def test_verification_overhead_stays_modest(tolerance):
    world = healthy_swarm(seed=7).build()
    metrics = run_swarm(world, requests=300, tolerance=tolerance, seed=7)

    assert metrics.verifications / max(metrics.total_hops, 1) < 0.25


def test_disabling_verification_lets_cheats_persist(tolerance):
    """Control: confirms the detections above come from verification, not from luck."""
    world = mixed_threat_swarm(seed=11).build()
    metrics = run_swarm(
        world, requests=300, tolerance=tolerance, seed=11, enable_verification=False
    )
    assert metrics.quarantined_malicious == {}


def test_simulation_is_deterministic(tolerance):
    a = run_swarm(healthy_swarm(seed=7).build(), requests=120, tolerance=tolerance, seed=5)
    b = run_swarm(healthy_swarm(seed=7).build(), requests=120, tolerance=tolerance, seed=5)
    assert (a.succeeded, a.total_hops, a.verifications) == (
        b.succeeded,
        b.total_hops,
        b.verifications,
    )


def test_gossip_aggregation_resists_a_ten_to_one_sybil_majority():
    result = run_gossip(honest_observers=5, sybil_observers=50)

    assert result.rejected_records == 0
    assert result.sybil_target_score < 0.5, "sybil fleet moved the aggregate"
    assert result.mismatch_confirmed
