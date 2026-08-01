"""A deterministic swarm simulator for the trust layer.

Real arrays, real sketches, real comparisons -- only the network and the GPUs are
simulated. See :mod:`seedmesh.sim.world` for why that distinction is the point.
"""

from seedmesh.sim.presets import healthy_swarm, mixed_threat_swarm, sybil_swarm
from seedmesh.sim.scenarios import (
    GossipResult,
    RunMetrics,
    calibrate_world,
    run_gossip,
    run_swarm,
)
from seedmesh.sim.world import Behaviour, HARDWARE, SimServer, SimWorld, SwarmBuilder

__all__ = [
    "Behaviour",
    "GossipResult",
    "HARDWARE",
    "RunMetrics",
    "SimServer",
    "SimWorld",
    "SwarmBuilder",
    "calibrate_world",
    "healthy_swarm",
    "mixed_threat_swarm",
    "run_gossip",
    "run_swarm",
    "sybil_swarm",
]
