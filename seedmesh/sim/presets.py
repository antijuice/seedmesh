"""Standard swarm configurations used by the CLI and the test suite.

Every preset gives each operator its own ASN unless the scenario is specifically about
operators pretending to be several people. That default matters: if honest servers
accidentally shared an ASN, the diversity constraint would starve verification of eligible
pairs and the scenario would measure the wrong thing.
"""

from __future__ import annotations

from seedmesh.sim.world import Behaviour, SwarmBuilder

N_BLOCKS = 32
MODEL_ID = "sim-model-32"


def _builder(seed: int) -> SwarmBuilder:
    return SwarmBuilder(model_id=MODEL_ID, n_blocks=N_BLOCKS, seed=seed)


def healthy_swarm(seed: int = 7) -> SwarmBuilder:
    """Only honest volunteers, on deliberately mismatched hardware.

    The baseline that matters most: heterogeneous hardware is the premise of a volunteer
    network, so if verification produces false accusations here, it is unusable in
    principle. Every range is covered by several independent operators.
    """
    builder = _builder(seed)
    for index, (start, end) in enumerate([(0, 8), (8, 16), (16, 24), (24, 32)]):
        builder.add(Behaviour.HONEST, (start, end), hardware="a100_bf16", first_seen=index * 3600.0)
        builder.add(Behaviour.HONEST, (start, end), hardware="rtx3090_fp16", first_seen=index * 7200.0)
        builder.add(Behaviour.HONEST, (start, end), hardware="rtx3050_int8", first_seen=index * 10800.0)
    builder.add(Behaviour.HONEST, (0, 16), hardware="h100_fp16", first_seen=500.0)
    builder.add(Behaviour.HONEST, (16, 32), hardware="h100_fp16", first_seen=900.0)
    return builder


def mixed_threat_swarm(seed: int = 11) -> SwarmBuilder:
    """Honest majority plus one of each failure mode, spread across block ranges."""
    builder = healthy_swarm(seed)
    builder.add(Behaviour.LAZY, (0, 8), hardware="rtx3090_fp16", first_seen=20000.0)
    builder.add(Behaviour.BYZANTINE, (8, 16), hardware="rtx3050_int8", first_seen=21000.0)
    builder.add(Behaviour.SUBTLE, (16, 24), hardware="a100_bf16", first_seen=22000.0)
    builder.add(Behaviour.FLAKY, (24, 32), hardware="rtx3050_int8", first_seen=23000.0)
    return builder


def sybil_swarm(seed: int = 13) -> SwarmBuilder:
    """One operator floods a block range with many identities in a single ASN.

    Tests whether an attacker who owns most of the *identities* covering a range can also
    own the verification of that range. They should not be able to: sharing an ASN means
    the diversity constraint refuses to pair them with each other.
    """
    builder = healthy_swarm(seed)
    builder.add(
        Behaviour.SUBTLE,
        (8, 16),
        hardware="rtx3090_fp16",
        operator="sybil-fleet",
        asn=64999,
        first_seen=30000.0,
        count=12,
    )
    return builder
