"""A simulated swarm: real arrays, real sketches, fake network.

This is not a mock. Servers compute actual numpy transforms over actual hidden states, and
honest servers on different simulated hardware add different floating-point noise -- so
:func:`~seedmesh.verification.compare.compare_sketches` makes genuine decisions on genuine
numerical disagreement, and calibration has a real distribution to fit.

That matters because it exercises the one failure mode a mocked test would hide: the
question is never "does the comparison run", it is "can it separate honest hardware
diversity from dishonest output", and only real numbers answer that.

The simulator satisfies :class:`~seedmesh.backends.base.InferenceBackend`, so the same
routing and verification code drives it and a real backend.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from seedmesh.backends.base import SegmentResult
from seedmesh.core.clock import ManualClock
from seedmesh.core.types import BlockRange, Outcome, PeerId, ServerInfo
from seedmesh.verification.sketch import Sketch, SketchSpec, derive_seed, sketch_hidden_state

HIDDEN_DIM = 64
"""Small enough that a full scenario runs in under a second; large enough that random
projection behaves as the Johnson-Lindenstrauss bound predicts."""


class Behaviour(enum.Enum):
    """How a simulated server responds to work."""

    HONEST = "honest"
    FLAKY = "flaky"
    """Honest arithmetic, unreliable delivery -- the ordinary volunteer on bad wifi."""

    LAZY = "lazy"
    """Skips the forward pass and echoes its input. Costs no GPU and looks fast."""

    BYZANTINE = "byzantine"
    """Returns plausibly-shaped garbage."""

    SUBTLE = "subtle"
    """Computes correctly, then perturbs the result slightly. The hardest case: large
    enough to corrupt output, small enough to hide in honest hardware noise."""


@dataclass(frozen=True, slots=True)
class HardwareClass:
    """Simulated device profile. ``noise`` stands in for dtype and kernel differences."""

    name: str
    noise: float
    latency_mean_ms: float
    latency_jitter_ms: float


HARDWARE = {
    "a100_bf16": HardwareClass("a100_bf16", 3e-3, 40.0, 8.0),
    "h100_fp16": HardwareClass("h100_fp16", 2e-3, 30.0, 6.0),
    "rtx3090_fp16": HardwareClass("rtx3090_fp16", 4e-3, 90.0, 25.0),
    "rtx3050_int8": HardwareClass("rtx3050_int8", 9e-3, 220.0, 70.0),
    "cpu_fp32": HardwareClass("cpu_fp32", 1e-4, 900.0, 300.0),
}


def _block_weights(block_index: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-block transform. Same everywhere, like real model weights."""
    generator = np.random.default_rng(0xB10C + block_index)
    weights = generator.standard_normal((dim, dim)) / np.sqrt(dim)
    bias = generator.standard_normal(dim) * 0.01
    return weights, bias


class _BlockCache:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, block_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(block_index)
        if cached is None:
            cached = _block_weights(block_index, self._dim)
            self._cache[block_index] = cached
        return cached


@dataclass
class SimServer:
    """A simulated volunteer node."""

    peer_id: PeerId
    block_range: BlockRange
    behaviour: Behaviour
    hardware: HardwareClass
    address: str
    asn: int
    first_seen: float
    operator: str = "solo"
    """Servers sharing an operator are the same real person. Ground truth for scoring how
    well the diversity constraints actually separated colluding nodes."""

    failure_rate: float = 0.01
    go_offline_prob: float = 0.002
    come_online_prob: float = 0.25
    online: bool = True
    requests_served: int = 0

    @property
    def is_malicious(self) -> bool:
        return self.behaviour in (Behaviour.LAZY, Behaviour.BYZANTINE, Behaviour.SUBTLE)

    def info(self) -> ServerInfo:
        return ServerInfo(
            peer_id=self.peer_id,
            block_range=self.block_range,
            first_seen=self.first_seen,
            address=self.address,
            asn=self.asn,
            advertised_throughput=1000.0 / max(self.hardware.latency_mean_ms, 1.0),
        )


class SimWorld:
    """The simulated swarm. Implements the backend protocol."""

    def __init__(
        self,
        model_id: str,
        n_blocks: int,
        servers: Sequence[SimServer],
        *,
        seed: int = 1234,
        clock: Optional[ManualClock] = None,
        dim: int = HIDDEN_DIM,
    ) -> None:
        self.model_id = model_id
        self._n_blocks = n_blocks
        self.servers = {server.peer_id: server for server in servers}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.clock = clock or ManualClock()
        self.dim = dim
        self._blocks = _BlockCache(dim)
        self._sketch_spec = SketchSpec()

    # ---- backend protocol ------------------------------------------------------

    def discover(self, model_id: str) -> list[ServerInfo]:
        """Servers currently online and announcing blocks of this model."""
        return [
            server.info()
            for server in self.servers.values()
            if server.online and server.block_range.model_id == model_id
        ]

    def n_blocks(self, model_id: str) -> int:
        if model_id != self.model_id:
            raise KeyError(f"unknown model {model_id!r}")
        return self._n_blocks

    def run_segment(
        self,
        server: ServerInfo,
        block_range: BlockRange,
        hidden: np.ndarray,
        *,
        nonce: bytes,
        step: int = 0,
    ) -> SegmentResult:
        sim = self.servers[server.peer_id]

        if not sim.online:
            return SegmentResult(Outcome.TIMEOUT, None, None, 5000.0, "server offline")

        latency = max(
            1.0,
            self.rng.gauss(sim.hardware.latency_mean_ms, sim.hardware.latency_jitter_ms),
        ) * len(block_range)

        failure_rate = sim.failure_rate
        if sim.behaviour is Behaviour.FLAKY:
            failure_rate = max(failure_rate, 0.25)
        if self.rng.random() < failure_rate:
            return SegmentResult(Outcome.ERROR, None, None, latency, "simulated failure")

        sim.requests_served += 1
        output = self._compute(sim, block_range, hidden)

        seed = derive_seed(nonce, block_range.key(), step)
        sketch = sketch_hidden_state(output, seed=seed, spec=self._sketch_spec)
        return SegmentResult(Outcome.OK, output, sketch, latency)

    # ---- mechanics -------------------------------------------------------------

    def _honest_forward(self, block_range: BlockRange, hidden: np.ndarray) -> np.ndarray:
        state = np.asarray(hidden, dtype=np.float64)
        for index in range(block_range.start, block_range.end):
            weights, bias = self._blocks.get(index)
            # Residual + tanh keeps magnitudes bounded over many blocks, like a real stack.
            state = state + np.tanh(state @ weights + bias)
        return state

    def _compute(
        self, sim: SimServer, block_range: BlockRange, hidden: np.ndarray
    ) -> np.ndarray:
        base = np.asarray(hidden, dtype=np.float64)

        if sim.behaviour is Behaviour.LAZY:
            # Echo the input, plus hardware noise so it does not look artificially perfect.
            return self._add_noise(base, sim.hardware.noise)

        if sim.behaviour is Behaviour.BYZANTINE:
            scale = float(np.linalg.norm(base) / max(np.sqrt(base.size), 1.0))
            return self.np_rng.standard_normal(base.shape) * max(scale, 1e-3)

        honest = self._honest_forward(block_range, base)

        if sim.behaviour is Behaviour.SUBTLE:
            # 30x the worst honest noise: corrupts results, but stays within an order of
            # magnitude of legitimate hardware disagreement.
            return self._add_noise(honest, sim.hardware.noise * 30.0)

        return self._add_noise(honest, sim.hardware.noise)

    def _add_noise(self, array: np.ndarray, magnitude: float) -> np.ndarray:
        if magnitude <= 0:
            return array
        scale = float(np.linalg.norm(array) / max(np.sqrt(array.size), 1.0))
        return array + self.np_rng.standard_normal(array.shape) * magnitude * max(scale, 1e-6)

    def churn(self) -> None:
        """Advance the online/offline Markov process for every server."""
        for server in self.servers.values():
            if server.online:
                if self.rng.random() < server.go_offline_prob:
                    server.online = False
            elif self.rng.random() < server.come_online_prob:
                server.online = True

    def fresh_hidden(self, tokens: int = 4) -> np.ndarray:
        return self.np_rng.standard_normal((tokens, self.dim))

    def sketch_of(self, hidden: np.ndarray, nonce: bytes, block_range: BlockRange, step: int) -> Sketch:
        return sketch_hidden_state(
            hidden, seed=derive_seed(nonce, block_range.key(), step), spec=self._sketch_spec
        )


# ---- swarm construction --------------------------------------------------------


@dataclass
class SwarmBuilder:
    """Builds populated swarms for scenarios, with sensible defaults."""

    model_id: str = "sim-model-32"
    n_blocks: int = 32
    seed: int = 7
    _servers: list[SimServer] = field(default_factory=list)
    _counter: int = 0
    _asn_counter: int = 64500

    def add(
        self,
        behaviour: Behaviour,
        block_range: tuple[int, int],
        *,
        hardware: str = "rtx3090_fp16",
        operator: Optional[str] = None,
        asn: Optional[int] = None,
        address: Optional[str] = None,
        first_seen: float = 0.0,
        count: int = 1,
    ) -> list[SimServer]:
        """Add ``count`` servers. Servers sharing an ``operator`` also share an ASN.

        Sharing the ASN is the point: it is what the diversity rules are supposed to
        notice, so a sybil fleet that did not share one would not be testing anything.
        """
        created: list[SimServer] = []
        operator = operator or f"op{self._counter}"
        if asn is None:
            self._asn_counter += 1
            asn = self._asn_counter

        for index in range(count):
            self._counter += 1
            peer_id = f"sm{behaviour.value[:3]}{self._counter:04d}"
            octet = self._counter % 250 + 1
            created.append(
                SimServer(
                    peer_id=peer_id,
                    block_range=BlockRange(self.model_id, block_range[0], block_range[1]),
                    behaviour=behaviour,
                    hardware=HARDWARE[hardware],
                    address=address or f"198.{asn % 200 + 18}.{index % 250 + 1}.{octet}",
                    asn=asn,
                    first_seen=first_seen,
                    operator=operator,
                )
            )
        self._servers.extend(created)
        return created

    def build(self, clock: Optional[ManualClock] = None) -> SimWorld:
        return SimWorld(
            self.model_id,
            self.n_blocks,
            self._servers,
            seed=self.seed,
            clock=clock,
        )

    @property
    def servers(self) -> list[SimServer]:
        return list(self._servers)
