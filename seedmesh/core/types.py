"""Domain vocabulary shared by the reputation and verification layers.

Nothing here imports a machine-learning framework. A ``BlockRange`` is a half-open
interval of transformer block indices; whether those blocks are Petals ``WrappedLlamaBlock``
objects or llama.cpp RPC shards is the backend's business, not ours.
"""

from __future__ import annotations

import enum
import ipaddress
from dataclasses import dataclass, field
from typing import Optional

PeerId = str
"""Self-certifying peer identifier: ``sm`` + base32(sha256(ed25519 public key)[:20]).

Self-certifying matters. Because the id is derived from the key, a signed record can be
checked against its claimed author with no registry, no CA and no trusted introducer --
which is the only way reputation gossip works in a permissionless swarm.
"""


class Outcome(enum.Enum):
    """How a single request against a single server ended."""

    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"
    REFUSED = "refused"

    @property
    def is_success(self) -> bool:
        return self is Outcome.OK

    @property
    def counts_against_reliability(self) -> bool:
        """Whether this outcome should be held against the server.

        REFUSED is deliberately excluded. A server that cleanly refuses because it is at
        capacity is behaving correctly and honestly; punishing it would push operators
        toward accepting work they cannot do, which is strictly worse for the swarm.
        """
        return self in (Outcome.TIMEOUT, Outcome.ERROR)


class Verdict(enum.Enum):
    """Result of comparing two servers' work on the same input."""

    MATCH = "match"
    MISMATCH = "mismatch"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True, order=True)
class BlockRange:
    """A half-open interval ``[start, end)`` of transformer blocks of one model."""

    model_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"block range start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"block range must be non-empty and half-open, got [{self.start}, {self.end})"
            )
        if not self.model_id:
            raise ValueError("block range requires a model_id")

    def __len__(self) -> int:
        return self.end - self.start

    def covers(self, other: "BlockRange") -> bool:
        return (
            self.model_id == other.model_id
            and self.start <= other.start
            and self.end >= other.end
        )

    def overlaps(self, other: "BlockRange") -> bool:
        return (
            self.model_id == other.model_id
            and self.start < other.end
            and other.start < self.end
        )

    def key(self) -> str:
        """Stable string key, used for DHT record addressing and dict indexing."""
        return f"{self.model_id}:{self.start}-{self.end}"

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True, slots=True)
class ComputeProfile:
    """How a server computes, insofar as it changes the *numbers* it produces.

    This must be published in a server's block announcement, because verification compares
    two servers' outputs against a calibrated tolerance and **the tolerance depends on this
    profile**. Measurement (``spike/quantization/``): two honest servers holding identical
    weights disagree by ~0.005 across fp16/bf16/int8, but by **~0.055 when one of them is
    NF4** -- an order of magnitude more. Compared against a single global threshold, an
    honest NF4 volunteer lands in the ambiguous band on every single check and is
    quarantined after six of them.

    Treating this as an optional detail is how heterogeneous honest hardware gets read as
    fraud, which is the failure mode this project has now hit four separate ways.
    """

    quant: str = "none"
    """``none`` | ``int8`` | ``nf4``. Dominates the numerical noise floor by ~10x."""

    dtype: str = "fp16"
    """``fp16`` | ``bf16`` | ``fp32``."""

    attn: str = "eager"
    """Attention kernel. Hosting a bare layer leaves this unset in transformers and the
    fallback is silent, so it is pinned and published rather than inferred."""

    experts: str = "none"
    """Mixture-of-experts kernel: ``none`` for dense models, else the transformers
    ``_experts_implementation`` (``grouped_mm`` | ``batched_mm`` | ``deepgemm`` |
    ``sonicmoe``).

    Same reasoning as ``attn``, and the same measured trap. A hosted block has no model
    wrapper to set this dispatch, and on a tiny Mixtral config (CPU/fp32) the unset default
    and ``batched_mm`` both return **NaN** while ``grouped_mm`` returns finite values --
    so it cannot be left implicit. Two honest servers on different expert kernels will
    disagree numerically for the same reason two servers on different attention kernels do,
    and an unpublished difference is read as fraud.

    ``none`` keeps the key stable for the dense models that are all this has been calibrated
    against; MoE tolerances are not yet measured."""

    @property
    def key(self) -> str:
        # `experts` is appended only when set, so every existing dense profile key and every
        # calibrated tolerance entry keeps its current value.
        base = f"{self.quant}/{self.dtype}/{self.attn}"
        return base if self.experts == "none" else f"{base}/{self.experts}"

    def __str__(self) -> str:
        return self.key


DEFAULT_PROFILE = ComputeProfile()


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """What the swarm knows about a block-hosting peer.

    ``address`` and ``asn`` drive the diversity constraints in
    :mod:`seedmesh.reputation.diversity`. They are hints, not secrets: a peer's address is
    observable by anyone who talks to it, and ``asn`` is resolved locally by the observer,
    never taken from the peer's own claims.
    """

    peer_id: PeerId
    block_range: BlockRange
    first_seen: float
    address: Optional[str] = None
    asn: Optional[int] = None
    advertised_throughput: float = 0.0
    """Self-reported tokens/sec. Untrusted -- upstream Petals ranks on this alone, which is
    exactly the ungamed self-report this project exists to replace. Used only as a
    tie-breaker among servers with equal measured reputation."""

    profile: ComputeProfile = DEFAULT_PROFILE
    """How this server computes. Selects the verification tolerance -- see
    :class:`ComputeProfile`.

    Unlike ``advertised_throughput``, a false claim here is self-punishing rather than
    profitable: it selects the wrong tolerance for the liar's own comparisons, so a server
    claiming NF4 while running fp16 gets judged against a band far too wide to help it, and
    one claiming fp16 while running NF4 fails every check it takes part in."""

    @property
    def model_id(self) -> str:
        return self.block_range.model_id

    def parsed_address(self) -> Optional[ipaddress._BaseAddress]:
        if not self.address:
            return None
        try:
            return ipaddress.ip_address(self.address)
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class Observation:
    """One observer's first-hand experience of one request against one server.

    Observations are the atomic unit of reputation. They are always attributed to the
    observer that made them, and they never travel unsigned -- see
    :mod:`seedmesh.reputation.records`.
    """

    observer: PeerId
    subject: PeerId
    block_range: BlockRange
    outcome: Outcome
    latency_ms: float
    at: float
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"latency must be non-negative, got {self.latency_ms}")


@dataclass(slots=True)
class VerificationEvent:
    """Outcome of a redundant-execution check against one server."""

    subject: PeerId
    peer: PeerId
    block_range: BlockRange
    verdict: Verdict
    at: float
    distance: float = 0.0
    threshold: float = 0.0
    detail: dict = field(default_factory=dict)
