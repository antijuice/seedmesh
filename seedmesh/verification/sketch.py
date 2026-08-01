"""Compact, comparison-friendly fingerprints of hidden states.

The spec (section 3.2) proposes sending "a hash/fingerprint of the hidden state, not the
full tensor, to keep overhead low". The bandwidth instinct is right and the hash is wrong,
for a reason that would only have surfaced after deployment:

**Two honest servers do not produce bit-identical hidden states.** Floating-point addition
is not associative, so results depend on reduction order, which depends on kernel choice,
which depends on GPU architecture, batch shape, library version and dtype. An A100 running
bf16 and a 3090 running fp16 will agree to a few decimal places and disagree in the low
bits, every single time. Under exact hash comparison every honest heterogeneous pair
mismatches -- and heterogeneous hardware is the entire premise of a volunteer swarm. The
check would have had a ~100% false-positive rate and evicted the network's honest nodes.

The fix keeps the bandwidth saving while tolerating numerical noise: project the hidden
state onto ``k`` pseudo-random unit vectors and send those ``k`` floats. By the
Johnson-Lindenstrauss lemma the projection approximately preserves relative distances, so
honest pairs stay close and a server that fabricated its output lands far away.

Two properties make it hard to game:

* **The seed is per-request and unpredictable.** A server that knew the projection in
  advance could craft a wrong tensor whose sketch matched. Seeds are derived from a
  client-chosen nonce, so the server must commit to its output before learning the basis.
* **The seed is derived, not transmitted.** Both sides compute the same basis from the same
  nonce, so nothing extra goes on the wire.

Cost: a 4096-dim hidden state over 512 tokens is ~2M floats; a 64-component sketch is 64.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_COMPONENTS = 64
SKETCH_CONTEXT = b"seedmesh/sketch/v1"


@dataclass(frozen=True, slots=True)
class SketchSpec:
    """Parameters both sides must agree on for sketches to be comparable."""

    components: int = DEFAULT_COMPONENTS
    include_norm: bool = True
    """Carry the tensor's L2 norm alongside the projection. A projection alone is scale-
    sensitive but norm-ambiguous under some degenerate fabrications; the norm makes
    'returned something of plausible shape but wrong magnitude' immediately visible."""

    def __post_init__(self) -> None:
        if self.components < 8:
            raise ValueError("sketches need at least 8 components to be discriminative")


def derive_seed(nonce: bytes, block_range_key: str, step: int) -> int:
    """Derive a sketch seed from a client nonce, bound to this block range and step.

    Binding to the range and step stops a server from replaying a sketch it computed for a
    different position in the pipeline.
    """
    if len(nonce) < 16:
        raise ValueError("nonce must be at least 16 bytes to be unpredictable")
    digest = hashlib.sha256(
        SKETCH_CONTEXT + b"\x00" + nonce + b"\x00" + block_range_key.encode("utf-8")
        + b"\x00" + str(int(step)).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _projection_matrix(seed: int, dim: int, components: int) -> np.ndarray:
    """Deterministic Gaussian projection basis, normalized to unit columns."""
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((dim, components), dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix


@dataclass(frozen=True, slots=True)
class Sketch:
    """A fingerprint of a hidden state: k projections plus optional norm and shape."""

    values: np.ndarray
    norm: float
    dim: int
    tokens: int

    def to_wire(self) -> dict:
        return {
            "values": [float(v) for v in self.values],
            "norm": float(self.norm),
            "dim": int(self.dim),
            "tokens": int(self.tokens),
        }

    @classmethod
    def from_wire(cls, data: dict) -> "Sketch":
        return cls(
            values=np.asarray(data["values"], dtype=np.float64),
            norm=float(data["norm"]),
            dim=int(data["dim"]),
            tokens=int(data["tokens"]),
        )

    @property
    def components(self) -> int:
        return int(self.values.shape[0])


def sketch_hidden_state(
    hidden: np.ndarray,
    *,
    seed: int,
    spec: Optional[SketchSpec] = None,
) -> Sketch:
    """Fingerprint a hidden state of shape ``(tokens, dim)`` or ``(dim,)``.

    Computation is forced to float64 before projecting. The point is to compare *semantic*
    agreement between servers, so the comparison itself must not add the very fp16/bf16
    noise it is trying to tolerate.
    """
    spec = spec or SketchSpec()
    array = np.asarray(hidden)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D (tokens, dim) hidden state, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("hidden state contains non-finite values")

    array = array.astype(np.float64, copy=False)
    tokens, dim = array.shape

    matrix = _projection_matrix(seed, dim, spec.components)
    # Mean over tokens keeps the sketch a fixed size regardless of sequence length, so
    # comparison does not depend on both servers seeing identical batch shapes.
    projected = (array @ matrix).mean(axis=0)
    norm = float(np.linalg.norm(array) / np.sqrt(tokens)) if spec.include_norm else 0.0

    return Sketch(values=projected, norm=norm, dim=dim, tokens=tokens)
