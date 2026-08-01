"""The backend seam.

Everything above this line -- reputation, verification, routing -- is pure Python over
numpy and knows nothing about transformers, CUDA or gRPC. This module defines the narrow
interface a real backend must implement to be driven by it.

The seam exists for a specific, evidenced reason. Petals has been unmaintained since
2024-09-07, hard-pins ``transformers==4.43.1``, and subclasses transformers internals that
the 4.48+ attention refactor rewrote (see ``docs/findings-upstream-audit.md``). Binding the
project's novel work directly to that codebase would make it inherit an unbounded porting
risk. Instead the backend is a plug: a Petals adapter, a llama.cpp RPC adapter and the
simulator all satisfy the same three methods, and the trust layer cannot tell them apart.

Note what the interface deliberately does *not* expose: prompt text. Block-hosting servers
see activations only. Keeping raw text out of this signature is what makes that a
structural property rather than a convention -- see ``docs/security-privacy.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from seedmesh.core.types import BlockRange, Outcome, ServerInfo
from seedmesh.verification.sketch import Sketch


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """Outcome of running one contiguous block range on one server."""

    outcome: Outcome
    hidden: Optional[np.ndarray]
    sketch: Optional[Sketch]
    latency_ms: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome.is_success and self.hidden is not None


@runtime_checkable
class InferenceBackend(Protocol):
    """What a concrete transport must provide for the trust layer to drive it."""

    def discover(self, model_id: str) -> Sequence[ServerInfo]:
        """Return servers currently announcing blocks of ``model_id``.

        For Petals this wraps a hivemind DHT lookup of the block-announcement records.
        """
        ...

    def n_blocks(self, model_id: str) -> int:
        """Total transformer block count for ``model_id``."""
        ...

    def run_segment(
        self,
        server: ServerInfo,
        block_range: BlockRange,
        hidden: np.ndarray,
        *,
        nonce: bytes,
        step: int = 0,
    ) -> SegmentResult:
        """Run ``block_range`` on ``server``, starting from ``hidden``.

        Must be a pure function of ``(block_range, hidden)`` up to the server's own
        numerical noise. Redundant verification depends on this: two honest servers given
        the same input and range must produce comparable output, or the whole check is
        meaningless.

        ``nonce`` and ``step`` derive the sketch seed. The server must not be able to
        predict them before committing to its output -- see
        :mod:`seedmesh.verification.sketch`.
        """
        ...
