"""Backend adapters. The core depends on :mod:`seedmesh.backends.base` and nothing else."""

from seedmesh.backends.base import InferenceBackend, SegmentResult

__all__ = ["InferenceBackend", "SegmentResult"]
