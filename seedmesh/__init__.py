"""Seedmesh: a reputation and verification layer for public P2P inference swarms.

This package is deliberately backend-agnostic. It knows about peers, block ranges,
observations and hidden-state fingerprints -- but nothing about PyTorch, Petals or
hivemind. See ``seedmesh.backends.base`` for the seam, and ``docs/architecture.md``
for the reasoning.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
