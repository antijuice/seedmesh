"""Model catalog. See ``registry.yaml`` -- planning artifact until a backend adapter exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")


def load_registry() -> dict[str, Any]:
    """Parse the launch catalog.

    Entries carry ``status`` and ``blocked_by``; nothing here is runnable until a backend
    adapter exists, and callers should not treat a listed model as available.
    """
    import yaml

    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


__all__ = ["REGISTRY_PATH", "load_registry"]
