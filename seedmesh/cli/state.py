"""Where a Seedmesh node keeps the things that must outlive a process.

Two of them, and they are different in kind:

**Identity** (`identity.key`) is who this node *is* to the rest of the swarm. Every
reputation record it publishes is signed with this key, and other peers' anti-replay and
per-observer weight caps are keyed to it. Regenerating it every run would make each session a
brand-new stranger with no history, which is exactly the sybil shape the aggregator is built
to discount -- an honest client would look like an attacker.

**Reputation** (`reputation/<swarm>.json`) is what this node has *learned*. Kept per swarm,
because a peer id means nothing across swarms and merging two swarms' evidence into one file
would let a peer carry a score it never earned here.

`SEEDMESH_HOME` overrides the location, which is what makes any of this testable without
writing to a developer's real home directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from seedmesh.core.identity import Identity

IDENTITY_FILE = "identity.key"


def seedmesh_home() -> Path:
    override = os.environ.get("SEEDMESH_HOME")
    return Path(override) if override else Path.home() / ".seedmesh"


def identity_path() -> Path:
    return seedmesh_home() / IDENTITY_FILE


def _safe_name(name: str) -> str:
    """Make a swarm name usable as a filename without letting it escape the directory.

    Swarm names arrive from a JSON file that someone else may have written, so `../../` and
    absolute paths have to stop here rather than at the filesystem.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return cleaned or "swarm"


def reputation_path(swarm_name: str) -> Path:
    return seedmesh_home() / "reputation" / f"{_safe_name(swarm_name)}.json"


def load_or_create_identity(path: Optional[Path] = None) -> tuple[Identity, bool]:
    """Load this node's signing key, creating it on first run.

    Returns ``(identity, created)``. A key file that exists but cannot be parsed is left
    alone and a fresh in-memory identity is returned instead: silently overwriting it would
    destroy the node's accumulated standing in every swarm at once, and the cost of not
    overwriting it is one session published under a new id.
    """
    target = Path(path) if path is not None else identity_path()
    if target.exists():
        try:
            return Identity.from_private_bytes(target.read_bytes()), False
        except Exception:
            return Identity.generate(), False

    identity = Identity.generate()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(identity.private_bytes())
    try:
        # Best effort -- Windows ignores the group/other bits, but on the Linux hosts that
        # actually serve, a world-readable signing key is a real problem.
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(target)
    return identity, True
