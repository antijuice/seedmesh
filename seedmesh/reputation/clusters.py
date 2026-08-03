"""Choosing how peers get placed on networks, for a live client.

Verification prices influence in *network diversity*, because identities are free and
networks are not. That only works if two peers under one operator land in the same cluster.

**The default was measuring the wrong thing.** With no resolver, `ClusterIndex` falls back to
address prefixes, so two DigitalOcean droplets in different /16s look like different
operators. They are not: one account, one bill, one person able to make them agree. The
swarm's first verification pair was exactly that — accepted by prefix, and an ASN-aware
resolver refuses it.

So a client loads the real table when it has one, and **says which it is using**. A swarm
that cannot tell operators apart must not look like one that checked and found them
independent — the same rule the rest of this project runs on.

Loading is not free: ~3.4 s and ~67 MiB resident for 573,088 routed ranges (measured; see
`tools/fetch_asn_table.py`). That is paid once per client, off the request path, and buys
the only signal that makes the diversity constraint mean anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

ENV_VAR = "SEEDMESH_ASN_TABLE"

# Where `tools/fetch_asn_table.py` puts it, relative to the repo root.
PACKAGED = Path(__file__).resolve().parents[2] / "data" / "ip2asn-combined.tsv.gz"


def table_path(explicit: Optional[str] = None) -> Optional[Path]:
    """The ASN table to use, or None if there is not one.

    Explicit argument, then $SEEDMESH_ASN_TABLE, then the packaged location -- the same
    precedence the swarm file uses, so there is one rule to remember.
    """
    for candidate in (explicit, os.environ.get(ENV_VAR)):
        if candidate:
            path = Path(candidate).expanduser()
            return path if path.is_file() else None
    return PACKAGED if PACKAGED.is_file() else None


def build_cluster_index(explicit: Optional[str] = None) -> Tuple[object, List[str]]:
    """(ClusterIndex, lines to print). Never raises -- a bad table degrades, it does not stop.

    Returns the prefix-based fallback if the table is missing or unreadable, and says so.
    Silence here would be the worst outcome: the diversity rule would appear to be working
    while grading two machines in one datacentre as independent operators.
    """
    from seedmesh.reputation.diversity import ClusterIndex, TableAsnResolver

    path = table_path(explicit)
    if path is None:
        return ClusterIndex(), [
            "  clustering  by address prefix (no ASN table)",
            "    two machines in one datacentre may count as independent operators, so"
            " verification",
            "    pairs are weaker than they look. Fetch the table:"
            " python tools/fetch_asn_table.py",
        ]

    try:
        resolver = TableAsnResolver.from_file(path)
    except Exception as exc:
        return ClusterIndex(), [
            f"  clustering  by address prefix -- could not read {path.name}: {exc}",
            "    re-fetch with: python tools/fetch_asn_table.py",
        ]

    return ClusterIndex(resolver), [f"  clustering  by ASN ({path.name})"]
