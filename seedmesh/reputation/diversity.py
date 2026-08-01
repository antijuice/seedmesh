"""Network-locality clustering, the substrate for every anti-sybil rule in Seedmesh.

Peer ids are free: an attacker can mint ten thousand keypairs in a second. What is *not*
free is network diversity -- addresses in many unrelated /16s across many autonomous
systems cost real money. So every place Seedmesh would otherwise count peers, it counts
*clusters* instead.

Two rules here are load-bearing and easy to get wrong:

1. **Address-less peers share one bucket.** If an unknown address produced a unique
   cluster per peer, an attacker would simply omit their address and mint unlimited
   clusters -- turning the defence into its own bypass. All peers with no resolvable
   address collectively get exactly one cluster's worth of influence.

2. **ASN is resolved locally, never self-reported.** A peer claiming "I am in AS64512"
   is claiming to be diverse. :class:`AsnResolver` is a local lookup, and
   :class:`NullAsnResolver` (the default) falls back to address prefixes rather than
   trusting the peer.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from seedmesh.core.types import PeerId, ServerInfo

UNKNOWN_CLUSTER = "cluster:unknown"
"""Single shared bucket for peers whose network location cannot be determined."""

COARSE_V4_PREFIX = 16
COARSE_V6_PREFIX = 32
FINE_V4_PREFIX = 24
FINE_V6_PREFIX = 48


class AsnResolver(Protocol):
    """Maps an IP address to an autonomous system number, locally."""

    def resolve(self, address: str) -> Optional[int]:  # pragma: no cover - protocol
        ...


class NullAsnResolver:
    """Default resolver: no ASN data, fall back to address prefixes.

    Deliberately not a stub to be filled in later with a network call -- a blocking DNS or
    HTTP lookup on the routing hot path would be a denial-of-service vector. A real
    deployment should load an offline table (MaxMind GeoLite ASN, Team Cymru bulk export)
    and answer from memory.
    """

    __slots__ = ()

    def resolve(self, address: str) -> Optional[int]:
        return None


class StaticAsnResolver:
    """Resolver backed by an in-memory table. Used by tests and the simulator."""

    __slots__ = ("_table",)

    def __init__(self, table: dict[str, int]) -> None:
        self._table = dict(table)

    def resolve(self, address: str) -> Optional[int]:
        return self._table.get(address)


def _prefix_cluster(address: str, v4_bits: int, v6_bits: int) -> Optional[str]:
    try:
        addr = ipaddress.ip_address(address)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv4Address):
        net = ipaddress.ip_network(f"{addr}/{v4_bits}", strict=False)
    else:
        net = ipaddress.ip_network(f"{addr}/{v6_bits}", strict=False)
    return f"net:{net.with_prefixlen}"


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """Where a peer sits in the network topology, at two granularities."""

    peer_id: PeerId
    coarse: str
    fine: str
    asn: Optional[int] = None

    @property
    def is_unknown(self) -> bool:
        return self.coarse == UNKNOWN_CLUSTER


class ClusterIndex:
    """Assigns peers to network clusters and answers diversity questions."""

    def __init__(self, resolver: Optional[AsnResolver] = None) -> None:
        self._resolver: AsnResolver = resolver or NullAsnResolver()
        self._cache: dict[PeerId, ClusterAssignment] = {}

    def assign(self, info: ServerInfo) -> ClusterAssignment:
        """Compute (and memoize) the cluster assignment for a peer."""
        cached = self._cache.get(info.peer_id)
        if cached is not None:
            return cached

        asn = info.asn
        if asn is None and info.address:
            asn = self._resolver.resolve(info.address)

        if asn is not None:
            coarse = f"asn:{asn}"
            fine = _prefix_cluster(info.address, FINE_V4_PREFIX, FINE_V6_PREFIX) if info.address else coarse
            assignment = ClusterAssignment(info.peer_id, coarse, fine or coarse, asn)
        elif info.address:
            coarse = _prefix_cluster(info.address, COARSE_V4_PREFIX, COARSE_V6_PREFIX)
            fine = _prefix_cluster(info.address, FINE_V4_PREFIX, FINE_V6_PREFIX)
            if coarse is None:
                assignment = ClusterAssignment(info.peer_id, UNKNOWN_CLUSTER, UNKNOWN_CLUSTER)
            else:
                assignment = ClusterAssignment(info.peer_id, coarse, fine or coarse)
        else:
            # See module docstring, rule 1: one shared bucket, never one bucket per peer.
            assignment = ClusterAssignment(info.peer_id, UNKNOWN_CLUSTER, UNKNOWN_CLUSTER)

        self._cache[info.peer_id] = assignment
        return assignment

    def coarse_of(self, info: ServerInfo) -> str:
        return self.assign(info).coarse

    def distinct_coarse(self, infos: Iterable[ServerInfo]) -> set[str]:
        return {self.assign(info).coarse for info in infos}

    def are_independent(
        self,
        a: ServerInfo,
        b: ServerInfo,
        *,
        min_first_seen_gap_s: float = 0.0,
    ) -> bool:
        """Whether two peers are plausibly under different operators.

        Used to pick verification pairs (spec section 9: "add a diversity constraint when
        selecting the verification pair"). Two peers are *not* independent if they share a
        coarse cluster, if either is in the unknown bucket, or if they appeared within
        ``min_first_seen_gap_s`` of each other -- simultaneous arrival is the signature of
        one operator starting a fleet with a single script.
        """
        if a.peer_id == b.peer_id:
            return False
        ca, cb = self.assign(a), self.assign(b)
        if ca.is_unknown or cb.is_unknown:
            return False
        if ca.coarse == cb.coarse:
            return False
        if min_first_seen_gap_s > 0 and abs(a.first_seen - b.first_seen) < min_first_seen_gap_s:
            return False
        return True

    def forget(self, peer_id: PeerId) -> None:
        self._cache.pop(peer_id, None)
