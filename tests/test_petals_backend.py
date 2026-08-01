"""Unit tests for the pure logic in the Petals adapter.

The adapter imports torch and petals lazily *inside* its methods, so this module is
importable — and this address-selection logic testable — on a machine with neither. That
is deliberate: the choice of which address represents a peer decides its anti-sybil
cluster, so it deserves tests that run everywhere rather than only on a WSL box with a
swarm up.
"""

from __future__ import annotations

from seedmesh.backends.petals_backend import PetalsBackend

rank = PetalsBackend._address_rank


def best(*addresses: str) -> str:
    return sorted(addresses, key=rank)[0]


def test_public_address_beats_private_and_loopback():
    """A loopback address says nothing about who operates a peer.

    Note the example is a genuinely globally-routable address. 203.0.113.0/24 looks public
    but is RFC 5737 documentation space, which `ipaddress` reports as non-global -- the
    distinction that `is_global` gets right and `not is_private` does not.
    """
    assert best("127.0.0.1", "192.168.1.5", "93.184.216.34") == "93.184.216.34"


def test_private_beats_loopback():
    assert best("127.0.0.1", "10.1.2.3") == "10.1.2.3"


def test_loopback_is_chosen_only_when_it_is_all_there_is():
    assert best("127.0.0.1") == "127.0.0.1"


def test_unparseable_address_ranks_last():
    """Garbage must never outrank a real address."""
    assert best("not-an-ip", "127.0.0.1") == "127.0.0.1"
    assert best("not-an-ip", "93.184.216.34") == "93.184.216.34"


def test_link_local_ranks_below_public():
    assert best("169.254.1.1", "93.184.216.34") == "93.184.216.34"


def test_documentation_range_is_not_treated_as_public():
    """Regression: `not is_private` classified RFC 5737 space as a public address."""
    assert rank("203.0.113.7")[0] == 1
    assert rank("93.184.216.34")[0] == 0


def test_ipv6_public_beats_ipv6_loopback():
    assert best("::1", "2606:4700:4700::1111") == "2606:4700:4700::1111"


def test_ordering_is_total_and_deterministic():
    """Equal-class addresses must break ties consistently, or clustering flaps."""
    addresses = ["203.0.113.9", "203.0.113.2", "198.51.100.4"]
    assert sorted(addresses, key=rank) == sorted(addresses, key=rank)
    assert best(*addresses) == "198.51.100.4"


def test_distinct_addresses_produce_distinct_clusters_under_an_asn_table():
    """The end the resolution serves: addresses only matter if they separate operators."""
    from seedmesh.core.types import BlockRange, ServerInfo
    from seedmesh.reputation.diversity import ClusterIndex, StaticAsnResolver

    block_range = BlockRange("m", 0, 4)
    left = ServerInfo("smleft", block_range, 0.0, address="127.0.0.2")
    right = ServerInfo("smright", block_range, 50_000.0, address="127.0.0.3")

    # Without an ASN table both loopback addresses share a /16 -- correctly "same network".
    plain = ClusterIndex()
    assert not plain.are_independent(left, right)

    # With one, they separate, which is what makes them a usable verification pair.
    geo = ClusterIndex(StaticAsnResolver({"127.0.0.2": 64501, "127.0.0.3": 64502}))
    assert geo.are_independent(left, right)


def test_unresolved_address_stays_unpairable():
    """A peer that has announced but never been dialled must not be verifiable."""
    from seedmesh.core.types import BlockRange, ServerInfo
    from seedmesh.reputation.diversity import UNKNOWN_CLUSTER, ClusterIndex

    block_range = BlockRange("m", 0, 4)
    known = ServerInfo("smknown", block_range, 0.0, address="203.0.113.7")
    unknown = ServerInfo("smunknown", block_range, 50_000.0, address=None)

    index = ClusterIndex()
    assert index.coarse_of(unknown) == UNKNOWN_CLUSTER
    assert not index.are_independent(known, unknown)
