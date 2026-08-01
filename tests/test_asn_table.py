"""Offline IP-to-ASN table lookup.

These build tiny synthetic tables rather than needing the 8.5 MiB real one, so they run
anywhere. `tools/fetch_asn_table.py --verify` covers the real table against known
allocations (8.8.8.8 -> AS15169 and friends).

Correctness here is not cosmetic: the ASN decides a peer's anti-sybil cluster, so a lookup
that silently returns the wrong answer either merges independent operators into one cluster
(weakening the defence) or splits one operator into several (defeating it entirely).
"""

from __future__ import annotations

import gzip

import pytest

from seedmesh.reputation.diversity import (
    UNKNOWN_CLUSTER,
    ClusterIndex,
    TableAsnResolver,
)

TABLE = "\n".join(
    [
        "1.0.0.0\t1.0.0.255\t13335\tUS\tCLOUDFLARENET",
        "8.8.8.0\t8.8.8.255\t15169\tUS\tGOOGLE",
        "8.8.9.0\t8.8.9.255\t0\tNone\tNot routed",
        "203.0.113.0\t203.0.113.255\t64500\tZZ\tEXAMPLE",
        "2606:4700::\t2606:4700:ffff:ffff:ffff:ffff:ffff:ffff\t13335\tUS\tCLOUDFLARENET",
    ]
) + "\n"


@pytest.fixture
def resolver(tmp_path):
    path = tmp_path / "ip2asn.tsv"
    path.write_text(TABLE, encoding="utf-8")
    return TableAsnResolver.from_file(path)


def test_resolves_a_known_range(resolver):
    assert resolver.resolve("8.8.8.8") == 15169
    assert resolver.resolve("1.0.0.42") == 13335


def test_range_bounds_are_inclusive(resolver):
    assert resolver.resolve("8.8.8.0") == 15169
    assert resolver.resolve("8.8.8.255") == 15169


def test_address_in_a_gap_resolves_to_none(resolver):
    """A gap must not fall through to the neighbouring range."""
    assert resolver.resolve("8.8.7.255") is None
    assert resolver.resolve("2.0.0.1") is None


def test_unrouted_asn_zero_is_treated_as_unknown(resolver):
    """ASN 0 means 'not routed'; it must not become a cluster of its own."""
    assert resolver.resolve("8.8.9.1") is None


def test_addresses_beyond_the_table_resolve_to_none(resolver):
    assert resolver.resolve("0.0.0.1") is None
    assert resolver.resolve("255.255.255.255") is None


def test_ipv6_is_supported(resolver):
    assert resolver.resolve("2606:4700:4700::1111") == 13335
    assert resolver.resolve("2001:db8::1") is None


def test_v4_and_v6_tables_do_not_bleed_into_each_other(resolver):
    """Both are integer ranges; mixing them would produce nonsense matches."""
    assert resolver.resolve("::1") is None
    assert len(resolver) == 4  # the ASN-0 row is dropped


def test_malformed_rows_are_skipped(tmp_path):
    path = tmp_path / "messy.tsv"
    path.write_text(
        "range_start\trange_end\tAS_number\n"        # header
        "garbage\n"                                    # too few columns
        "notanip\t1.2.3.4\t123\n"                      # unparseable address
        "1.2.3.0\t1.2.3.255\tnotanint\n"               # unparseable ASN
        "5.5.5.0\t5.5.5.255\t99\tUS\tOK\n",
        encoding="utf-8",
    )
    resolver = TableAsnResolver.from_file(path)
    assert len(resolver) == 1
    assert resolver.resolve("5.5.5.5") == 99


def test_gzipped_tables_load(tmp_path):
    path = tmp_path / "ip2asn.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(TABLE)
    resolver = TableAsnResolver.from_file(path)
    assert resolver.resolve("8.8.8.8") == 15169


def test_invalid_address_resolves_to_none(resolver):
    assert resolver.resolve("not-an-ip") is None
    assert resolver.resolve("") is None


def test_lookup_is_cached_and_stable(resolver):
    first = resolver.resolve("8.8.8.8")
    assert resolver.resolve("8.8.8.8") == first
    assert resolver.resolve("8.8.7.255") is None
    assert resolver.resolve("8.8.7.255") is None


def test_empty_table_resolves_everything_to_none():
    resolver = TableAsnResolver()
    assert len(resolver) == 0
    assert resolver.resolve("8.8.8.8") is None


# ---- what this buys the diversity rules -------------------------------------


def _server(peer_id, address, first_seen=0.0):
    from seedmesh.core.types import BlockRange, ServerInfo

    return ServerInfo(peer_id, BlockRange("m", 0, 4), first_seen, address=address)


def test_real_asns_separate_operators_that_prefixes_would_merge(resolver):
    """Two peers in unrelated ASNs are independent even if prefix rules were coarse."""
    index = ClusterIndex(resolver)
    google = _server("smgoogle", "8.8.8.8")
    cloudflare = _server("smcloudflare", "1.0.0.42", first_seen=50_000.0)

    assert index.coarse_of(google) == "asn:15169"
    assert index.coarse_of(cloudflare) == "asn:13335"
    assert index.are_independent(google, cloudflare)


def test_same_asn_peers_are_not_independent(resolver):
    """The case the defence exists for: one operator, many addresses, one AS."""
    index = ClusterIndex(resolver)
    left = _server("smleft", "8.8.8.1")
    right = _server("smright", "8.8.8.200", first_seen=50_000.0)

    assert index.coarse_of(left) == index.coarse_of(right) == "asn:15169"
    assert not index.are_independent(left, right)


def test_unroutable_addresses_fall_back_to_prefix_clustering(resolver):
    """Private/loopback peers have no AS, so clustering degrades to address prefixes."""
    index = ClusterIndex(resolver)
    private = _server("smprivate", "192.168.1.5")
    assert index.coarse_of(private).startswith("net:")
    assert index.coarse_of(private) != UNKNOWN_CLUSTER
