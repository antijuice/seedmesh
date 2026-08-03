"""Choosing how a live client places peers on networks.

The default fell back to address prefixes, which grades two machines in one datacentre as
different operators. The swarm's first verification pair was exactly that: two DigitalOcean
droplets, accepted because their /16s differ. One account, one bill, one person able to make
them agree.
"""

from __future__ import annotations

from seedmesh.core.types import BlockRange, ComputeProfile, ServerInfo
from seedmesh.reputation.clusters import ENV_VAR, build_cluster_index, table_path

RANGE = BlockRange("m", 0, 12)


def server(peer_id, address):
    return ServerInfo(
        peer_id=peer_id, block_range=RANGE, first_seen=0.0, address=address,
        profile=ComputeProfile("none", "bfloat16", "eager"),
    )


# ---- choosing the table ------------------------------------------------------


def test_an_explicit_path_wins(tmp_path, monkeypatch):
    chosen = tmp_path / "mine.tsv"
    chosen.write_text("", encoding="utf-8")
    other = tmp_path / "env.tsv"
    other.write_text("", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(other))

    assert table_path(str(chosen)) == chosen


def test_the_env_var_is_used_when_nothing_explicit(tmp_path, monkeypatch):
    table = tmp_path / "env.tsv"
    table.write_text("", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(table))

    assert table_path() == table


def test_a_path_that_does_not_exist_is_not_silently_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "absent.tsv"))
    assert table_path() is None


# ---- degrading honestly ------------------------------------------------------


def test_a_missing_table_falls_back_and_says_so(tmp_path, monkeypatch):
    """Silence would be the worst outcome: the diversity rule would look like it was working
    while grading two machines in one datacentre as independent operators."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "absent.tsv"))
    monkeypatch.setattr("seedmesh.reputation.clusters.PACKAGED", tmp_path / "nope.gz")

    index, lines = build_cluster_index()
    text = "\n".join(lines)
    assert index is not None
    assert "address prefix" in text
    assert "fetch_asn_table" in text, "say how to fix it"


def test_an_unreadable_table_degrades_rather_than_raising(tmp_path, monkeypatch):
    broken = tmp_path / "broken.tsv"
    broken.write_bytes(b"\x00\x01 not a tsv")
    monkeypatch.setenv(ENV_VAR, str(broken))

    index, lines = build_cluster_index()
    assert index is not None, "a bad table must not stop a client from running"


def test_a_working_table_is_named(tmp_path, monkeypatch):
    table = tmp_path / "tiny.tsv"
    table.write_text("1.0.0.0\t1.0.0.255\t13335\tUS\tCLOUDFLARENET\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(table))

    index, lines = build_cluster_index()
    assert "by ASN" in "\n".join(lines)
    assert index.assign(server("p", "1.0.0.5")).coarse == "asn:13335"


# ---- the property the whole thing is for -------------------------------------


def test_two_hosts_in_one_asn_are_not_independent(tmp_path, monkeypatch):
    """The exact pair the live swarm produced: different /16s, same operator. Prefix
    clustering accepts it; ASN clustering must not."""
    table = tmp_path / "do.tsv"
    table.write_text(
        "143.244.128.0\t143.244.191.255\t14061\tUS\tDIGITALOCEAN\n"
        "104.248.0.0\t104.248.255.255\t14061\tUS\tDIGITALOCEAN\n"
        "8.8.8.0\t8.8.8.255\t15169\tUS\tGOOGLE\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_VAR, str(table))
    index, _ = build_cluster_index()

    droplet_a = server("a", "143.244.144.71")
    droplet_b = server("b", "104.248.51.133")
    google = server("c", "8.8.8.8")

    assert index.are_independent(droplet_a, droplet_b) is False
    # And a genuinely different operator still passes -- the rule must discriminate, not
    # simply refuse everything.
    assert index.are_independent(droplet_a, google) is True


def test_prefix_clustering_accepts_what_asn_refuses(tmp_path, monkeypatch):
    """Measured against the live droplets before this was wired in. Recording it so the
    difference cannot quietly regress to 'no difference'."""
    from seedmesh.reputation.diversity import ClusterIndex

    table = tmp_path / "do.tsv"
    table.write_text(
        "143.244.128.0\t143.244.191.255\t14061\tUS\tDIGITALOCEAN\n"
        "104.248.0.0\t104.248.255.255\t14061\tUS\tDIGITALOCEAN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_VAR, str(table))
    by_asn, _ = build_cluster_index()

    a, b = server("a", "143.244.144.71"), server("b", "104.248.51.133")
    assert ClusterIndex().are_independent(a, b) is True
    assert by_asn.are_independent(a, b) is False


def test_an_unrouted_address_falls_back_to_prefix_not_its_own_cluster(tmp_path, monkeypatch):
    """ASN 0 means 'not routed'. Giving it a cluster of its own would make every unrouted
    peer independent of every other, which is backwards."""
    table = tmp_path / "t.tsv"
    table.write_text("1.0.0.0\t1.0.0.255\t0\tNone\tNot routed\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(table))
    index, _ = build_cluster_index()

    coarse = index.assign(server("p", "1.0.0.5")).coarse
    assert coarse is None or coarse.startswith("net:"), coarse
