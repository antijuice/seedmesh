"""Tests for the swarm monitor's report logic.

Everything here runs on the plain mappings `collapse_spans` takes, so no DHT, no Petals, and
no swarm is involved -- the live path is a thin shell over these functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from seedmesh.cli.monitor import (
    ServerRow,
    build_report,
    collapse_spans,
    count_replicas,
    coverage_bar,
    render_html,
    render_json,
    render_text,
)


def announce(peer, **kwargs):
    base = {"state": "ONLINE", "throughput": 100.0}
    base.update(kwargs)
    return (peer, base)


def blocks(n, *assignments):
    """Build per-block announcements: each assignment is (peer, start, end, extra)."""
    per_block = [{} for _ in range(n)]
    for peer, start, end, extra in assignments:
        for index in range(start, end):
            key, value = announce(peer, **extra)
            per_block[index][key] = value
    return per_block


@dataclass
class FakeAggregate:
    reliability: float
    observer_count: int
    distinct_clusters: int


# ---- collapsing --------------------------------------------------------------


def test_one_row_per_peer_not_per_block():
    # Petals writes a DHT record per block, so a peer hosting 12 blocks appears 12 times.
    rows = collapse_spans(blocks(12, ("peer-a", 0, 12, {})))
    assert len(rows) == 1
    assert (rows[0].start, rows[0].end, rows[0].blocks) == (0, 12, 12)


def test_two_peers_on_different_ranges():
    rows = collapse_spans(blocks(12, ("peer-a", 0, 6, {}), ("peer-b", 6, 12, {})))
    assert [(r.peer_id, r.start, r.end) for r in rows] == [
        ("peer-a", 0, 6),
        ("peer-b", 6, 12),
    ]


def test_compute_profile_is_surfaced():
    # Differing quant/dtype/attn is the most common reason two honest servers disagree
    # numerically. Hiding it in a monitor invites reading heterogeneity as fraud.
    rows = collapse_spans(
        blocks(2, ("peer-a", 0, 2, {"quant_type": "nf4", "torch_dtype": "float16",
                                    "attn_impl": "sdpa"}))
    )
    assert rows[0].profile == "nf4/fp16/sdpa"


def test_a_relayed_server_says_so():
    rows = collapse_spans(blocks(2, ("peer-a", 0, 2, {"quant_type": "none",
                                                      "using_relay": True})))
    assert "relay" in rows[0].profile


def test_a_named_volunteer_shows_their_name():
    rows = collapse_spans(blocks(2, ("peer-abcdefgh", 0, 2, {"public_name": "hewitt"})))
    assert rows[0].label == "hewitt"


def test_an_unnamed_volunteer_falls_back_to_a_short_id():
    rows = collapse_spans(blocks(2, ("QmLongPeerIdabcdefgh", 0, 2, {})))
    assert rows[0].label == "...abcdefgh"


# ---- coverage ----------------------------------------------------------------


def test_coverage_counts_hosts_per_block():
    rows = collapse_spans(blocks(4, ("peer-a", 0, 4, {}), ("peer-b", 2, 4, {})))
    assert count_replicas(rows, 4) == [1, 1, 2, 2]


def test_a_joining_server_does_not_count_as_coverage():
    rows = collapse_spans(blocks(4, ("peer-a", 0, 4, {"state": "JOINING"})))
    assert count_replicas(rows, 4) == [0, 0, 0, 0]


def test_many_servers_on_the_same_blocks_do_not_make_a_model_usable():
    # The point of leading with coverage: twelve servers all on blocks 0-3 serve nothing.
    per_block = blocks(12, *[(f"peer-{i}", 0, 4, {}) for i in range(12)])
    report = build_report("m", 12, per_block)
    assert len(report.servers) == 12
    assert report.covered is False
    assert report.missing_blocks == [4, 5, 6, 7, 8, 9, 10, 11]


def test_full_coverage_reports_the_weakest_block():
    per_block = blocks(4, ("peer-a", 0, 4, {}), ("peer-b", 0, 2, {}))
    report = build_report("m", 4, per_block)
    assert report.covered is True
    assert report.min_replicas == 1  # not 2: the weakest block is what limits the swarm


def test_an_empty_swarm_is_not_covered():
    report = build_report("m", 4, [{} for _ in range(4)])
    assert report.covered is False
    assert report.servers == []


def test_coverage_bar_buckets_by_worst_not_average():
    # A gap next to a well-covered block must not average away into looking healthy.
    bar = coverage_bar([3, 3, 3, 0] * 20, width=4)
    assert set(bar) == {"_"}


def test_coverage_bar_is_one_cell_per_block_when_it_fits():
    assert coverage_bar([0, 1, 2, 3, 5]) == "_123*"


# ---- reputation is kept separate from self-reports ---------------------------


def test_observed_reliability_is_attached_to_the_right_peer():
    per_block = blocks(4, ("peer-a", 0, 2, {}), ("peer-b", 2, 4, {}))
    report = build_report(
        "m", 4, per_block,
        aggregates={"peer-a": FakeAggregate(0.97, 3, 2)},
        observer_count=3,
        records_accepted=3,
    )
    by_id = {r.peer_id: r for r in report.servers}
    assert by_id["peer-a"].reliability == 0.97
    assert by_id["peer-a"].observers == 3
    assert by_id["peer-b"].reliability is None


def test_an_untested_server_says_so_rather_than_looking_good():
    per_block = blocks(2, ("peer-a", 0, 2, {"throughput": 9999.0}))
    text = "\n".join(render_text(build_report("m", 2, per_block)))
    # A high self-reported throughput with nobody having tested it must not read as quality.
    assert "not yet measured" in text
    assert "unverified" in text


def test_the_two_sources_are_never_blended_into_one_number():
    per_block = blocks(2, ("peer-a", 0, 2, {"throughput": 500.0}))
    report = build_report("m", 2, per_block, aggregates={"peer-a": FakeAggregate(0.2, 4, 3)})
    row = report.servers[0]
    assert row.throughput == 500.0
    assert row.reliability == 0.2


# ---- rendering ---------------------------------------------------------------


def test_text_report_names_the_missing_blocks():
    per_block = blocks(8, ("peer-a", 0, 3, {}))
    text = "\n".join(render_text(build_report("m", 8, per_block)))
    assert "NOT USABLE" in text
    assert "3, 4, 5, 6, 7" in text


def test_text_report_is_helpful_when_nobody_is_hosting():
    text = "\n".join(render_text(build_report("m", 4, [{} for _ in range(4)])))
    assert "nobody is hosting" in text
    assert "take a minute to" in text  # the warm-up window, which looks like failure


def test_html_is_self_contained():
    per_block = blocks(4, ("peer-a", 0, 4, {"public_name": "hewitt"}))
    page = render_html(build_report("m", 4, per_block))
    # No external fetches: a monitor page must render from disk or a bare static host.
    for external in ("http://", "https://", "<script", "@import"):
        assert external not in page


def test_html_escapes_a_hostile_public_name():
    # public_name is chosen by the volunteer and travels through the DHT unchecked.
    per_block = blocks(2, ("peer-a", 0, 2, {"public_name": "<img src=x onerror=alert(1)>"}))
    page = render_html(build_report("m", 2, per_block))
    assert "<img src=x" not in page
    assert "&lt;img" in page


def test_json_carries_the_derived_verdicts():
    import json

    per_block = blocks(4, ("peer-a", 0, 2, {}))
    payload = json.loads(render_json(build_report("m", 4, per_block)))
    assert payload["covered"] is False
    assert payload["missing_blocks"] == [2, 3]
    assert payload["servers"][0]["peer_id"] == "peer-a"


@pytest.mark.parametrize("state", ["ONLINE", "JOINING", "OFFLINE"])
def test_rendering_never_raises_on_sparse_announcements(state):
    row = ServerRow(peer_id="p", start=0, end=1, state=state)
    report = build_report("m", 1, [{"p": {"state": state}}])
    assert render_text(report)
    assert render_html(report)
    assert render_json(report)
    assert row.profile


def test_a_truncated_field_is_marked_not_silently_cut():
    from seedmesh.cli.monitor import fit

    assert fit("short", 10) == "short     "
    assert fit("a-very-long-volunteer-name", 10) == "a-very-lo~"


def test_the_relay_marker_survives_the_narrow_profile_column():
    # It did not: `nf4/bfloat16/eager (relayed)` was cut to `nf4/bfloat16/eager (`, dropping
    # the single most operationally important part of the field.
    from seedmesh.cli.monitor import fit

    rows = collapse_spans(
        blocks(2, ("peer-a", 0, 2, {"quant_type": "nf4", "torch_dtype": "bfloat16",
                                    "attn_impl": "eager", "using_relay": True}))
    )
    assert rows[0].profile == "nf4/bf16/eager+relay"
    assert "~" not in fit(rows[0].profile, 21)
