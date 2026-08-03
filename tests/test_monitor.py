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


# ---- reading a swarm without importing Petals --------------------------------
#
# The monitor needs two facts (block count, DHT prefix) plus the per-block records. Going
# through AutoDistributedConfig drags in petals -> transformers -> torch: several hundred MB
# resident to read a handful of DHT keys, on the $5/1 GiB box that is the natural home for a
# public dashboard. Both are reimplemented -- and a wrong prefix reads an EMPTY namespace and
# reports a healthy swarm as dead, so the rules are pinned here.


@pytest.mark.parametrize(
    "model, model_type, expected",
    [
        ("JackFram/llama-160m", "llama", "llama-160m-hf"),
        ("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama", "Meta-Llama-3-1-8B-Instruct-hf"),
        ("some/already-hf", "llama", "already-hf"),          # suffix is not doubled
        ("tiiuae/falcon-7b", "falcon", "falcon-7b"),         # falcon takes no -hf
        ("bigscience/bloom-560m", "bloom", "bigscience/bloom-560m-petals"),  # full path
        ("mistralai/Mixtral-8x7B-v0.1", "mixtral", "mistralai/Mixtral-8x7B-v0-1"),  # full path
        ("deepseek-ai/DeepSeek-V3", "deepseek_v3", "DeepSeek-V3-hf"),
    ],
)
def test_dht_prefix_matches_petals_rules(model, model_type, expected):
    from seedmesh.cli.monitor import dht_prefix_for

    assert dht_prefix_for(model, model_type) == expected


def test_dht_prefix_agrees_with_the_installed_petals():
    """The parametrised cases above are copied from Petals' source. This one checks the copy
    has not drifted from the Petals actually installed -- skipped where there is none."""
    petals_config = pytest.importorskip("petals.utils.auto_config")
    from seedmesh.cli.monitor import dht_prefix_for

    config = petals_config.AutoDistributedConfig.from_pretrained("JackFram/llama-160m")
    assert dht_prefix_for("JackFram/llama-160m", config.model_type) == config.dht_prefix


def test_an_announcement_decodes_without_petals():
    from seedmesh.cli.monitor import announcement_from_tuple

    # Exactly the shape ServerInfo.to_tuple produces.
    decoded = announcement_from_tuple((2, 173.5, {"public_name": "hewitt", "quant_type": "none"}))
    assert decoded["state"] == "ONLINE"
    assert decoded["throughput"] == 173.5
    assert decoded["public_name"] == "hewitt"


def test_unknown_extra_fields_pass_through_rather_than_breaking_the_read():
    from seedmesh.cli.monitor import announcement_from_tuple

    # Petals adds fields over time; a decoder that enumerated them would break on upgrade.
    decoded = announcement_from_tuple((2, 1.0, {"a_field_from_the_future": 42}))
    assert decoded["a_field_from_the_future"] == 42


def test_a_malformed_announcement_is_offline_not_an_exception():
    from seedmesh.cli.monitor import announcement_from_tuple

    # Any peer can write any bytes to a DHT key; the monitor must not be crashable by one.
    for junk in (None, "nonsense", (), (2,), {"not": "a tuple"}):
        assert announcement_from_tuple(junk)["state"] == "OFFLINE"


def test_states_map_to_the_names_coverage_counting_uses():
    from seedmesh.cli.monitor import announcement_from_tuple, count_replicas

    rows = collapse_spans([{"p": announcement_from_tuple((1, 1.0, {}))}])  # JOINING
    assert count_replicas(rows, 1) == [0]
    rows = collapse_spans([{"p": announcement_from_tuple((2, 1.0, {}))}])  # ONLINE
    assert count_replicas(rows, 1) == [1]

# ---- a report must not contradict itself ------------------------------------
#
# Bringing up the 8B model produced "1 server(s) donating 23 block-slot(s)" immediately above
# "NOT USABLE: 32 block(s) have no host". Both were true -- coverage counts only ONLINE peers
# and the single server was still JOINING -- but read together they say the tool is broken.
# Nothing was broken; the server was loading 15 GiB of weights.


def has(lines, phrase):
    return any(phrase in line for line in lines)


def test_a_joining_only_swarm_says_so_instead_of_looking_broken():
    per_block = blocks(32, ("peer-a", 0, 23, {"state": "JOINING"}))
    lines = render_text(build_report("m", 32, per_block))
    assert has(lines, "still JOINING")
    assert has(lines, "Nothing is wrong")
    # The generic line is advice for a different situation and must not appear here.
    assert not has(lines, "same blocks does not help")


def test_it_still_warns_when_the_joining_servers_will_not_be_enough():
    # 23 of 32 blocks: even once loaded the model stays unusable. Saying "just wait"
    # without saying that would be false reassurance.
    per_block = blocks(32, ("peer-a", 0, 23, {"state": "JOINING"}))
    assert has(render_text(build_report("m", 32, per_block)), "short of 32")


def test_no_false_alarm_when_the_joining_servers_do_cover_it():
    per_block = blocks(12, ("peer-a", 0, 12, {"state": "JOINING"}))
    lines = render_text(build_report("m", 12, per_block))
    assert has(lines, "still JOINING")
    assert not has(lines, "short of")


def test_a_genuinely_uncovered_online_swarm_keeps_the_original_advice():
    per_block = blocks(12, ("peer-a", 0, 6, {"state": "ONLINE"}))
    lines = render_text(build_report("m", 12, per_block))
    assert has(lines, "same blocks does not help")
    assert not has(lines, "still JOINING")
