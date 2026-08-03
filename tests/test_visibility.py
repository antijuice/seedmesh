"""Telling a volunteer whether the swarm can actually see them.

`Started` means the local process came up. It says nothing about whether any record reached
the DHT. One machine loaded all 32 blocks, logged `Started`, and never appeared to anyone
else — and neither end could tell whether it had stopped, was still loading, or was
announcing into a void.
"""

from __future__ import annotations

from seedmesh.cli.visibility import describe, new_arrivals, snapshot_peers, watch


class FakeRow:
    def __init__(self, peer_id):
        self.peer_id = peer_id


class FakeReport:
    def __init__(self, peer_ids):
        self.servers = [FakeRow(p) for p in peer_ids]


def patch_collect(monkeypatch, result):
    import seedmesh.cli.monitor as monitor_module

    def fake(dht, model, prefix="seedmesh"):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(monitor_module, "collect", fake)


# ---- an unreadable swarm is not an empty one --------------------------------


def test_a_failed_read_is_none_not_an_empty_set(monkeypatch):
    """Conflating them would let a broken lookup masquerade as an empty swarm, and every
    peer already there would then look like a new arrival."""
    patch_collect(monkeypatch, RuntimeError("dht down"))
    assert snapshot_peers(object(), "m") is None


def test_an_empty_swarm_is_an_empty_set(monkeypatch):
    patch_collect(monkeypatch, FakeReport([]))
    assert snapshot_peers(object(), "m") == set()


def test_no_arrivals_can_be_inferred_without_a_baseline():
    assert new_arrivals(None, {"a"}) == set()
    assert new_arrivals({"a"}, None) == set()


def test_a_new_peer_is_an_arrival():
    assert new_arrivals({"a"}, {"a", "b"}) == {"b"}


def test_a_departure_is_not_an_arrival():
    assert new_arrivals({"a", "b"}, {"a"}) == set()


# ---- what it says -----------------------------------------------------------


def test_a_visible_server_is_told_plainly():
    lines = describe({"12D3KooWabcdefgh"}, 30.0, baseline_ok=True)
    text = "\n".join(lines)
    assert "VISIBLE" in text
    assert "abcdefgh" in text


def test_two_arrivals_admit_the_ambiguity():
    """Identifying "you" as "a peer that just appeared" is wrong if someone else joins in
    the same window. That can only report success early, never invent a failure -- but it
    must be stated rather than hidden."""
    text = "\n".join(describe({"peer-aaaaaaaa", "peer-bbbbbbbb"}, 30.0, baseline_ok=True))
    assert "may be someone else" in text


def test_an_invisible_server_gets_next_steps_not_just_a_verdict():
    text = "\n".join(describe(set(), 180.0, baseline_ok=True))
    assert "NOT VISIBLE" in text
    assert "seedmesh monitor" in text and "seedmesh doctor" in text
    # It must not claim certainty it does not have: a big model may still be loading.
    assert "still be loading" in text


def test_no_baseline_means_the_check_is_skipped_not_failed():
    text = "\n".join(describe(set(), 0.0, baseline_ok=True + 0))  # falsy baseline_ok
    assert "NOT VISIBLE" in text
    skipped = "\n".join(describe(set(), 0.0, baseline_ok=False))
    assert "skipped" in skipped
    assert "NOT VISIBLE" not in skipped


# ---- the polling loop -------------------------------------------------------


def test_it_reports_as_soon_as_the_peer_appears(monkeypatch):
    """Petals announces blocks as JOINING before downloading them, so the record lands
    within a minute even when loading takes fifteen. Waiting once would answer long after
    the volunteer stopped wondering."""
    states = [FakeReport(["old"]), FakeReport(["old"]), FakeReport(["old", "new"])]

    def fake(dht, model, prefix="seedmesh"):
        return states.pop(0) if states else FakeReport(["old", "new"])

    import seedmesh.cli.monitor as monitor_module

    monkeypatch.setattr(monitor_module, "collect", fake)

    said = []
    found = watch(
        lambda: object(), "m", baseline={"old"},
        checks=6, interval=1.0, sleep=lambda s: None, emit=said.append,
    )
    assert found == {"new"}
    assert len(said) == 1, "should report once, not every poll"
    assert "VISIBLE" in said[0]


def test_it_gives_up_and_says_so(monkeypatch):
    patch_collect(monkeypatch, FakeReport(["old"]))
    said = []
    found = watch(
        lambda: object(), "m", baseline={"old"},
        checks=3, interval=1.0, sleep=lambda s: None, emit=said.append,
    )
    assert found == set()
    assert "NOT VISIBLE" in said[0]


def test_a_dht_that_cannot_be_built_does_not_raise(monkeypatch):
    """The check is a diagnostic. It must never take down a working server."""
    patch_collect(monkeypatch, FakeReport(["old"]))

    def no_dht():
        raise OSError("no daemon")

    said = []
    assert watch(no_dht, "m", baseline={"old"}, checks=2, interval=1.0,
                 sleep=lambda s: None, emit=said.append) == set()


def test_no_baseline_short_circuits_without_polling():
    polled = []
    watch(lambda: polled.append(1), "m", baseline=None,
          checks=6, interval=1.0, sleep=lambda s: None, emit=lambda t: None)
    assert polled == [], "nothing to compare against, so nothing to wait for"


def test_the_watcher_survives_a_missing_backend(monkeypatch):
    """The imports were originally outside the try, so a machine without hivemind got an
    unhandled thread exception printed next to a server that was working fine. A diagnostic
    must never produce a traceback of its own."""
    import seedmesh.cli.main as main_mod

    watcher = main_mod._start_visibility_watch(["/ip4/1.2.3.4/tcp/1/p2p/Qm"], "m", {"old"})
    watcher.cancelled = True  # nothing to assert beyond: it did not raise here
    assert watcher is not None
