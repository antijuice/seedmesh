"""Clearing server processes left behind by a session that is no longer running.

Stopping `seedmesh serve` does not always stop the server: Petals runs a process tree, and a
closed terminal or `kill -9` leaves children holding GPU memory and their DHT identity.
Starting again then fails or half-works, and the remedy volunteers learn is
`pkill -f run_server` -- a blunt instrument that also kills servers they meant to keep.

Most of these test what must NOT be killed.
"""

from __future__ import annotations

from seedmesh.cli.stale import (
    ServerProcess,
    describe,
    list_petals_servers,
    others_for,
    stale_for,
    terminate,
)

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def server(pid, ppid, model=MODEL, extra=""):
    return ServerProcess(
        pid=pid, ppid=ppid,
        cmdline=f"python -m petals.cli.run_server {model} --num_blocks 12 {extra}".strip(),
    )


# ---- what must NOT be killed -------------------------------------------------


def test_a_deliberately_running_server_is_left_alone():
    """Two servers on one model is legitimate and this project does it -- one machine hosted
    blocks 9-32 while another hosted 0-9. A live parent means somebody is running it."""
    alive = server(pid=100, ppid=99)
    assert stale_for(MODEL, [alive]) == []
    assert others_for(MODEL, [alive]) == [alive]


def test_a_server_for_a_different_model_is_never_touched():
    """A different model is a different swarm and none of our business."""
    other = server(pid=100, ppid=1, model="JackFram/llama-160m")
    assert stale_for(MODEL, [other]) == []
    assert others_for(MODEL, [other]) == []


def test_a_non_petals_process_is_ignored(tmp_path):
    def reader(pid):
        return ServerProcess(pid=pid, ppid=1, cmdline="python -m http.server")

    assert list_petals_servers(reader=reader, pids=[123]) == []


# ---- what must be killed -----------------------------------------------------


def test_an_orphaned_server_for_this_model_is_stale():
    orphan = server(pid=100, ppid=1)
    assert stale_for(MODEL, [orphan]) == [orphan]
    assert others_for(MODEL, [orphan]) == []


def test_ppid_zero_also_counts_as_orphaned():
    assert stale_for(MODEL, [server(pid=100, ppid=0)]) != []


def test_a_mixed_set_is_split_correctly():
    orphan, alive = server(pid=100, ppid=1), server(pid=200, ppid=199)
    processes = [orphan, alive]
    assert stale_for(MODEL, processes) == [orphan]
    assert others_for(MODEL, processes) == [alive]


# ---- how they are killed -----------------------------------------------------


def test_sigterm_first_so_the_dht_announcement_is_withdrawn():
    """A Petals server that gets SIGTERM deregisters from the DHT. Straight to SIGKILL
    leaves stale announcements, which make a model look covered when it is not."""
    sent = []
    terminate([server(100, 1)], kill=lambda pid, sig: sent.append((pid, sig)),
              wait=lambda s: None, still_alive=lambda pid: False)
    assert len(sent) == 1
    assert sent[0][0] == 100
    import signal as signal_module

    assert sent[0][1] == signal_module.SIGTERM


def test_sigkill_only_for_what_ignores_sigterm():
    import signal as signal_module

    # SIGKILL does not exist on Windows; the code falls back to SIGTERM there rather than
    # raising AttributeError, so the test asks for the same thing the code does.
    hard = getattr(signal_module, "SIGKILL", signal_module.SIGTERM)
    sent = []
    terminate([server(100, 1)], kill=lambda pid, sig: sent.append((pid, sig)),
              wait=lambda s: None, still_alive=lambda pid: True)
    assert [s for _, s in sent] == [signal_module.SIGTERM, hard]
    assert len(sent) == 2, "escalate exactly once, not repeatedly"


def test_a_process_that_vanishes_mid_kill_is_not_an_error():
    def kill(pid, sig):
        raise ProcessLookupError("already gone")

    assert terminate([server(100, 1)], kill=kill, wait=lambda s: None) == []


def test_nothing_to_kill_does_not_sleep():
    slept = []
    terminate([], kill=lambda p, s: None, wait=slept.append)
    assert slept == [], "no reason to pause when there is nothing to wait for"


# ---- what it says ------------------------------------------------------------


def test_it_says_what_it_cleared_and_why():
    text = "\n".join(describe([server(100, 1)], []))
    assert "100" in text
    assert "orphaned" in text, "say why these and not others"


def test_it_reports_servers_it_deliberately_left_running():
    text = "\n".join(describe([], [server(200, 199)]))
    assert "live parent" in text
    assert "leaving them alone" in text


def test_it_is_silent_when_there_is_nothing_to_report():
    assert describe([], []) == []


# ---- reading /proc -----------------------------------------------------------


def test_the_current_process_is_never_listed():
    import os

    def reader(pid):
        return ServerProcess(pid=pid, ppid=1, cmdline="python -m petals.cli.run_server x")

    assert list_petals_servers(reader=reader, pids=[os.getpid()]) == []


def test_an_unreadable_process_is_skipped():
    assert list_petals_servers(reader=lambda pid: None, pids=[1, 2, 3]) == []


def test_block_indices_does_not_need_a_network_fetch(monkeypatch):
    """`--block-indices 0:9` fixes the count, so auto-sizing is redundant -- and it fetched
    a config over the network and REFUSED TO START when that failed, turning an explicit
    instruction into an avoidable failure. Found when the suite lost network access."""
    import seedmesh.cli.main as main_mod

    captured = {}
    monkeypatch.setattr(main_mod, "_refuse_on_windows", lambda what: False)
    monkeypatch.setattr(main_mod, "_backend_missing", lambda what: False)

    def must_not_fetch(model):
        raise AssertionError("auto-sizing should be skipped entirely")

    monkeypatch.setattr("seedmesh.cli.hardware.fetch_config", must_not_fetch)
    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: captured.update(cmd=cmd) or 0)

    assert main_mod.main(
        ["serve", "--no-visibility-check", "--block-indices", "0:9"]
    ) == 0
    assert captured["cmd"][captured["cmd"].index("--block_indices") + 1] == "0:9"
