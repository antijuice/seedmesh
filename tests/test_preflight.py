"""Checking coverage before committing to a long download.

`seedmesh gateway --model 8b` printed two lines and then sat silent for minutes -- fetching
~2 GiB of embeddings for a model the swarm could not serve, then waiting out a five-minute
routing warm-up to say so. Both halves were wrong: no feedback, and the wrong order.
"""

from __future__ import annotations

import pytest

from seedmesh.cli.preflight import (
    check_before_loading,
    client_download_bytes,
    coverage_problem,
    describe_download,
)


class FakeReport:
    def __init__(self, covered):
        self.covered = covered


@pytest.fixture
def patched(monkeypatch):
    """Swap out the monitor's DHT read; these tests are about the decision, not the lookup."""

    def install(report=None, raises=None):
        import seedmesh.cli.monitor as monitor_module

        def fake_collect(dht, model, prefix="seedmesh"):
            if raises is not None:
                raise raises
            return report

        monkeypatch.setattr(monitor_module, "collect", fake_collect)
        monkeypatch.setattr(monitor_module, "render_text", lambda r: ["NOT USABLE: 9 blocks"])

    return install


# ---- the decision -----------------------------------------------------------


def test_a_covered_swarm_proceeds_silently(patched):
    patched(FakeReport(covered=True))
    ok, lines = check_before_loading(object(), "m")
    assert ok is True
    assert lines == []


def test_an_uncovered_swarm_stops_before_the_download(patched):
    patched(FakeReport(covered=False))
    ok, lines = check_before_loading(object(), "m")
    assert ok is False
    text = "\n".join(lines)
    assert "NOT USABLE" in text
    # The reason for stopping has to be stated, or it looks like an unrelated failure.
    assert "could not be used once downloaded" in text
    assert "--skip-coverage-check" in text


def test_the_check_can_be_skipped(patched):
    # Someone who knows a server is seconds from ready should not be told no.
    patched(FakeReport(covered=False))
    ok, lines = check_before_loading(object(), "m", skip=True)
    assert ok is True
    assert lines == []


# ---- failing open -----------------------------------------------------------
#
# The check is advisory. Refusing to start because a DHT lookup timed out would be a worse
# failure than the one it prevents, and says nothing about whether the swarm works.


def test_a_failed_lookup_proceeds_rather_than_blocking(patched):
    patched(raises=RuntimeError("dht timeout"))
    ok, lines = check_before_loading(object(), "m")
    assert ok is True
    assert lines == []


def test_coverage_problem_returns_none_when_it_cannot_tell(patched):
    patched(raises=OSError("network down"))
    assert coverage_problem(object(), "m") is None


def test_a_covered_swarm_reports_no_problem(patched):
    patched(FakeReport(covered=True))
    assert coverage_problem(object(), "m") is None


# ---- the size estimate ------------------------------------------------------


def test_the_estimate_is_the_clients_share_not_the_whole_model(monkeypatch):
    import seedmesh.cli.hardware as hardware

    # Llama-3.1-8B: 128256 x 4096, twice (embeddings + lm_head), 2 bytes each = 1.96 GiB.
    monkeypatch.setattr(hardware, "fetch_config",
                        lambda m: {"vocab_size": 128256, "hidden_size": 4096})
    size = client_download_bytes("llama-8b")
    assert size == 2 * 128256 * 4096 * 2
    assert 1.9 < size / 2**30 < 2.0, "should be ~2 GiB, not the 15 GiB of block weights"


def test_the_message_says_block_weights_are_not_downloaded(monkeypatch):
    import seedmesh.cli.hardware as hardware

    monkeypatch.setattr(hardware, "fetch_config",
                        lambda m: {"vocab_size": 128256, "hidden_size": 4096})
    text = "\n".join(describe_download("llama-8b"))
    assert "1.9" in text or "2.0" in text
    # Otherwise "8B model" primes people to expect 15 GiB and kill it.
    assert "servers" in text and "one-time" in text


def test_a_small_model_gets_a_short_message(monkeypatch):
    import seedmesh.cli.hardware as hardware

    monkeypatch.setattr(hardware, "fetch_config",
                        lambda m: {"vocab_size": 32000, "hidden_size": 768})
    lines = describe_download("llama-160m")
    assert len(lines) == 1, "no need for three lines about 94 MiB"


def test_an_unreachable_config_still_says_something(monkeypatch):
    import seedmesh.cli.hardware as hardware

    def boom(model):
        raise OSError("huggingface unreachable")

    monkeypatch.setattr(hardware, "fetch_config", boom)
    assert client_download_bytes("m") is None
    # Silence is the bug being fixed, so an unknown size must not produce no output.
    assert describe_download("m")
