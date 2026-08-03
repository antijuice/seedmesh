"""The local web UI.

The backend is injected, so all of this runs with no Petals, no swarm and no network. What is
tested is the part that has its own failure modes: routing, the single-request lock, the token
gate, and whether the status payload tells the truth about verification.
"""

from __future__ import annotations

import threading

import pytest

from seedmesh.cli.gateway import GatewayState, build_status, route


def state(reply="hello", **kwargs):
    return GatewayState(lambda prompt, n: reply, **kwargs)


# ---- chat --------------------------------------------------------------------


def test_a_prompt_returns_the_reply_and_a_duration():
    result = state("the answer").chat("question")
    assert result["ok"] is True
    assert result["reply"] == "the answer"
    assert result["elapsed_ms"] >= 0


def test_an_empty_prompt_is_refused_without_touching_the_swarm():
    calls = []
    gateway = GatewayState(lambda p, n: calls.append(p) or "x")
    assert gateway.chat("   ")["ok"] is False
    assert calls == []


def test_a_backend_failure_is_reported_not_raised():
    def explode(prompt, n):
        raise RuntimeError("routing: not found")

    result = GatewayState(explode).chat("hi")
    assert result["ok"] is False
    assert result["error"] == "RuntimeError"
    assert "routing" in result["detail"]


def test_a_huge_error_is_truncated():
    def explode(prompt, n):
        raise RuntimeError("x" * 5000)

    assert len(GatewayState(explode).chat("hi")["detail"]) <= 400


# ---- one request at a time ---------------------------------------------------
#
# Petals inference sessions are stateful and not thread-safe, and the reputation observer
# wraps a shared class. Two concurrent generations would corrupt both.


def test_a_second_request_is_refused_while_one_is_running():
    started, release = threading.Event(), threading.Event()

    def slow(prompt, n):
        started.set()
        release.wait(5)
        return "done"

    gateway = GatewayState(slow)
    worker = threading.Thread(target=gateway.chat, args=("first",))
    worker.start()
    assert started.wait(5)

    # Refused, not queued: a queued click looks like a hung UI with no way to tell which
    # request is running.
    refused = gateway.chat("second")
    assert refused["ok"] is False
    assert refused["error"] == "busy"

    release.set()
    worker.join(5)


def test_the_lock_is_released_after_a_failure():
    def explode(prompt, n):
        raise RuntimeError("boom")

    gateway = GatewayState(explode)
    gateway.chat("first")
    # A lock leaked on the error path would wedge the gateway permanently.
    assert gateway.chat("second")["error"] == "RuntimeError"


# ---- routing -----------------------------------------------------------------


def test_status_route():
    code, payload = route(state(model="m"), "GET", "/api/status", None)
    assert code == 200 and payload["model"] == "m"


def test_busy_returns_409_so_the_page_can_tell_the_difference():
    gateway = state()
    gateway._lock.acquire()
    try:
        code, payload = route(gateway, "POST", "/api/chat", {"prompt": "x"})
        assert code == 409 and payload["error"] == "busy"
    finally:
        gateway._lock.release()


def test_a_non_object_body_is_rejected():
    code, _ = route(state(), "POST", "/api/chat", ["not", "an", "object"])
    assert code == 400


def test_unknown_paths_404():
    assert route(state(), "GET", "/secrets", None)[0] == 404
    assert route(state(), "POST", "/api/chat/../etc", {})[0] == 404


# ---- status tells the truth about verification -------------------------------


class FakeRow:
    label, short_id, start, end = "droplet-1", "...abcdefgh", 0, 12
    profile, throughput, reliability, observers = "none/fp32/eager", 173.0, 0.99, 3


class FakeReport:
    covered, n_blocks, min_replicas, missing_blocks = True, 12, 2, []
    servers = [FakeRow()]


class FakeVerifier:
    checked, skipped_no_verifier, skipped_unlocated, outcomes = 0, 4, 4, []


class FakeObserver:
    class stats:
        requests, ok, timeout, error = 20, 20, 0, 0

    subjects = {"a", "b"}


class FakeReputation:
    observer = FakeObserver()
    verifier = FakeVerifier()
    gate = None


def test_status_reports_swarm_and_measurements():
    payload = build_status(FakeReputation(), lambda: FakeReport())()
    assert payload["swarm"]["covered"] is True
    assert payload["measured"]["requests"] == 20
    assert payload["swarm"]["servers"][0]["label"] == "droplet-1"


def test_skipped_verification_is_surfaced_not_hidden():
    # The property that matters most in this whole file: a swarm that COULD NOT verify must
    # never be presentable as one that verified and found nothing wrong.
    payload = build_status(FakeReputation(), lambda: FakeReport())()
    assert payload["verification"]["checked"] == 0
    assert payload["verification"]["skipped"] == 4
    # Kept apart because the remedies differ: another volunteer vs a reachable address.
    assert payload["verification"]["skipped_unlocated"] == 4


def test_a_broken_swarm_read_does_not_break_the_status():
    def explode():
        raise RuntimeError("dht down")

    payload = build_status(FakeReputation(), explode)()
    assert "swarm_error" in payload
    assert payload["measured"]["requests"] == 20, "measurements should survive"


def test_status_works_with_no_reputation_at_all():
    payload = build_status(None, lambda: FakeReport())()
    assert payload["swarm"]["covered"] is True
    assert "measured" not in payload


def test_a_status_exception_does_not_take_down_chat():
    def explode():
        raise RuntimeError("status is broken")

    gateway = GatewayState(lambda p, n: "fine", status=explode)
    assert "status_error" in gateway.status()
    assert gateway.chat("hi")["ok"] is True


# ---- the token gate ----------------------------------------------------------


def test_the_page_and_api_require_the_token():
    from seedmesh.cli.gateway import make_handler

    handler_class = make_handler(state(), "s3cret", "<html>")

    def check(path, header=None):
        """Exercises _authorised without standing up a socket."""
        fake = type("Fake", (), {
            "path": path,
            "headers": type("H", (), {
                "get": lambda self, key: {"X-Seedmesh-Token": header}.get(key)
            })(),
        })()
        return handler_class._authorised(fake)

    assert check("/", "s3cret") is True
    assert check("/?token=s3cret") is True
    assert check("/") is False
    assert check("/?token=wrong") is False
    # Any page in the browser can POST to a localhost port; without a token an unrelated
    # site could drive this one.
    assert check("/api/chat", "guessed") is False


def test_tokens_are_not_predictable():
    from seedmesh.cli.gateway import TOKEN_BYTES
    import secrets

    seen = {secrets.token_urlsafe(TOKEN_BYTES) for _ in range(200)}
    assert len(seen) == 200
    assert all(len(t) >= 30 for t in seen)


# ---- the page ----------------------------------------------------------------


def test_the_page_is_self_contained():
    from seedmesh.cli.gateway_page import PAGE

    # A machine deep behind a NAT may have no usable internet at all, and an external fetch
    # would also leak that this gateway exists.
    for external in ("http://", "https://", "//cdn", "@import"):
        assert external not in PAGE


def test_the_page_escapes_model_output():
    from seedmesh.cli.gateway_page import PAGE

    # Replies come from a swarm of strangers' machines and land in innerHTML.
    assert "const esc =" in PAGE
    assert "esc(entry.reply)" in PAGE
    assert "esc(s.label)" in PAGE


@pytest.mark.parametrize("field", ["skipped", "excluded", "untested"])
def test_the_page_shows_the_uncomfortable_states(field):
    from seedmesh.cli.gateway_page import PAGE

    assert field in PAGE
