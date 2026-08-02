"""Tests for turning a real inference client into a reputation observer.

These deliberately never import Petals. `attach()` takes a class so the whole mechanism can
be exercised with a stand-in that has the same shape -- which is also the only way these run
on Windows, where the backend does not exist.
"""

from __future__ import annotations

import asyncio

import pytest

from seedmesh.core.types import Outcome
from seedmesh.reputation.observer import (
    REFUSAL_MARKER,
    InferenceObserver,
    attach,
    classify,
)
from seedmesh.reputation.scorer import ReputationScorer

MODEL = "test/model"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


class FakeSpan:
    def __init__(self, peer_id: str, start: int, end: int) -> None:
        self.peer_id = peer_id
        self.start = start
        self.end = end


class FakeInputs:
    def __init__(self, tokens: int) -> None:
        self.shape = (1, tokens, 64)


class FakeSession:
    """Same shape as Petals' `_ServerInferenceSession`, minus everything irrelevant."""

    def __init__(self, peer_id: str, start: int, end: int, raises: BaseException | None = None):
        self.span = FakeSpan(peer_id, start, end)
        self.raises = raises
        self.calls = 0

    def step(self, inputs, prompts=None, hypo_ids=None):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return inputs


@pytest.fixture
def observer():
    clock = FakeClock()
    return InferenceObserver("observer-1", ReputationScorer(clock=clock), MODEL, clock=clock)


@pytest.fixture
def attached(observer):
    detach = attach(FakeSession, observer)
    yield observer
    detach()


# ---- classification ----------------------------------------------------------


def test_success_is_ok(attached):
    session = FakeSession("peer-a", 0, 4)
    session.step(FakeInputs(3))
    assert attached.stats.ok == 1
    assert attached.stats.tokens == 3


def test_timeout_is_churn_not_fault():
    assert classify(TimeoutError("slow")) is Outcome.TIMEOUT
    assert classify(asyncio.TimeoutError()) is Outcome.TIMEOUT


def test_timeout_wrapped_by_petals_still_reads_as_timeout():
    # Petals re-raises remote failures in its own exception type, so isinstance is not
    # enough -- the class name only survives inside the message.
    class P2PDaemonError(Exception):
        pass

    assert classify(P2PDaemonError("TimeoutError: no response")) is Outcome.TIMEOUT


def test_capacity_refusal_is_not_held_against_the_server(attached):
    # The whole point of REFUSED: a server that says "I am full" is behaving correctly, and
    # must not lose reliability for it.
    session = FakeSession("peer-full", 0, 4, raises=RuntimeError(f"{REFUSAL_MARKER}: no room"))
    with pytest.raises(RuntimeError):
        session.step(FakeInputs(1))
    assert attached.stats.refused == 1
    assert attached.scorer.breakdown("peer-full").reliability == pytest.approx(
        attached.scorer.breakdown("never-seen").reliability
    )


def test_plain_failure_is_an_error(attached):
    session = FakeSession("peer-bad", 0, 4, raises=ValueError("garbage"))
    with pytest.raises(ValueError):
        session.step(FakeInputs(1))
    assert attached.stats.error == 1
    # Stated as a comparison against an unmeasured peer rather than an absolute threshold:
    # one failure against a beta prior does not drive reliability near zero, and asserting a
    # bare number here would only encode the prior's current shape.
    assert (
        attached.scorer.breakdown("peer-bad").reliability
        < attached.scorer.breakdown("never-seen").reliability
    )


# ---- the wrapper must never change behaviour ---------------------------------


def test_original_return_value_is_passed_through(attached):
    session = FakeSession("peer-a", 0, 4)
    sentinel = FakeInputs(2)
    assert session.step(sentinel) is sentinel


def test_original_exception_is_reraised_unchanged(attached):
    boom = ValueError("exact instance")
    session = FakeSession("peer-a", 0, 4, raises=boom)
    with pytest.raises(ValueError) as caught:
        session.step(FakeInputs(1))
    assert caught.value is boom


def test_bookkeeping_failure_cannot_break_the_request(observer):
    def explode(*args, **kwargs):
        raise RuntimeError("scorer is broken")

    observer.note = explode  # type: ignore[assignment]
    detach = attach(FakeSession, observer)
    try:
        session = FakeSession("peer-a", 0, 4)
        assert session.step(FakeInputs(1)) is not None  # request still succeeds
    finally:
        detach()


def test_session_without_a_span_is_left_alone(attached):
    class Spanless(FakeSession):
        def __init__(self):
            super().__init__("x", 0, 1)
            self.span = None

    Spanless.step = FakeSession.step  # unwrapped: attach patched FakeSession, not this
    session = Spanless()
    FakeSession.step(session, FakeInputs(1))
    assert attached.stats.requests == 0


# ---- attachment mechanics ----------------------------------------------------


def test_detach_restores_the_original(observer):
    before = FakeSession.step
    detach = attach(FakeSession, observer)
    assert FakeSession.step is not before
    detach()
    assert FakeSession.step is before


def test_attaching_twice_does_not_double_count(observer):
    first = attach(FakeSession, observer)
    second = attach(FakeSession, observer)
    try:
        FakeSession("peer-a", 0, 4).step(FakeInputs(1))
        assert observer.stats.requests == 1
    finally:
        second()
        first()


# ---- observations reach the scorer -------------------------------------------


def test_each_server_is_scored_separately(attached):
    for _ in range(3):
        FakeSession("peer-good", 0, 4).step(FakeInputs(1))
    bad = FakeSession("peer-bad", 4, 8, raises=ValueError("nope"))
    for _ in range(3):
        with pytest.raises(ValueError):
            bad.step(FakeInputs(1))

    assert attached.subjects == {"peer-good", "peer-bad"}
    ranked = attached.scorer.ranked(["peer-good", "peer-bad"])
    assert ranked[0].peer_id == "peer-good"


def test_block_range_comes_from_the_span(attached):
    FakeSession("peer-a", 8, 12).step(FakeInputs(1))
    ranges = attached.scorer._peers["peer-a"].ranges
    assert any(r.start == 8 and r.end == 12 and r.model_id == MODEL for r in ranges)


def test_empty_span_is_dropped_rather_than_raising(attached):
    # BlockRange rightly refuses a zero-length range; that must not surface as a failed
    # inference request.
    FakeSession("peer-a", 4, 4).step(FakeInputs(1))
    assert attached.stats.requests == 0


def test_describe_is_quiet_before_anything_is_measured(observer):
    assert observer.describe() == ["  no servers measured this session"]


def test_describe_names_each_measured_server(attached):
    FakeSession("peer-abcdefgh", 0, 4).step(FakeInputs(1))
    text = "\n".join(attached.describe())
    assert "1 request(s) across 1 server(s)" in text
    assert "abcdefgh" in text
