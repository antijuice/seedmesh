"""Tests for persistent node state and the assembled client-side reputation stack.

No hivemind, no Petals: the gossip transport is two dicts and the inference session is a
stand-in class, which is what makes these run on the platform most of this repo's users
develop on.
"""

from __future__ import annotations

import pytest

from seedmesh.cli.state import (
    identity_path,
    load_or_create_identity,
    reputation_path,
    seedmesh_home,
)
from seedmesh.core.identity import Identity
from seedmesh.reputation.diversity import UNKNOWN_CLUSTER
from seedmesh.reputation.session import ClientReputation

MODEL = "test/model"


@pytest.fixture(autouse=True)
def unpatched_session():
    """Guarantee the stand-in class is clean between tests.

    `attach` mutates a class, so a test that fails between attach and detach would leave
    every later test wired to a dead observer -- a failure mode that reads as six unrelated
    bugs.
    """
    original = FakeSession.step
    yield
    FakeSession.step = original


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SEEDMESH_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeSpan:
    def __init__(self, peer_id, start, end):
        self.peer_id, self.start, self.end = peer_id, start, end


class FakeInputs:
    def __init__(self, tokens=1):
        self.shape = (1, tokens, 8)


class FakeSession:
    def __init__(self, peer_id, start=0, end=4, raises=None):
        self.span = FakeSpan(peer_id, start, end)
        self.raises = raises

    def step(self, inputs, prompts=None, hypo_ids=None):
        if self.raises is not None:
            raise self.raises
        return inputs


class FakeDHT:
    """A shared key/value space, so two ClientReputations can actually reach each other."""

    def __init__(self, store=None):
        self.store_data = {} if store is None else store

    def publish(self, key, value, expiration_time, subkey=None):
        if subkey is not None:
            self.store_data.setdefault(key, {})[subkey] = value
        else:
            self.store_data[key] = value
        return True

    def fetch(self, keys):
        return {key: self.store_data.get(key) for key in keys}


def build(dht, transport_id, *, clock=None, state_path=None, identity=None, **kwargs):
    reputation = ClientReputation(
        identity=identity or Identity.generate(),
        model_id=MODEL,
        state_path=state_path,
        publish_fn=dht.publish,
        fetch_fn=dht.fetch,
        transport_id=transport_id,
        **kwargs,
    )
    if clock is not None:
        reputation.clock = clock
        reputation.scorer.clock = clock
        reputation.observer.clock = clock
        reputation.gossip.clock = clock
    return reputation


# ---- node state --------------------------------------------------------------


def test_home_honours_the_environment_override(home):
    assert seedmesh_home() == home
    assert identity_path() == home / "identity.key"


def test_identity_is_created_once_and_then_reused(home):
    first, created = load_or_create_identity()
    assert created is True
    second, created_again = load_or_create_identity()
    assert created_again is False
    # The whole point: a client that regenerates its key every run looks like a fresh sybil
    # to every other peer, and loses all its accumulated standing.
    assert first.peer_id == second.peer_id


def test_a_corrupt_key_file_is_never_overwritten(home):
    load_or_create_identity()
    path = identity_path()
    original = path.read_bytes()
    path.write_bytes(b"not a key")

    identity, created = load_or_create_identity()
    assert created is False
    assert isinstance(identity.peer_id, str)
    # Silently regenerating would destroy this node's standing in every swarm at once, so
    # the damaged file is left exactly as found for a human to look at.
    assert path.read_bytes() == b"not a key"
    assert original != b"not a key"


def test_reputation_is_stored_per_swarm(home):
    assert reputation_path("alpha") != reputation_path("beta")
    assert reputation_path("alpha").parent == home / "reputation"


def test_a_hostile_swarm_name_cannot_escape_the_directory(home):
    # Swarm names come from a JSON file someone else may have written.
    path = reputation_path("../../etc/passwd")
    assert path.parent == home / "reputation"
    assert ".." not in path.name


# ---- measurement feeds the scorer --------------------------------------------


def test_requests_are_measured_and_persisted_across_sessions(home, tmp_path):
    state = tmp_path / "rep.json"
    identity = Identity.generate()

    first = build(FakeDHT(), "transport-1", state_path=state, identity=identity)
    first.attach(FakeSession)
    for _ in range(3):
        FakeSession("peer-a").step(FakeInputs())
    first.close()

    second = build(FakeDHT(), "transport-1", state_path=state, identity=identity)
    assert second.loaded == 1
    # Not exactly 3: reloading applies the decay for the time that elapsed while the client
    # was away, which is the behaviour persistence is supposed to have.
    assert second.scorer.breakdown("peer-a").observations == pytest.approx(3.0, rel=1e-3)


def test_no_state_path_means_nothing_is_written(tmp_path):
    reputation = build(FakeDHT(), "transport-1", state_path=None)
    reputation.attach(FakeSession)
    FakeSession("peer-a").step(FakeInputs())
    assert reputation.save() is False
    assert list(tmp_path.iterdir()) == []


def test_detach_stops_measurement(tmp_path):
    reputation = build(FakeDHT(), "transport-1")
    reputation.attach(FakeSession)
    FakeSession("peer-a").step(FakeInputs())
    reputation.detach()
    FakeSession("peer-a").step(FakeInputs())
    assert reputation.observer.stats.requests == 1


# ---- gossip actually crosses between two clients -----------------------------


def test_one_clients_measurements_reach_another(tmp_path):
    shared = FakeDHT()

    alice = build(shared, "transport-alice")
    alice.attach(FakeSession)
    for _ in range(5):
        FakeSession("peer-server").step(FakeInputs())
    alice.detach()
    assert alice.gossip.publish() is not None

    # Bob has measured nothing himself. Everything he knows comes off the wire.
    bob = build(shared, "transport-bob")
    assert bob.gossip.sync()[1] == 1
    assert "peer-server" in bob.aggregator.subjects()
    assert bob.aggregator.aggregate("peer-server").reliability > 0.5


def test_an_unlocatable_observer_lands_in_the_shared_bucket(tmp_path):
    shared = FakeDHT()
    alice = build(shared, "transport-alice")
    alice.attach(FakeSession)
    FakeSession("peer-server").step(FakeInputs())
    alice.detach()
    alice.gossip.publish()

    bob = build(shared, "transport-bob")
    bob.gossip.sync()
    # A client cannot resolve another client's network location, and a wrong label would be
    # worse than none: it would let colluding peers read as independent. So they share one
    # low-budget bucket rather than each getting a cluster of their own.
    assert bob._cluster_of(alice.identity.peer_id) == UNKNOWN_CLUSTER
    assert bob.aggregator.aggregate("peer-server").distinct_clusters == 1


def test_sybils_cannot_outweigh_by_being_unlocatable(tmp_path):
    shared = FakeDHT()
    for index in range(10):
        sybil = build(shared, f"transport-sybil-{index}")
        sybil.attach(FakeSession)
        failing = FakeSession("peer-target", 0, 4, raises=ValueError("nope"))
        for _ in range(5):
            with pytest.raises(ValueError):
                failing.step(FakeInputs())
        sybil.detach()
        sybil.gossip.publish()

    victim = build(shared, "transport-victim")
    victim.gossip.sync()
    score = victim.aggregator.aggregate("peer-target")
    # Ten identities, all unplaceable, share one bucket whose budget is deliberately below a
    # single normal cluster's -- so the swarm should not treat this as strong evidence.
    assert score.distinct_clusters == 1
    assert score.total_weight <= 0.5
    assert not score.is_trustworthy_signal


# ---- sync cadence ------------------------------------------------------------


def test_sync_waits_for_the_interval(tmp_path):
    clock = FakeClock()
    reputation = build(FakeDHT(), "transport-1", clock=clock, sync_interval_s=300.0)
    reputation.attach(FakeSession)
    FakeSession("peer-a").step(FakeInputs())

    reputation._last_sync = clock.now()
    assert reputation.maybe_sync() is None
    clock.advance(299)
    assert reputation.maybe_sync() is None
    clock.advance(2)
    assert reputation.maybe_sync() is not None


def test_forced_sync_ignores_the_interval(tmp_path):
    clock = FakeClock()
    reputation = build(FakeDHT(), "transport-1", clock=clock, sync_interval_s=300.0)
    reputation.attach(FakeSession)
    FakeSession("peer-a").step(FakeInputs())
    reputation._last_sync = clock.now()
    assert reputation.maybe_sync(force=True) is not None


def test_a_broken_dht_does_not_break_the_session(tmp_path):
    class BrokenDHT(FakeDHT):
        def publish(self, *a, **k):
            raise RuntimeError("network gone")

        def fetch(self, keys):
            raise RuntimeError("network gone")

    state = tmp_path / "rep.json"
    reputation = build(BrokenDHT(), "transport-1", state_path=state)
    reputation.attach(FakeSession)
    FakeSession("peer-a").step(FakeInputs())
    saved, accepted = reputation.close()
    # Gossip failed; the local evidence still survives, which is the part that must not be
    # lost when a chat ends on a flaky link.
    assert saved is True
    assert accepted == 0
    assert state.exists()


def test_describe_reports_both_halves(tmp_path):
    shared = FakeDHT()
    alice = build(shared, "transport-alice")
    alice.attach(FakeSession)
    FakeSession("peer-server").step(FakeInputs())
    alice.detach()
    alice.gossip.publish()

    bob = build(shared, "transport-bob")
    bob.attach(FakeSession)
    FakeSession("peer-server").step(FakeInputs())
    bob.detach()
    bob.gossip.sync()

    text = "\n".join(bob.describe())
    assert "measured 1 request(s)" in text
    assert "other observers reported on 1 server(s)" in text
