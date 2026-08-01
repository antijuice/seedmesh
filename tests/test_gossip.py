"""Reputation gossip and persistence.

The gossip layer is where a client stops trusting only itself, which makes it the place a
malicious peer would attack. These tests exercise it through an in-memory DHT so the whole
publish/verify/aggregate cycle runs without hivemind, a swarm, or Linux.
"""

from __future__ import annotations

import pytest

from seedmesh.core.clock import ManualClock
from seedmesh.core.identity import Identity
from seedmesh.core.types import BlockRange, Observation, Outcome
from seedmesh.reputation.aggregate import AggregationConfig, ReputationAggregator
from seedmesh.reputation.gossip import ReputationGossip
from seedmesh.reputation.scorer import ReputationScorer, ScorerConfig

RANGE = BlockRange("m", 0, 4)


class FakeDHT:
    """Minimal stand-in: a dict with expirations, matching hivemind's TTL semantics."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.store: dict[str, tuple[dict, float]] = {}
        self.writes = 0

    def publish(self, key: str, value: dict, expiration_time: float, subkey=None) -> bool:
        if subkey is not None:
            existing = self.store.get(key)
            merged = dict(existing[0]) if existing and isinstance(existing[0], dict) else {}
            merged[subkey] = value
            self.store[key] = (merged, expiration_time)
        else:
            self.store[key] = (value, expiration_time)
        self.writes += 1
        return True

    def fetch(self, keys: list[str]) -> dict[str, dict | None]:
        now = self.clock.now()
        out = {}
        for key in keys:
            entry = self.store.get(key)
            out[key] = entry[0] if entry and entry[1] > now else None
        return out


def make_node(name: str, clock: ManualClock, dht: FakeDHT, *, seed: int, locator=None):
    identity = Identity.from_seed(bytes([seed]) * 32)
    scorer = ReputationScorer(ScorerConfig(), clock=clock)
    aggregator = ReputationAggregator(
        lambda peer: CLUSTERS.get(peer, "cluster:unknown"), AggregationConfig()
    )
    gossip = ReputationGossip(
        identity,
        scorer,
        aggregator,
        publish_fn=dht.publish,
        fetch_fn=dht.fetch,
        transport_id=name,
        clock=clock,
        observer_locator=locator,
    )
    return gossip


CLUSTERS: dict[str, str] = {}


def observe(scorer, clock, subject, outcome=Outcome.OK, count=5, latency=50.0):
    for _ in range(count):
        scorer.record(
            Observation("local", subject, RANGE, outcome, latency, clock.now())
        )


@pytest.fixture(autouse=True)
def clear_clusters():
    CLUSTERS.clear()
    yield
    CLUSTERS.clear()


# ---- publishing --------------------------------------------------------------


def test_publishing_nothing_is_not_published():
    """An empty record is noise that costs a DHT write and tells readers nothing."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    node = make_node("alice", clock, dht, seed=1)
    assert node.publish() is None
    assert dht.writes == 0


def test_published_record_carries_local_observations():
    clock = ManualClock()
    dht = FakeDHT(clock)
    node = make_node("alice", clock, dht, seed=1)
    observe(node.scorer, clock, "smserver1")

    record = node.publish()
    assert record is not None
    assert record["observer"] == node.identity.peer_id
    assert [r["subject"] for r in record["reports"]] == ["smserver1"]


def test_epoch_increases_every_publish():
    """Anti-replay depends on it, so a stalled epoch would silently disable the defence."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    node = make_node("alice", clock, dht, seed=1)
    observe(node.scorer, clock, "smserver1")

    first = node.publish()
    clock.advance(60)
    second = node.publish()
    assert second["epoch"] > first["epoch"]


def test_batches_are_capped_and_keep_the_best_evidenced():
    from seedmesh.reputation.records import MAX_REPORTS_PER_BATCH

    clock = ManualClock()
    dht = FakeDHT(clock)
    node = make_node("alice", clock, dht, seed=1)
    for index in range(MAX_REPORTS_PER_BATCH + 20):
        observe(node.scorer, clock, f"smserver{index:04d}", count=1 + (index % 7))

    reports = node.build_reports()
    assert len(reports) == MAX_REPORTS_PER_BATCH
    assert reports[0].attempts >= reports[-1].attempts


# ---- fetching ----------------------------------------------------------------


def test_a_peers_record_is_ingested_and_influences_aggregation():
    clock = ManualClock()
    dht = FakeDHT(clock)
    CLUSTERS.update({"smtarget": "asn:1"})

    alice = make_node("alice", clock, dht, seed=1)
    bob = make_node("bob", clock, dht, seed=2)
    CLUSTERS[alice.identity.peer_id] = "asn:10"
    CLUSTERS[bob.identity.peer_id] = "asn:20"

    # Bob measures the target badly and publishes.
    observe(bob.scorer, clock, "smtarget", outcome=Outcome.ERROR, count=20)
    bob.publish()

    assert alice.fetch(["bob"]) == 1
    assert alice.aggregator.aggregate("smtarget").reliability < 0.4


def test_you_never_fetch_your_own_record():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    observe(alice.scorer, clock, "smtarget")
    alice.publish()
    assert alice.fetch(["alice"]) == 0


def test_expired_records_are_not_returned():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    bob = make_node("bob", clock, dht, seed=2)
    observe(bob.scorer, clock, "smtarget")
    bob.publish()

    clock.advance(bob.ttl_s + 60)
    assert alice.fetch(["bob"]) == 0


def test_a_replayed_record_is_rejected():
    """Re-publishing an old epoch must not overwrite a newer view."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    bob = make_node("bob", clock, dht, seed=2)

    observe(bob.scorer, clock, "smtarget")
    first = bob.publish()
    assert alice.fetch(["bob"]) == 1

    clock.advance(30)
    bob.publish()
    assert alice.fetch(["bob"]) == 1

    # Put the stale record back and re-fetch.
    dht.store[bob.key_for("bob")] = (first, clock.now() + 600)
    assert alice.fetch(["bob"]) == 0
    assert alice.stats.rejected >= 1


def test_a_forged_record_is_rejected():
    """A record claiming another author must not be accepted just because it parses."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    bob = make_node("bob", clock, dht, seed=2)

    observe(bob.scorer, clock, "smtarget")
    record = bob.publish()
    record["observer"] = Identity.from_seed(bytes([9]) * 32).peer_id  # claim someone else
    dht.store[bob.key_for("bob")] = (record, clock.now() + 600)

    assert alice.fetch(["bob"]) == 0
    assert alice.stats.rejected == 1


def test_tampered_reports_are_rejected():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    bob = make_node("bob", clock, dht, seed=2)

    observe(bob.scorer, clock, "smtarget", outcome=Outcome.ERROR, count=10)
    record = bob.publish()
    record["reports"][0]["ok"] = 9999.0  # flip a bad report into a glowing one
    dht.store[bob.key_for("bob")] = (record, clock.now() + 600)

    assert alice.fetch(["bob"]) == 0


def test_missing_records_are_skipped_quietly():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    assert alice.fetch(["nobody", "alsonobody"]) == 0
    assert alice.stats.rejected == 0


def test_observer_locator_is_told_where_a_record_came_from():
    """The aggregator's per-cluster caps need an observer's network, or they do not bind."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    seen: list[tuple[str, str]] = []

    alice = make_node("alice", clock, dht, seed=1, locator=lambda o, t: seen.append((o, t)))
    bob = make_node("bob", clock, dht, seed=2)
    observe(bob.scorer, clock, "smtarget")
    bob.publish()
    alice.fetch(["bob"])

    assert seen == [(bob.identity.peer_id, "bob")]


def test_publishing_advertises_the_observer_in_the_directory():
    """Discovery only returns servers, so observers must be findable some other way."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    bob = make_node("bob", clock, dht, seed=2)
    observe(bob.scorer, clock, "smtarget")
    bob.publish()

    alice = make_node("alice", clock, dht, seed=1)
    assert alice.known_observers() == ["bob"]


def test_the_directory_excludes_yourself():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    observe(alice.scorer, clock, "smtarget")
    alice.publish()
    assert alice.known_observers() == []


def test_an_observer_with_nothing_to_say_does_not_advertise():
    """A directory full of empty observers costs every reader a lookup per round."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    bob = make_node("bob", clock, dht, seed=2)
    assert bob.publish() is None

    alice = make_node("alice", clock, dht, seed=1)
    assert alice.known_observers() == []


def test_sync_finds_observers_it_was_never_told_about():
    """The whole point of the directory: client-to-client gossip with no prior knowledge."""
    clock = ManualClock()
    dht = FakeDHT(clock)
    CLUSTERS["smtarget"] = "asn:1"

    bob = make_node("bob", clock, dht, seed=2)
    CLUSTERS[bob.identity.peer_id] = "asn:20"
    observe(bob.scorer, clock, "smtarget", outcome=Outcome.ERROR, count=20)
    bob.publish()

    alice = make_node("alice", clock, dht, seed=1)
    CLUSTERS[alice.identity.peer_id] = "asn:10"
    observe(alice.scorer, clock, "smtarget", count=1)

    published, accepted = alice.sync()  # no peer list supplied at all
    assert published is not None
    assert accepted == 1
    assert alice.aggregator.aggregate("smtarget").reliability < 0.5


def test_empty_directory_is_handled():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    assert alice.known_observers() == []


def test_sync_publishes_and_fetches_in_one_round():
    clock = ManualClock()
    dht = FakeDHT(clock)
    alice = make_node("alice", clock, dht, seed=1)
    bob = make_node("bob", clock, dht, seed=2)

    observe(alice.scorer, clock, "smtarget")
    observe(bob.scorer, clock, "smtarget")
    bob.publish()

    published, accepted = alice.sync(["bob"])
    assert published is not None and accepted == 1


# ---- persistence -------------------------------------------------------------


def test_scores_survive_a_restart(tmp_path):
    clock = ManualClock()
    scorer = ReputationScorer(ScorerConfig(), clock=clock)
    observe(scorer, clock, "smgood", count=40)
    observe(scorer, clock, "smbad", outcome=Outcome.ERROR, count=40)
    before_good = scorer.breakdown("smgood").reliability
    before_bad = scorer.breakdown("smbad").reliability

    path = tmp_path / "state" / "reputation.json"
    scorer.save(path)

    restored = ReputationScorer(ScorerConfig(), clock=clock)
    assert restored.load(path) == 2
    assert restored.breakdown("smgood").reliability == pytest.approx(before_good, abs=1e-6)
    assert restored.breakdown("smbad").reliability == pytest.approx(before_bad, abs=1e-6)


def test_quarantine_survives_a_restart(tmp_path):
    """Integrity has a 30-day half-life precisely so it outlives a session."""
    clock = ManualClock()
    scorer = ReputationScorer(ScorerConfig(), clock=clock)
    scorer.record_direct_fault("smcheat")
    assert scorer.breakdown("smcheat").is_quarantined

    path = tmp_path / "reputation.json"
    scorer.save(path)

    restored = ReputationScorer(ScorerConfig(), clock=clock)
    restored.load(path)
    assert restored.breakdown("smcheat").is_quarantined


def test_decay_applies_across_the_restart_gap(tmp_path):
    """Reloading must not resume as though no time passed while the client was down."""
    clock = ManualClock()
    scorer = ReputationScorer(ScorerConfig(reliability_half_life_s=3600.0), clock=clock)
    observe(scorer, clock, "smpeer", count=64)
    path = tmp_path / "reputation.json"
    scorer.save(path)

    later = ManualClock(clock.now() + 3600.0)
    restored = ReputationScorer(ScorerConfig(reliability_half_life_s=3600.0), clock=later)
    restored.load(path)

    assert restored.breakdown("smpeer").observations == pytest.approx(32.0, rel=1e-3)


def test_loading_absent_or_corrupt_state_is_not_fatal(tmp_path):
    scorer = ReputationScorer(ScorerConfig(), clock=ManualClock())
    assert scorer.load(tmp_path / "missing.json") == 0

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert scorer.load(corrupt) == 0


def test_save_is_atomic(tmp_path):
    """A crash mid-save must not leave a truncated file that resets every score."""
    clock = ManualClock()
    scorer = ReputationScorer(ScorerConfig(), clock=clock)
    observe(scorer, clock, "smpeer")
    path = tmp_path / "reputation.json"
    scorer.save(path)
    scorer.save(path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
