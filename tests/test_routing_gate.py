"""The routing gate: letting measured trust change where requests actually go.

Most of these test what must NOT be excluded. Excluding a cheat is easy; the hard part is
not excluding the honest volunteer on a slow connection, which is the failure this project
has had to undo more than once.
"""

from __future__ import annotations

import pytest

from seedmesh.core.types import BlockRange, Observation, Outcome, PeerId, Verdict
from seedmesh.reputation.routing import ReputationGate, make_gate
from seedmesh.reputation.routing_bias import Router
from seedmesh.reputation.scorer import ReputationScorer

RANGE = BlockRange("m", 0, 4)


class FakeBlockInfo:
    def __init__(self, servers):
        self.servers = {peer: object() for peer in servers}


class FakeSequenceInfo:
    def __init__(self, peers):
        self.block_infos = [FakeBlockInfo(peers)]


class FakeState:
    def __init__(self, peers):
        self.sequence_info = FakeSequenceInfo(peers)


class FakeManager:
    """Shaped like Petals' RemoteSequenceManager, for the two attributes the gate touches."""

    def __init__(self, peers):
        self.state = FakeState(peers)
        self.blocked_servers = None
        self.updates = 0

    def update(self, wait=False):
        self.updates += 1


def gate_for(peers, scorer=None):
    scorer = scorer or ReputationScorer()
    manager = FakeManager(peers)
    return ReputationGate(manager, Router(scorer)), manager, scorer


def observe(scorer, peer, outcome=Outcome.OK, count=10, latency=50.0):
    for _ in range(count):
        scorer.record(Observation("me", peer, RANGE, outcome, latency, scorer.clock.now()))


def convict(scorer, peer, partners=("cluster:a", "cluster:b")):
    """Corroborated mismatch: distinct partners, which is what quarantine requires."""
    for partner in partners:
        for _ in range(4):
            scorer.record_verification(peer, Verdict.MISMATCH, partner_cluster=partner)


# ---- what must NOT be excluded -----------------------------------------------


def test_a_slow_but_honest_server_is_never_excluded():
    # The whole reason this is a veto and not a ranking. A score blends latency in, so
    # "prefer high scores" means "prefer fast", and a volunteer on a home connection gets
    # starved for being honest and slow.
    gate, manager, scorer = gate_for(["slowpoke"])
    observe(scorer, "slowpoke", latency=9000.0, count=50)

    assert gate.refresh() == set()
    assert manager.blocked_servers is None


def test_an_unreliable_server_is_not_excluded():
    # Timeouts are churn -- a laptop closing its lid -- not evidence of incorrect work.
    gate, manager, scorer = gate_for(["flaky"])
    observe(scorer, "flaky", outcome=Outcome.TIMEOUT, count=40)

    assert gate.refresh() == set()
    assert scorer.breakdown("flaky").reliability < 0.2, "should still look unreliable"


def test_an_unproven_newcomer_is_not_excluded():
    # Cold start: a peer with no history must be routable or it can never earn any.
    gate, _, _ = gate_for(["brand-new"])
    assert gate.refresh() == set()


def test_a_single_accuser_cannot_evict_anyone():
    # Otherwise one malicious peer could clear the swarm of honest competitors.
    gate, manager, scorer = gate_for(["accused"])
    for _ in range(20):
        scorer.record_verification("accused", Verdict.MISMATCH, partner_cluster="cluster:one")

    assert gate.refresh() == set()
    assert manager.blocked_servers is None


def test_occasional_inconclusive_results_exclude_nobody():
    """Ambiguous results count only if the peer's ambiguous RATE is anomalous (>25%), not
    on raw count. That gate is what protects honest heterogeneity -- and it is what makes
    MoE serving safe: a trained router flips expert selection on at most ~3% of tokens
    (spike/moe/trained_weights.py), an order of magnitude under the threshold."""
    gate, _, scorer = gate_for(["mostly-fine"])
    for cluster in ("cluster:a", "cluster:b"):
        for _ in range(2):
            scorer.record_ambiguous("mostly-fine", cluster)
    for _ in range(96):  # 4 ambiguous in 100 comparisons = 4%
        scorer.record_verification("mostly-fine", Verdict.MATCH, partner_cluster="cluster:c")

    assert gate.refresh() == set()


def test_a_peer_that_is_never_comparable_is_excluded():
    """The other side of the same rule. Every comparison landing ambiguous, across
    independent clusters, is not honest noise -- it is a peer that has never once produced a
    checkable answer."""
    gate, _, scorer = gate_for(["never-comparable"])
    for cluster in ("cluster:a", "cluster:b"):
        for _ in range(10):
            scorer.record_ambiguous("never-comparable", cluster)

    assert gate.refresh() == {"never-comparable"}


# ---- what must be excluded ---------------------------------------------------


def test_a_corroborated_cheat_is_excluded():
    gate, manager, scorer = gate_for(["cheat", "honest"])
    convict(scorer, "cheat")

    assert gate.refresh() == {"cheat"}
    assert manager.blocked_servers is not None
    assert len(manager.blocked_servers) == 1


def test_exclusion_reaches_petals_own_filter():
    # `blocked_servers` is consulted inside _update(), so an excluded peer never enters the
    # routing graph. Setting it without triggering an update would leave the old graph live.
    gate, manager, scorer = gate_for(["cheat"])
    convict(scorer, "cheat")
    gate.refresh()
    assert manager.updates == 1


def test_an_honest_peer_alongside_a_cheat_keeps_routing():
    gate, manager, scorer = gate_for(["cheat", "honest"])
    observe(scorer, "honest", count=30)
    convict(scorer, "cheat")
    gate.refresh()

    blocked = {str(p) for p in (manager.blocked_servers or set())}
    assert "honest" not in blocked


# ---- the set is recomputed, not accumulated ----------------------------------


def test_nothing_is_reapplied_when_the_set_is_unchanged():
    gate, manager, scorer = gate_for(["cheat"])
    convict(scorer, "cheat")
    assert gate.refresh() == {"cheat"}
    assert gate.refresh() == set(), "a stable set should not re-trigger an update"
    assert manager.updates == 1


def test_a_cleared_peer_is_unblocked_without_anyone_remembering_to():
    gate, manager, scorer = gate_for(["cheat"])
    convict(scorer, "cheat")
    gate.refresh()
    assert manager.blocked_servers

    # Integrity recovers (the scorer decays disputes); the gate must follow it back.
    scorer._peers["cheat"].confirmed_mismatches = 0.0
    scorer._peers["cheat"].disputes.clear()
    scorer._peers["cheat"].integrity_pass = 50.0
    gate.refresh()
    assert not manager.blocked_servers


# ---- robustness --------------------------------------------------------------


def test_a_client_with_no_routing_table_yet_is_not_an_error():
    gate = ReputationGate(FakeManager([]), Router(ReputationScorer()))
    gate.sequence_manager.state = object()  # no sequence_info at all
    assert gate.known_peers() == []
    assert gate.refresh() == set()


def test_make_gate_declines_without_a_sequence_manager():
    assert make_gate(None, ReputationScorer()) is None


def test_describe_is_silent_until_something_is_excluded():
    gate, _, scorer = gate_for(["honest"])
    observe(scorer, "honest", count=20)
    gate.refresh()
    assert gate.describe() == []


def test_describe_shows_the_evidence_not_just_the_verdict():
    # A volunteer told they were excluded deserves to see why, in a community-run network.
    gate, _, scorer = gate_for(["cheat"])
    convict(scorer, "cheat")
    gate.refresh()
    text = "\n".join(gate.describe())
    assert "integrity" in text and "disputing partner" in text


@pytest.mark.parametrize("outcome", [Outcome.OK, Outcome.TIMEOUT, Outcome.ERROR, Outcome.REFUSED])
def test_no_request_outcome_alone_ever_excludes(outcome):
    """Reliability and integrity are separate axes on purpose. Only proven-wrong *work*
    closes the gate; failing to answer is a different thing entirely."""
    gate, _, scorer = gate_for(["peer"])
    observe(scorer, "peer", outcome=outcome, count=60)
    assert gate.refresh() == set()
