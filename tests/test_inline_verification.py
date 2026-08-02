"""Inline verification: checking real requests while a client is being used.

The Petals-shaped operations are injected, so all of this runs with no backend, no GPU and
no swarm. What is being tested is the policy -- what gets captured, what gets replayed, and
what a disagreement is allowed to conclude.
"""

from __future__ import annotations

import numpy as np
import pytest

from seedmesh.core.types import (
    BlockRange,
    ComputeProfile,
    ServerInfo,
    Verdict,
)
from seedmesh.reputation.diversity import ClusterIndex
from seedmesh.reputation.scorer import ReputationScorer
from seedmesh.verification.inline import InlineVerifier, is_prefill
from seedmesh.verification.sampler import SamplerConfig, VerificationSampler

MODEL = "test/model"
DIM = 32
PROFILE = ComputeProfile(quant="none", dtype="float32", attn="eager")


# Distinct addresses on purpose. The sampler refuses to pair peers it cannot show are
# independent operators, so two servers behind one address are correctly ineligible to check
# each other -- the first version of these tests shared an address and the sampler was right
# to refuse.
ADDRESSES = {"a": "203.0.113.1", "b": "198.51.100.7", "c": "192.0.2.9"}


def server(peer_id, start=0, end=4, address=None):
    address = address or ADDRESSES.get(peer_id, f"203.0.113.{abs(hash(peer_id)) % 250 + 1}")
    return ServerInfo(
        peer_id=peer_id,
        block_range=BlockRange(MODEL, start, end),
        first_seen=0.0,
        address=address,
        profile=PROFILE,
    )


def activation(seed=0, tokens=6):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, tokens, DIM)).astype(np.float32)


class FakeSession:
    """Shaped like Petals' `_ServerInferenceSession`, for the prefill check."""

    def __init__(self, position):
        self.position = position


# Every rate pinned to 1.0, so these test the mechanism rather than the dice.
# `verification_rate` interpolates between unproven_rate and base_rate by confidence, and
# drops to suspicion_rate once integrity falls -- which a MISMATCH causes. Setting only
# base_rate left the second replay in a two-round test to a coin flip.
ALWAYS = SamplerConfig(
    min_first_seen_gap_s=0.0,
    base_rate=1.0,
    unproven_rate=1.0,
    suspicion_rate=1.0,
    max_rate=1.0,
)


def build(run_on, servers, *, config=None):
    scorer = ReputationScorer()
    sampler = VerificationSampler(scorer, ClusterIndex(), config or ALWAYS)
    verifier = InlineVerifier(sampler, run_on=run_on, discover=lambda: servers)
    return verifier, scorer


# ---- the prefill constraint --------------------------------------------------
#
# This is the one that would manufacture fraud. Petals inference is stateful: a decode step's
# output depends on the KV cache built from every previous token, so replaying its input on
# another server compares a computation that had history against one that did not. The
# outputs differ for entirely honest reasons, and a sampler that replayed decode steps would
# report constant MISMATCHes against innocent peers.


def test_only_the_prefill_step_is_replayable():
    assert is_prefill(FakeSession(position=0)) is True
    assert is_prefill(FakeSession(position=1)) is False
    assert is_prefill(FakeSession(position=57)) is False


def test_a_session_without_a_position_is_not_assumed_replayable():
    # Unknown state must fail closed. Guessing wrong here accuses honest servers.
    assert is_prefill(object()) is False
    assert is_prefill(None) is False


# ---- capture -----------------------------------------------------------------


def test_capture_keeps_what_is_needed_to_replay():
    verifier, _ = build(lambda *a: None, [])
    assert verifier.capture("peer-a", 0, 4, MODEL, activation(), activation(1)) is True
    step = verifier.pending[0]
    assert step.subject == "peer-a"
    assert step.block_range == BlockRange(MODEL, 0, 4)


def test_capture_is_bounded():
    # Each entry holds a prompt-sized activation; an unbounded queue on the request path is
    # a memory leak that grows with how much someone uses the client.
    verifier, _ = build(lambda *a: None, [])
    for _ in range(50):
        verifier.capture("peer-a", 0, 4, MODEL, activation(), activation())
    assert len(verifier.pending) == verifier.max_pending


def test_capture_never_raises_into_the_request_path():
    verifier, _ = build(lambda *a: None, [])
    assert verifier.capture("peer-a", 0, 4, MODEL, "not a tensor", None) is False
    assert verifier.capture("peer-a", 4, 4, MODEL, activation(), activation()) is False


# ---- verdicts ----------------------------------------------------------------


def test_an_identical_replay_is_a_match():
    output = activation(7)
    verifier, scorer = build(lambda v, br, hidden, nonce: output, [server("a"), server("b")])
    verifier.capture("a", 0, 4, MODEL, activation(), output)

    (outcome,) = verifier.run_pending()
    assert outcome.verdict is Verdict.MATCH
    assert outcome.subject == "a"
    assert outcome.verifier == "b"


def test_a_different_result_is_a_mismatch():
    verifier, scorer = build(
        lambda v, br, hidden, nonce: activation(99), [server("a"), server("b")]
    )
    verifier.capture("a", 0, 4, MODEL, activation(), activation(7))

    (outcome,) = verifier.run_pending()
    assert outcome.verdict is Verdict.MISMATCH


def test_a_mismatch_is_charged_to_both_parties():
    # A disagreement proves one of the two is wrong, not which. Charging only the subject
    # would let a malicious verifier defame honest servers at will.
    verifier, scorer = build(
        lambda v, br, hidden, nonce: activation(99), [server("a"), server("b")]
    )
    verifier.capture("a", 0, 4, MODEL, activation(), activation(7))
    verifier.run_pending()

    assert scorer.breakdown("a").confirmed_mismatches > 0
    assert scorer.breakdown("b").confirmed_mismatches > 0


def test_a_verifier_that_does_not_answer_convicts_nobody():
    def refuse(*args):
        raise RuntimeError("verifier offline")

    verifier, scorer = build(refuse, [server("a"), server("b")])
    verifier.capture("a", 0, 4, MODEL, activation(), activation(7))

    (outcome,) = verifier.run_pending()
    assert outcome.verdict is Verdict.INCONCLUSIVE
    assert outcome.comparable is False
    # A failure to compare is evidence about neither party's honesty.
    assert scorer.breakdown("a").confirmed_mismatches == 0
    assert scorer.breakdown("b").confirmed_mismatches == 0


# ---- when verification is impossible -----------------------------------------


def test_no_second_server_means_skipped_not_passed():
    ran = []
    verifier, scorer = build(lambda *a: ran.append(a) or activation(), [server("a")])
    verifier.capture("a", 0, 4, MODEL, activation(), activation())

    assert verifier.run_pending() == []
    assert ran == []
    assert verifier.skipped_no_verifier == 1
    assert verifier.checked == 0


def test_a_skip_is_reported_out_loud():
    # Silence here reads as "everything checked out", which is the opposite of the truth.
    verifier, _ = build(lambda *a: activation(), [server("a")])
    verifier.capture("a", 0, 4, MODEL, activation(), activation())
    verifier.run_pending()
    text = "\n".join(verifier.describe())
    assert "no independent verifier" in text


def test_a_subject_that_has_gone_offline_is_dropped_quietly():
    verifier, _ = build(lambda *a: activation(), [server("b")])
    verifier.capture("vanished", 0, 4, MODEL, activation(), activation())
    assert verifier.run_pending() == []
    assert verifier.checked == 0


def test_a_broken_discovery_does_not_raise():
    def explode():
        raise RuntimeError("dht down")

    scorer = ReputationScorer()
    sampler = VerificationSampler(scorer, ClusterIndex(), SamplerConfig())
    verifier = InlineVerifier(sampler, run_on=lambda *a: None, discover=explode)
    verifier.capture("a", 0, 4, MODEL, activation(), activation())
    assert verifier.run_pending() == []


# ---- pacing ------------------------------------------------------------------


def test_only_the_requested_number_are_replayed_per_round():
    # Verification runs between prompts; draining a whole queue at once would stall the
    # user behind work that has no deadline.
    calls = []
    verifier, _ = build(
        lambda v, br, hidden, nonce: calls.append(1) or activation(7),
        [server("a"), server("b")],
    )
    for _ in range(4):
        verifier.capture("a", 0, 4, MODEL, activation(), activation(7))

    verifier.run_pending(limit=1)
    assert len(calls) == 1
    assert len(verifier.pending) == 3


@pytest.mark.parametrize("verdict_name", ["MATCH", "MISMATCH"])
def test_outcomes_accumulate_for_the_session_summary(verdict_name):
    same = activation(7)
    replay = same if verdict_name == "MATCH" else activation(99)
    verifier, _ = build(lambda v, br, h, n: replay, [server("a"), server("b")])
    for _ in range(2):
        verifier.capture("a", 0, 4, MODEL, activation(), same)
    verifier.run_pending(limit=2)

    assert verifier.checked == 2
    assert verdict_name.lower() in "\n".join(verifier.describe())


# ---- a relayed address is the RELAY's, not the peer's -------------------------
#
# Measured on the live swarm: both servers' only multiaddr was
#   /ip4/143.244.144.71/tcp/31337/p2p/QmRelay.../p2p-circuit
# and that IP is a bootstrap droplet, not the machine hosting the blocks. Taking it at face
# value breaks the diversity rule in the dangerous direction: one operator running two NAT'd
# servers relayed through two different droplets would read as two independent networks and
# could verify its own work.


class FakePeer:
    def __init__(self, peer_id, addrs):
        self.peer_id = peer_id
        self.addrs = addrs


class FakeP2P:
    def __init__(self, peers):
        self._peers = peers

    async def list_peers(self):
        return self._peers


class FakeManager:
    def __init__(self, peers):
        self.state = type("S", (), {"p2p": FakeP2P(peers)})()


def _resolve(peers, monkeypatch):
    import seedmesh.verification.petals_bridge as bridge

    class Worker:
        @staticmethod
        def run_coroutine(coro):
            import asyncio

            return asyncio.new_event_loop().run_until_complete(coro)

    monkeypatch.setitem(
        __import__("sys").modules,
        "hivemind.moe.client.remote_expert_worker",
        type("M", (), {"RemoteExpertWorker": Worker}),
    )
    return bridge.make_address_resolver(FakeManager(peers))()


def test_a_relay_address_is_not_attributed_to_the_peer(monkeypatch):
    relayed = FakePeer(
        "QmServer",
        ["/ip4/143.244.144.71/tcp/31337/p2p/QmRelay/p2p-circuit"],
    )
    assert _resolve([relayed], monkeypatch) == {}


def test_a_direct_address_is_used(monkeypatch):
    direct = FakePeer("QmServer", ["/ip4/66.31.117.31/tcp/31338"])
    assert _resolve([direct], monkeypatch) == {"QmServer": "66.31.117.31"}


def test_a_direct_address_wins_over_a_relayed_one(monkeypatch):
    both = FakePeer(
        "QmServer",
        [
            "/ip4/143.244.144.71/tcp/31337/p2p/QmRelay/p2p-circuit",
            "/ip4/66.31.117.31/tcp/31338",
        ],
    )
    assert _resolve([both], monkeypatch) == {"QmServer": "66.31.117.31"}


def test_two_relay_only_servers_cannot_verify_each_other():
    # The end-to-end consequence, stated as a property: unlocated peers land in the unknown
    # cluster, and the sampler refuses to pair anyone it cannot show is independent.
    clusters = ClusterIndex()
    # Built directly: the `server()` helper substitutes a default address, which is exactly
    # the situation this test needs to NOT have.
    unlocated = [
        ServerInfo(
            peer_id=peer_id,
            block_range=BlockRange(MODEL, 0, 4),
            first_seen=0.0,
            address=None,
            profile=PROFILE,
        )
        for peer_id in ("a", "b")
    ]
    assert clusters.are_independent(*unlocated, min_first_seen_gap_s=0.0) is False


def test_a_relay_only_swarm_is_told_what_would_fix_it():
    """"No independent verifier" has two very different causes and two different remedies."""
    unlocated = ServerInfo(
        peer_id="b", block_range=BlockRange(MODEL, 0, 4), first_seen=0.0,
        address=None, profile=PROFILE,
    )
    verifier, _ = build(lambda *a: activation(), [server("a"), unlocated])
    verifier.capture("a", 0, 4, MODEL, activation(), activation())
    verifier.run_pending()

    assert verifier.skipped_unlocated == 1
    text = "\n".join(verifier.describe())
    assert "belongs to the relay" in text
    assert "Port-forwarded servers can verify" in text
