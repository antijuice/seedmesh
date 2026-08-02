"""Signed observation records: forgery, replay, expiry and resource limits."""

from __future__ import annotations

import base64
import copy

import pytest

from seedmesh.core.clock import ManualClock
from seedmesh.core.identity import Identity
from seedmesh.reputation.records import (
    MAX_REPORTS_PER_BATCH,
    EpochStore,
    PeerReport,
    RecordError,
    build_batch,
    verify_batch,
)


def _report(subject="smtarget", ok=9.0, failed=1.0, mismatches=0.0):
    return PeerReport(subject, ok, failed, mismatches, 50.0, 120.0)


def test_valid_batch_roundtrips():
    clock = ManualClock()
    identity = Identity.generate()
    raw = build_batch(identity, [_report()], epoch=1, clock=clock)
    batch = verify_batch(raw, clock=clock)
    assert batch.observer == identity.peer_id
    assert batch.report_for("smtarget").ok == 9.0


def test_forged_signature_is_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report()], epoch=1, clock=clock)
    raw["sig"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_swapping_the_key_but_keeping_the_observer_id_is_rejected():
    """The peer-id binding check: signature valid, authorship still false."""
    clock = ManualClock()
    victim = Identity.from_seed(b"\x07" * 32)
    attacker = Identity.from_seed(b"\x08" * 32)

    raw = build_batch(attacker, [_report()], epoch=1, clock=clock)
    raw["observer"] = victim.peer_id  # claim to be the victim, keep attacker's key

    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_tampering_with_reports_is_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report(ok=1.0, failed=9.0)], epoch=1, clock=clock)
    raw["reports"][0]["ok"] = 999.0
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_replaying_the_same_epoch_is_rejected():
    clock = ManualClock()
    identity = Identity.generate()
    epochs = EpochStore()
    raw = build_batch(identity, [_report()], epoch=5, clock=clock)

    verify_batch(copy.deepcopy(raw), clock=clock, epochs=epochs)
    with pytest.raises(RecordError):
        verify_batch(copy.deepcopy(raw), clock=clock, epochs=epochs)


def test_rolling_back_to_an_older_epoch_is_rejected():
    """A peer must not be able to republish a stale, more flattering batch."""
    clock = ManualClock()
    identity = Identity.generate()
    epochs = EpochStore()

    verify_batch(build_batch(identity, [_report()], epoch=9, clock=clock), clock=clock, epochs=epochs)
    with pytest.raises(RecordError):
        verify_batch(
            build_batch(identity, [_report()], epoch=3, clock=clock), clock=clock, epochs=epochs
        )


def test_a_rejected_batch_does_not_burn_epoch_space():
    """A malformed record must not lock its own author out of publishing valid ones."""
    clock = ManualClock()
    identity = Identity.generate()
    epochs = EpochStore()

    bad = build_batch(identity, [_report()], epoch=4, clock=clock)
    bad["reports"][0]["ok"] = 1234.0
    with pytest.raises(RecordError):
        verify_batch(bad, clock=clock, epochs=epochs)

    assert epochs.highest(identity.peer_id) is None
    verify_batch(build_batch(identity, [_report()], epoch=4, clock=clock), clock=clock, epochs=epochs)


def test_expired_batch_is_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report()], epoch=1, clock=clock, ttl_s=60.0)
    clock.advance(61.0)
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_batch_from_the_far_future_is_rejected():
    clock = ManualClock()
    identity = Identity.generate()
    future = ManualClock(clock.now() + 10_000)
    raw = build_batch(identity, [_report()], epoch=1, clock=future)
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_overlong_ttl_is_rejected():
    clock = ManualClock()
    with pytest.raises(ValueError):
        build_batch(Identity.generate(), [_report()], epoch=1, clock=clock, ttl_s=99_999.0)


def test_self_reports_are_dropped_when_building():
    clock = ManualClock()
    identity = Identity.generate()
    raw = build_batch(
        identity,
        [_report(subject=identity.peer_id), _report(subject="smother")],
        epoch=1,
        clock=clock,
    )
    batch = verify_batch(raw, clock=clock)
    assert [r.subject for r in batch.reports] == ["smother"]


def test_self_reports_are_rejected_when_verifying():
    clock = ManualClock()
    identity = Identity.generate()
    raw = build_batch(identity, [_report(subject="smother")], epoch=1, clock=clock)
    raw["reports"][0]["subject"] = identity.peer_id
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_duplicate_subjects_are_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report(), _report()], epoch=1, clock=clock)
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_oversized_batch_is_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report()], epoch=1, clock=clock)
    raw["reports"] = raw["reports"] * (MAX_REPORTS_PER_BATCH + 2)
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_build_truncates_rather_than_overflowing():
    """CORRECTED 2026-08-02. This asserted truncation to exactly MAX_REPORTS_PER_BATCH,
    which is the behaviour that caused the bug: 512 reports serialize to 76.6 KB, and a DHT
    store of that size almost never succeeds. The count cap is now a ceiling and the BYTE
    budget is what binds."""
    import json

    from seedmesh.reputation.records import MAX_BATCH_BYTES

    clock = ManualClock()
    reports = [_report(subject=f"sm{i:05d}", ok=float(i)) for i in range(MAX_REPORTS_PER_BATCH + 50)]
    raw = build_batch(Identity.generate(), reports, epoch=1, clock=clock)

    assert len(json.dumps(raw)) <= MAX_BATCH_BYTES
    kept = verify_batch(raw, clock=clock).reports
    assert 0 < len(kept) < MAX_REPORTS_PER_BATCH


def test_a_trimmed_batch_keeps_the_best_evidenced_reports():
    """What survives the cut decides whether trimming is safe. Dropping well-supported
    claims at random would make gossip worse than not gossiping."""
    clock = ManualClock()
    reports = [_report(subject=f"sm{i:05d}", ok=float(i)) for i in range(300)]
    raw = build_batch(Identity.generate(), reports, epoch=1, clock=clock)

    kept = {r.subject for r in verify_batch(raw, clock=clock).reports}
    # `ok` rises with i, so the highest-numbered subjects have the most evidence.
    assert "sm00299" in kept
    assert "sm00000" not in kept


def test_a_single_huge_report_still_produces_a_valid_batch():
    """The trim loop must terminate even when one report alone exceeds the budget --
    an empty or malformed batch would be worse than an oversized one."""
    clock = ManualClock()
    raw = build_batch(
        Identity.generate(), [_report(subject="x" * 4000)], epoch=1, clock=clock
    )
    assert len(verify_batch(raw, clock=clock).reports) == 1


def test_negative_counts_are_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report()], epoch=1, clock=clock)
    raw["reports"][0]["ok"] = -5.0
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_unknown_version_is_rejected():
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report()], epoch=1, clock=clock)
    raw["v"] = 99
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


@pytest.mark.parametrize("field", ["observer", "epoch", "issued_at", "expires_at", "key", "sig"])
def test_missing_required_fields_are_rejected(field):
    clock = ManualClock()
    raw = build_batch(Identity.generate(), [_report()], epoch=1, clock=clock)
    del raw[field]
    with pytest.raises(RecordError):
        verify_batch(raw, clock=clock)


def test_non_object_input_is_rejected():
    with pytest.raises(RecordError):
        verify_batch(["not", "a", "dict"], clock=ManualClock())
