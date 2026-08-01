"""Identity and canonical serialization."""

from __future__ import annotations

import pytest

from seedmesh.core.clock import ManualClock
from seedmesh.core.identity import (
    Identity,
    SignatureError,
    canonical_bytes,
    check_peer_id_binding,
    peer_id_from_public_bytes,
    public_identity_from_b64,
)
from seedmesh.core.types import Outcome


def test_peer_id_is_derived_from_key_not_chosen():
    identity = Identity.from_seed(b"\x01" * 32)
    assert identity.peer_id == peer_id_from_public_bytes(identity.public_bytes)
    assert identity.peer_id.startswith("sm")
    assert Identity.from_seed(b"\x02" * 32).peer_id != identity.peer_id


def test_signature_roundtrip():
    identity = Identity.generate()
    payload = {"a": 1, "b": "x"}
    signature = identity.sign(payload, context="ctx/v1")
    identity.public.verify(payload, signature, context="ctx/v1")


def test_canonical_form_is_key_order_independent():
    left = canonical_bytes({"b": 2, "a": 1}, context="c")
    right = canonical_bytes({"a": 1, "b": 2}, context="c")
    assert left == right


def test_canonical_form_quantizes_floats():
    """Shortest-repr differences across platforms must not change the signed bytes."""
    assert canonical_bytes({"x": 1.0000000001}, context="c") == canonical_bytes(
        {"x": 1.0}, context="c"
    )


def test_canonical_form_normalizes_negative_zero():
    assert canonical_bytes({"x": -0.0}, context="c") == canonical_bytes({"x": 0.0}, context="c")


def test_canonical_form_rejects_non_finite():
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("nan")}, context="c")
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("inf")}, context="c")


def test_canonical_form_encodes_enums_by_value():
    assert canonical_bytes({"o": Outcome.OK}, context="c") == canonical_bytes(
        {"o": "ok"}, context="c"
    )


def test_canonical_form_rejects_unknown_types():
    with pytest.raises(TypeError):
        canonical_bytes({"x": object()}, context="c")


def test_domain_separation_blocks_cross_context_replay():
    identity = Identity.generate()
    payload = {"v": 1}
    signature = identity.sign(payload, context="observation/v1")
    with pytest.raises(SignatureError):
        identity.public.verify(payload, signature, context="announcement/v1")


def test_tampering_with_payload_invalidates_signature():
    identity = Identity.generate()
    signature = identity.sign({"ok": 10}, context="c")
    with pytest.raises(SignatureError):
        identity.public.verify({"ok": 9999}, signature, context="c")


def test_peer_id_binding_catches_key_substitution():
    """A valid signature by the wrong key is still a forgery."""
    victim = Identity.from_seed(b"\x03" * 32)
    attacker = Identity.from_seed(b"\x04" * 32)
    with pytest.raises(SignatureError):
        check_peer_id_binding(attacker.public, victim.peer_id)


def test_public_identity_roundtrips_through_base64():
    import base64

    identity = Identity.generate()
    encoded = base64.b64encode(identity.public_bytes).decode("ascii")
    assert public_identity_from_b64(encoded).peer_id == identity.peer_id


def test_malformed_public_key_is_rejected():
    with pytest.raises(SignatureError):
        public_identity_from_b64("not-base64!!")


def test_clock_refuses_to_move_backwards():
    clock = ManualClock(100.0)
    clock.advance(10)
    with pytest.raises(ValueError):
        clock.set(50.0)
    with pytest.raises(ValueError):
        clock.advance(-1)
