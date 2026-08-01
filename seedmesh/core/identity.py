"""Self-certifying peer identity and canonical signing.

Two properties carry the whole trust model:

1. **Self-certifying ids.** A peer id is a hash of the peer's Ed25519 public key, so any
   record can be checked against its claimed author with no registry and no trusted
   introducer. Forging a record "from" another peer requires their private key.

2. **Canonical bytes.** Two peers must serialize the same logical record to byte-identical
   input before signing, or signatures fail across machines for no reason. Plain
   ``json.dumps`` is not canonical: key order and float repr both vary. This module pins
   sorted keys, compact separators, and fixed float precision.

Note what identity does *not* buy: it makes records **attributable**, not **true**. Anyone
can generate unlimited keypairs, so a peer id is a pseudonym, not a scarce resource.
Sybil resistance comes from the aggregation rules in :mod:`seedmesh.reputation.aggregate`,
not from here.
"""

from __future__ import annotations

import base64
import enum
import hashlib
import json
import math
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from seedmesh.core.types import PeerId

PEER_ID_PREFIX = "sm"
PEER_ID_HASH_BYTES = 20
FLOAT_PRECISION = 6
"""Decimal places retained when canonicalizing floats.

Latencies are milliseconds and scores are in [0, 1]; six decimals is far finer than any
quantity we sign, and it removes the platform-dependent shortest-repr differences that
would otherwise make a signature valid on one machine and invalid on another.
"""


class SignatureError(Exception):
    """Raised when a signature is absent, malformed, or does not verify."""


def _normalize(value: Any) -> Any:
    """Recursively coerce a value into a canonical, JSON-serializable form."""
    if isinstance(value, enum.Enum):
        return _normalize(value.value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"cannot canonicalize non-finite float: {value}")
        rounded = round(value, FLOAT_PRECISION)
        # -0.0 and 0.0 are equal but serialize differently; collapse them.
        return rounded + 0.0 if rounded == 0 else rounded
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical records require string keys, got {type(key)!r}")
            out[key] = _normalize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"cannot canonicalize value of type {type(value)!r}")


def canonical_bytes(payload: dict, *, context: str) -> bytes:
    """Serialize ``payload`` to deterministic bytes, domain-separated by ``context``.

    ``context`` is prepended so a signature over one kind of record can never be replayed
    as another kind. Without it, an attacker who obtained a signed observation batch could
    potentially present those same bytes as a signed server announcement.
    """
    if not context:
        raise ValueError("signing context must be a non-empty domain separator")
    body = json.dumps(
        _normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return context.encode("utf-8") + b"\x00" + body


def peer_id_from_public_bytes(public_bytes: bytes) -> PeerId:
    """Derive the canonical peer id from raw Ed25519 public key bytes."""
    if len(public_bytes) != 32:
        raise ValueError(f"expected a 32-byte Ed25519 public key, got {len(public_bytes)}")
    digest = hashlib.sha256(public_bytes).digest()[:PEER_ID_HASH_BYTES]
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return f"{PEER_ID_PREFIX}{encoded}"


class PublicIdentity:
    """A peer's public half: verifies signatures, proves the peer id binding."""

    __slots__ = ("_key", "_public_bytes", "_peer_id")

    def __init__(self, public_bytes: bytes) -> None:
        self._public_bytes = bytes(public_bytes)
        self._key = ed25519.Ed25519PublicKey.from_public_bytes(self._public_bytes)
        self._peer_id = peer_id_from_public_bytes(self._public_bytes)

    @property
    def peer_id(self) -> PeerId:
        return self._peer_id

    @property
    def public_bytes(self) -> bytes:
        return self._public_bytes

    def verify(self, payload: dict, signature: bytes, *, context: str) -> None:
        """Raise :class:`SignatureError` unless ``signature`` covers ``payload``."""
        try:
            self._key.verify(signature, canonical_bytes(payload, context=context))
        except InvalidSignature as exc:
            raise SignatureError(
                f"signature does not verify for peer {self._peer_id} in context {context!r}"
            ) from exc

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PublicIdentity) and other._public_bytes == self._public_bytes

    def __hash__(self) -> int:
        return hash(self._public_bytes)

    def __repr__(self) -> str:
        return f"PublicIdentity({self._peer_id})"


class Identity:
    """A peer's keypair. Signs records; its peer id is derived, never chosen."""

    __slots__ = ("_private", "_public")

    def __init__(self, private_key: ed25519.Ed25519PrivateKey) -> None:
        self._private = private_key
        self._public = PublicIdentity(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )

    @classmethod
    def generate(cls) -> "Identity":
        return cls(ed25519.Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> "Identity":
        """Deterministic identity from a 32-byte seed. For tests and simulation only."""
        if len(seed) != 32:
            raise ValueError(f"seed must be exactly 32 bytes, got {len(seed)}")
        return cls(ed25519.Ed25519PrivateKey.from_private_bytes(seed))

    @classmethod
    def from_private_bytes(cls, data: bytes) -> "Identity":
        return cls(ed25519.Ed25519PrivateKey.from_private_bytes(data))

    def private_bytes(self) -> bytes:
        return self._private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def peer_id(self) -> PeerId:
        return self._public.peer_id

    @property
    def public(self) -> PublicIdentity:
        return self._public

    @property
    def public_bytes(self) -> bytes:
        return self._public.public_bytes

    def sign(self, payload: dict, *, context: str) -> bytes:
        return self._private.sign(canonical_bytes(payload, context=context))

    def __repr__(self) -> str:
        return f"Identity({self.peer_id})"


def public_identity_from_b64(encoded: str) -> PublicIdentity:
    """Parse a base64 public key as carried inside a wire record."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise SignatureError(f"malformed public key encoding: {exc}") from exc
    try:
        return PublicIdentity(raw)
    except ValueError as exc:
        raise SignatureError(str(exc)) from exc


def check_peer_id_binding(identity: PublicIdentity, claimed: PeerId) -> None:
    """Verify a record's claimed author id actually matches its embedded public key.

    Skipping this check is the classic mistake: a record can carry attacker key ``K`` and
    a valid signature by ``K``, while claiming to be authored by victim peer ``V``. The
    signature verifies, and the record is still a forgery.
    """
    if identity.peer_id != claimed:
        raise SignatureError(
            f"peer id binding failed: key derives {identity.peer_id}, record claims {claimed}"
        )


def optional_public_identity(encoded: Optional[str]) -> Optional[PublicIdentity]:
    return public_identity_from_b64(encoded) if encoded else None
