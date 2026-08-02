"""Signed observation records: how reputation travels between peers.

The spec proposes storing reputation "in the DHT alongside block announcements". A DHT is
an open write surface -- Kademlia will happily store whatever you hand it -- so an
unsigned record there is worth nothing: anyone could publish "peer X is terrible" or
"peer Y (me) is perfect".

This module makes every record **attributable and tamper-evident**:

* signed by the observer's Ed25519 key, with the peer id bound to that key;
* domain-separated, so a signature cannot be replayed as a different record type;
* monotonic per-observer ``epoch``, so an old favourable batch cannot be replayed to
  overwrite a newer unfavourable one;
* short TTL, so an observer that goes offline stops influencing routing;
* self-reports dropped, so a peer cannot vouch for itself;
* hard size caps, so one record cannot exhaust a reader's memory.

Attribution is not truth. A verified batch means "this observer really said this" -- not
"this is accurate". Turning many attributed claims into one number is
:mod:`seedmesh.reputation.aggregate`'s job.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from seedmesh.core.clock import Clock, SystemClock
from seedmesh.core.identity import (
    Identity,
    PublicIdentity,
    SignatureError,
    check_peer_id_binding,
    public_identity_from_b64,
)
from seedmesh.core.types import PeerId

OBSERVATION_BATCH_CONTEXT = "seedmesh/observation-batch/v1"
RECORD_VERSION = 1

# base64 of a 32-byte key (44) + a 64-byte signature (88), plus JSON punctuation.
SIGNATURE_ENVELOPE_BYTES = 160

MAX_REPORTS_PER_BATCH = 512
"""Hard ceiling on report count. NOT the binding constraint -- see MAX_BATCH_BYTES."""

MAX_BATCH_BYTES = 2048
"""Serialized size a batch may reach before reports are dropped.

The cap that matters, and it was missing. A batch was capped only by report COUNT, and 512
reports serialize to **76.6 KB**, while a DHT store's success falls off as the payload grows.

Two runs against the live swarm, and they agree on direction while disagreeing sharply on
magnitude:

    payload        run 1    run 2
      ~12-256 B      5%      40%
      ~2-4 KB       55%   20-50%
      8-16 KB         -      70%

So the honest claim is "bigger is worse, and 76 KB is hopeless" -- NOT a calibrated curve.
The absolute rate swings enough between runs that any single number for it, including the
"1 in 3" previously recorded here, describes one afternoon rather than the system.

The mechanism is not the network. The first sweep died inside hivemind's own DHT process:

    BlockingIOError: [Errno 11] Resource temporarily unavailable
      hivemind.dht.dht._run -> self._inner_pipe.recv()

hivemind runs the DHT in a separate process and talks to it over a multiprocessing pipe, and
large values overflow it. That makes this a LOCAL limit rather than swarm flakiness, which
is why capping the payload is the right fix and why raising the retry count is not.

Dropping reports is safe in a way truncating most things is not: the batch is sorted by
evidence first, so what is lost is the least-supported claims, and the next round carries
them again."""
MAX_TTL_S = 3600.0
"""Longest a batch may claim to stay valid. Caps how long a departed observer keeps voting."""

MAX_CLOCK_SKEW_S = 300.0
"""Tolerance for honest clock disagreement. Records from further in the future are rejected
rather than trusted, so a peer cannot mint a batch that outlives MAX_TTL_S by lying about
``issued_at``."""


class RecordError(Exception):
    """Raised when a record is malformed, expired, replayed, or fails validation."""


class UnchangedRecord(RecordError):
    """A record already ingested, re-read before its author published a new one.

    A subclass of RecordError so every existing caller keeps rejecting it -- this is not a
    relaxation of the anti-replay rule. It exists only so callers that report to a human can
    separate "nothing new yet" from "someone is feeding us bad records".
    """


@dataclass(frozen=True, slots=True)
class PeerReport:
    """One observer's summary of its experience with one subject."""

    subject: PeerId
    ok: float
    failed: float
    mismatches: float
    p50_ms: float
    p95_ms: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "ok": float(self.ok),
            "failed": float(self.failed),
            "mismatches": float(self.mismatches),
            "p50_ms": float(self.p50_ms),
            "p95_ms": float(self.p95_ms),
        }

    @classmethod
    def from_wire(cls, data: Any) -> "PeerReport":
        if not isinstance(data, dict):
            raise RecordError("report must be an object")
        try:
            subject = data["subject"]
            report = cls(
                subject=subject,
                ok=float(data["ok"]),
                failed=float(data["failed"]),
                mismatches=float(data["mismatches"]),
                p50_ms=float(data["p50_ms"]),
                p95_ms=float(data["p95_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordError(f"malformed report: {exc}") from exc
        if not isinstance(subject, str) or not subject:
            raise RecordError("report subject must be a non-empty peer id")
        for name, value in (
            ("ok", report.ok),
            ("failed", report.failed),
            ("mismatches", report.mismatches),
            ("p50_ms", report.p50_ms),
            ("p95_ms", report.p95_ms),
        ):
            if value < 0 or value != value or value in (float("inf"), float("-inf")):
                raise RecordError(f"report field {name} must be finite and non-negative")
        return report

    @property
    def attempts(self) -> float:
        return self.ok + self.failed


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """A verified batch of one observer's reports.

    Only ever produced by :func:`build_batch` (locally) or :func:`verify_batch` (from the
    wire). There is intentionally no way to construct a "trusted" batch from raw data
    without passing signature verification.
    """

    observer: PeerId
    epoch: int
    issued_at: float
    expires_at: float
    reports: tuple[PeerReport, ...]

    def report_for(self, subject: PeerId) -> Optional[PeerReport]:
        for report in self.reports:
            if report.subject == subject:
                return report
        return None

    def is_expired(self, now: float) -> bool:
        return now > self.expires_at


def _body(
    observer: PeerId,
    epoch: int,
    issued_at: float,
    expires_at: float,
    reports: Iterable[PeerReport],
) -> dict[str, Any]:
    """The exact structure that gets signed. Field names are part of the wire contract."""
    return {
        "v": RECORD_VERSION,
        "observer": observer,
        "epoch": int(epoch),
        "issued_at": float(issued_at),
        "expires_at": float(expires_at),
        "reports": [report.to_wire() for report in reports],
    }


def _signed_size(body: dict[str, Any]) -> int:
    """Size of the record as it goes on the wire, including the signature envelope.

    The signature and public key are fixed-size base64 fields, so they are added as a
    constant rather than by signing a batch that is about to be discarded.

    Deliberately measured with JSON's default separators rather than compact ones. The wire
    format is actually msgpack, which is smaller than either -- so this over-estimates, and
    over-estimating is the safe direction: the cost is a few reports deferred to the next
    round, where under-estimating puts a record on the wire that will not store.
    """
    return len(json.dumps(body)) + SIGNATURE_ENVELOPE_BYTES


def _fit_to_budget(identity, reports, body, epoch, issued_at, ttl_s):
    """Drop the least-evidenced reports until the batch fits MAX_BATCH_BYTES.

    Proportional estimate then verify, rather than one-at-a-time: a full batch is ~37x over
    budget, and removing reports singly would serialize it hundreds of times.
    """
    size = _signed_size(body)
    if size <= MAX_BATCH_BYTES or not reports:
        return reports

    for _ in range(8):
        if len(reports) <= 1:
            # One report can exceed the budget on its own (a pathological subject id). An
            # oversized batch still verifies and may still store; an EMPTY one is a
            # correctly-signed claim about nothing, which is strictly worse.
            break
        estimate = int(len(reports) * MAX_BATCH_BYTES / size)
        keep = max(1, min(estimate, len(reports) - 1))
        reports = reports[:keep]
        body = _body(identity.peer_id, epoch, issued_at, issued_at + ttl_s, reports)
        size = _signed_size(body)
        if size <= MAX_BATCH_BYTES:
            break
    return reports


def build_batch(
    identity: Identity,
    reports: Iterable[PeerReport],
    *,
    epoch: int,
    clock: Optional[Clock] = None,
    ttl_s: float = 900.0,
) -> dict[str, Any]:
    """Build a signed, wire-ready observation batch.

    Self-reports are dropped here rather than rejected, so a caller that naively passes its
    whole scorer state produces a valid batch instead of an error.
    """
    clock = clock or SystemClock()
    if ttl_s <= 0 or ttl_s > MAX_TTL_S:
        raise ValueError(f"ttl_s must be in (0, {MAX_TTL_S}], got {ttl_s}")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    filtered = [r for r in reports if r.subject != identity.peer_id]
    # Sorted before either cap so both drop the LEAST-evidenced reports. A batch truncated
    # arbitrarily would discard well-supported claims at random.
    filtered.sort(key=lambda r: -r.attempts)
    filtered = filtered[:MAX_REPORTS_PER_BATCH]

    issued_at = clock.now()
    body = _body(identity.peer_id, epoch, issued_at, issued_at + ttl_s, filtered)
    filtered = _fit_to_budget(identity, filtered, body, epoch, issued_at, ttl_s)
    body = _body(identity.peer_id, epoch, issued_at, issued_at + ttl_s, filtered)
    signature = identity.sign(body, context=OBSERVATION_BATCH_CONTEXT)
    return {
        **body,
        "key": base64.b64encode(identity.public_bytes).decode("ascii"),
        "sig": base64.b64encode(signature).decode("ascii"),
    }


class EpochStore:
    """Highest epoch seen per observer -- the anti-replay memory.

    Without this, a captured batch stays valid for its whole TTL and can be re-published
    repeatedly. Worse, an observer whose score dropped could re-publish an older, more
    flattering batch. Requiring strictly increasing epochs closes both.
    """

    __slots__ = ("_highest",)

    def __init__(self) -> None:
        self._highest: dict[PeerId, int] = {}

    def check_and_update(self, observer: PeerId, epoch: int) -> None:
        previous = self._highest.get(observer)
        if previous is not None and epoch == previous:
            # Benign and extremely common: an observer publishes every few minutes, and any
            # reader polling faster than that re-reads the same record. Rejected all the
            # same -- ingesting it twice would double that observer's weight -- but split
            # out as its own type so a healthy swarm does not report a rising "rejected"
            # count that reads like an attack.
            raise UnchangedRecord(
                f"unchanged batch from {observer}: epoch {epoch} already ingested"
            )
        if previous is not None and epoch < previous:
            raise RecordError(
                f"stale or replayed batch from {observer}: epoch {epoch} < seen {previous}"
            )
        self._highest[observer] = epoch

    def highest(self, observer: PeerId) -> Optional[int]:
        return self._highest.get(observer)


def verify_batch(
    raw: Any,
    *,
    clock: Optional[Clock] = None,
    epochs: Optional[EpochStore] = None,
) -> ObservationBatch:
    """Validate a batch from the wire. Raises :class:`RecordError` on any failure.

    Order matters: cheap structural checks run before the expensive signature check, so
    malformed floods cost an attacker more than they cost us.
    """
    clock = clock or SystemClock()
    now = clock.now()

    if not isinstance(raw, dict):
        raise RecordError("batch must be an object")
    if raw.get("v") != RECORD_VERSION:
        raise RecordError(f"unsupported record version: {raw.get('v')!r}")

    observer = raw.get("observer")
    if not isinstance(observer, str) or not observer:
        raise RecordError("batch requires a non-empty observer id")

    epoch = raw.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise RecordError("batch requires a non-negative integer epoch")

    try:
        issued_at = float(raw["issued_at"])
        expires_at = float(raw["expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordError(f"batch requires numeric issued_at/expires_at: {exc}") from exc

    if expires_at <= issued_at:
        raise RecordError("batch expires_at must be after issued_at")
    if expires_at - issued_at > MAX_TTL_S:
        raise RecordError(
            f"batch ttl {expires_at - issued_at:.0f}s exceeds maximum {MAX_TTL_S:.0f}s"
        )
    if issued_at > now + MAX_CLOCK_SKEW_S:
        raise RecordError("batch issued too far in the future")
    if now > expires_at:
        raise RecordError("batch has expired")

    raw_reports = raw.get("reports")
    if not isinstance(raw_reports, list):
        raise RecordError("batch reports must be a list")
    if len(raw_reports) > MAX_REPORTS_PER_BATCH:
        raise RecordError(
            f"batch carries {len(raw_reports)} reports, limit is {MAX_REPORTS_PER_BATCH}"
        )

    reports = [PeerReport.from_wire(item) for item in raw_reports]

    seen: set[PeerId] = set()
    for report in reports:
        if report.subject in seen:
            raise RecordError(f"duplicate report for subject {report.subject}")
        seen.add(report.subject)
        if report.subject == observer:
            raise RecordError("observer may not report on itself")

    key_b64, sig_b64 = raw.get("key"), raw.get("sig")
    if not isinstance(key_b64, str) or not isinstance(sig_b64, str):
        raise RecordError("batch requires base64 'key' and 'sig' fields")

    try:
        public: PublicIdentity = public_identity_from_b64(key_b64)
        check_peer_id_binding(public, observer)
        signature = base64.b64decode(sig_b64, validate=True)
        public.verify(
            _body(observer, epoch, issued_at, expires_at, reports),
            signature,
            context=OBSERVATION_BATCH_CONTEXT,
        )
    except SignatureError as exc:
        raise RecordError(str(exc)) from exc
    except Exception as exc:
        raise RecordError(f"signature check failed: {exc}") from exc

    # Only bump the epoch after everything else passes, so a malformed batch cannot burn
    # an observer's epoch space and lock out its own later, valid records.
    if epochs is not None:
        epochs.check_and_update(observer, epoch)

    return ObservationBatch(
        observer=observer,
        epoch=epoch,
        issued_at=issued_at,
        expires_at=expires_at,
        reports=tuple(reports),
    )
