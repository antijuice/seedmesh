"""Sharing reputation across the swarm.

`records.py` and `aggregate.py` have always been able to sign, verify and combine
observations. Nothing published them. Until this module existed, every client learned only
from its own requests and shared nothing -- so the property the design is *for*, that a new
client benefits from the swarm's collective experience, was not true in practice.

**Storage layout.** One DHT key per observer:

    key   = "{prefix}/rep/{observer transport id}"
    value = a signed ObservationBatch covering every peer that observer has data on

Keying by observer rather than by subject matters for two reasons. It keeps each observer's
epoch strictly monotonic (a per-subject layout would publish many records per round sharing
one epoch, and the anti-replay check would reject all but the first). And it avoids a
hotspot: a per-subject key accumulates a value proportional to the number of observers, so a
popular server's record would grow without bound.

**Finding observers.** A DHT cannot enumerate keys, and discovery only returns *servers* --
but the peers with observations to share are *clients*, who announce no blocks and are
therefore invisible. Without solving that, a client can publish and nobody ever reads it.

So observers advertise themselves in a small directory:

    key    = "{prefix}/observers"
    subkey = observer transport id
    value  = a tiny pointer record

hivemind stores per-subkey values under one key and returns them together, which is exactly
an enumerable set. The directory entry is a few dozen bytes, so the shared key stays small
however many observers join, while the bulky batches live under their own per-observer keys.

**Two identities, deliberately.** The DHT slot is addressed by the peer's *transport* id
(libp2p), because that is what discovery returns. Authorship comes from the Ed25519 signature
inside the record. So the transport id says *where a record was found* and the signature says
*who wrote it* -- and a peer relaying someone else's valid record is harmless, because it
cannot alter the attribution.

**Transport-agnostic on purpose.** The DHT is injected as two callables rather than imported,
so the whole publish/fetch/verify/aggregate cycle is testable without hivemind, a running
swarm, or Linux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol

from seedmesh.core.clock import Clock, SystemClock
from seedmesh.core.identity import Identity
from seedmesh.core.types import PeerId
from seedmesh.reputation.aggregate import ReputationAggregator
from seedmesh.reputation.records import (
    MAX_REPORTS_PER_BATCH,
    EpochStore,
    PeerReport,
    RecordError,
    build_batch,
    verify_batch,
)
from seedmesh.reputation.scorer import ReputationScorer

DEFAULT_TTL_S = 900.0


class PublishFn(Protocol):
    def __call__(
        self, key: str, value: dict, expiration_time: float, subkey: Optional[str] = None
    ) -> bool:  # pragma: no cover
        ...


class FetchFn(Protocol):
    def __call__(self, keys: list[str]) -> dict[str, Optional[dict]]:  # pragma: no cover
        ...


@dataclass
class GossipStats:
    published: int = 0
    reports_published: int = 0
    fetched: int = 0
    accepted: int = 0
    rejected: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def note_rejection(self, reason: str) -> None:
        self.rejected += 1
        key = reason.split(":")[0][:60]
        self.rejections[key] = self.rejections.get(key, 0) + 1


class ReputationGossip:
    """Publishes what this node measured, and ingests what others measured."""

    def __init__(
        self,
        identity: Identity,
        scorer: ReputationScorer,
        aggregator: ReputationAggregator,
        *,
        publish_fn: PublishFn,
        fetch_fn: FetchFn,
        transport_id: str,
        prefix: str = "seedmesh",
        clock: Optional[Clock] = None,
        ttl_s: float = DEFAULT_TTL_S,
        observer_locator: Optional[Callable[[PeerId, str], None]] = None,
    ) -> None:
        self.identity = identity
        self.scorer = scorer
        self.aggregator = aggregator
        self.publish_fn = publish_fn
        self.fetch_fn = fetch_fn
        self.transport_id = transport_id
        self.prefix = prefix
        self.clock: Clock = clock or SystemClock()
        self.ttl_s = ttl_s
        self.epochs = EpochStore()
        self.stats = GossipStats()

        # Called with (ed25519 observer id, transport id it was fetched from) so the caller
        # can attribute an observer to a network cluster. Without this the aggregator has no
        # address for an observer and its per-cluster weight caps would not bind -- the
        # anti-sybil rule would be present but toothless.
        self.observer_locator = observer_locator

        self._epoch = 0

    def key_for(self, transport_id: str) -> str:
        return f"{self.prefix}/rep/{transport_id}"

    @property
    def directory_key(self) -> str:
        return f"{self.prefix}/observers"

    def announce_self(self) -> bool:
        """Advertise that this node publishes reputation, so others can find its records."""
        return bool(
            self.publish_fn(
                key=self.directory_key,
                value={"t": self.transport_id, "at": self.clock.now()},
                expiration_time=self.clock.now() + self.ttl_s,
                subkey=self.transport_id,
            )
        )

    def known_observers(self) -> list[str]:
        """Transport ids advertised in the directory, excluding ourselves.

        Entries are hints, not claims to trust: anything fetched because of one still has to
        pass full signature verification. A hostile directory entry costs a wasted lookup,
        which is why the directory carries pointers rather than data.
        """
        found = self.fetch_fn([self.directory_key])
        entry = found.get(self.directory_key)
        if not isinstance(entry, dict):
            return []
        observers = []
        for subkey, value in entry.items():
            transport = None
            if isinstance(value, dict):
                transport = value.get("t")
            # hivemind returns ValueWithExpiration per subkey; unwrap if present.
            elif hasattr(value, "value") and isinstance(getattr(value, "value"), dict):
                transport = value.value.get("t")
            transport = transport or (subkey if isinstance(subkey, str) else None)
            if transport and transport != self.transport_id:
                observers.append(str(transport))
        return sorted(set(observers))

    # ---- publishing ------------------------------------------------------------

    def build_reports(self) -> list[PeerReport]:
        """Turn local first-hand measurement into shareable reports."""
        reports = []
        for subject, summary in self.scorer.export_reports().items():
            reports.append(
                PeerReport(
                    subject=subject,
                    ok=summary["ok"],
                    failed=summary["failed"],
                    mismatches=summary["mismatches"],
                    p50_ms=summary["p50_ms"],
                    p95_ms=summary["p95_ms"],
                )
            )
        # Publish the best-evidenced subjects first; the batch is capped, and a truncated
        # batch should carry the reports backed by the most observations.
        reports.sort(key=lambda r: -r.attempts)
        return reports[:MAX_REPORTS_PER_BATCH]

    def publish(self) -> Optional[dict]:
        """Sign and publish this node's observations. Returns the record, or None if empty."""
        reports = self.build_reports()
        if not reports:
            return None  # nothing measured yet; publishing an empty record is just noise

        self._epoch += 1
        record = build_batch(
            self.identity, reports, epoch=self._epoch, clock=self.clock, ttl_s=self.ttl_s
        )
        expiration = self.clock.now() + self.ttl_s
        if self.publish_fn(key=self.key_for(self.transport_id), value=record,
                           expiration_time=expiration):
            self.stats.published += 1
            self.stats.reports_published += len(reports)
            # Only advertise once there is something worth fetching -- a directory full of
            # observers with nothing to say costs every reader a lookup per round.
            self.announce_self()
            return record
        return None

    # ---- fetching --------------------------------------------------------------

    def fetch(self, transport_ids: Iterable[str]) -> int:
        """Fetch, verify and ingest other peers' records. Returns how many were accepted.

        Every record is verified before it can influence anything: signature, peer-id
        binding, expiry, size caps, self-report rejection and epoch monotonicity. A record
        that fails any of those is counted and dropped -- never partially applied.
        """
        targets = [t for t in transport_ids if t != self.transport_id]
        if not targets:
            return 0

        keys = {self.key_for(t): t for t in targets}
        found = self.fetch_fn(list(keys))
        accepted = 0

        for key, raw in found.items():
            if raw is None:
                continue
            self.stats.fetched += 1
            try:
                batch = verify_batch(raw, clock=self.clock, epochs=self.epochs)
            except RecordError as exc:
                self.stats.note_rejection(str(exc))
                continue
            except Exception as exc:  # malformed beyond RecordError's expectations
                self.stats.note_rejection(f"{type(exc).__name__}: {exc}")
                continue

            if self.observer_locator is not None:
                self.observer_locator(batch.observer, keys.get(key, ""))
            self.aggregator.ingest(batch)
            self.stats.accepted += 1
            accepted += 1

        self.aggregator.drop_expired(self.clock.now())
        return accepted

    # ---- convenience -----------------------------------------------------------

    def sync(self, transport_ids: Iterable[str] = ()) -> tuple[Optional[dict], int]:
        """One publish/fetch round.

        Fetches from the directory *and* any explicitly supplied peers, since a node that
        both serves and observes is reachable by either route.
        """
        published = self.publish()
        targets = set(transport_ids) | set(self.known_observers())
        accepted = self.fetch(targets)
        return published, accepted

    def describe(self) -> list[str]:
        stats = self.stats
        lines = [
            f"  published   {stats.published} record(s), {stats.reports_published} report(s)",
            f"  fetched     {stats.fetched}, accepted {stats.accepted}, rejected {stats.rejected}",
        ]
        for reason, count in sorted(stats.rejections.items(), key=lambda kv: -kv[1]):
            lines.append(f"    rejected x{count}: {reason}")
        return lines
