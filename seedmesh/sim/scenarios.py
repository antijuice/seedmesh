"""Closed-loop scenarios: routing feeds verification feeds reputation feeds routing.

Each scenario runs the real modules against :mod:`seedmesh.sim.world`. Nothing is stubbed,
so the numbers these produce are evidence about the design rather than about the test.

The questions each scenario is built to answer:

``calibrate_world``    what distance separates honest hardware disagreement from cheating?
``run_swarm``          does a swarm carrying bad actors converge on routing around them,
                       and does it do so without evicting honest volunteers?
``run_gossip``         can a sybil fleet move the aggregate score of a peer it has never
                       actually used?
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from seedmesh.core.clock import ManualClock
from seedmesh.core.identity import Identity
from seedmesh.core.types import BlockRange, Observation, PeerId, Verdict
from seedmesh.reputation.aggregate import AggregationConfig, ReputationAggregator
from seedmesh.reputation.diversity import ClusterIndex
from seedmesh.reputation.records import EpochStore, PeerReport, build_batch, verify_batch
from seedmesh.reputation.routing_bias import Router, RoutingConfig
from seedmesh.reputation.scorer import ReputationScorer, ScorerConfig
from seedmesh.verification.calibrate import CalibrationCollector, calibrate
from seedmesh.verification.compare import Tolerance, compare_sketches, detect_passthrough
from seedmesh.verification.sampler import SamplerConfig, VerificationSampler
from seedmesh.sim.world import Behaviour, SimWorld

CLIENT_PEER_ID = "smclient0000"


def calibrate_world(
    world: SimWorld,
    *,
    samples: int = 120,
    target_false_positive_rate: float = 0.001,
) -> Tolerance:
    """Measure honest-pair distances in this swarm and fit thresholds to them.

    This is the workflow a real deployment follows during bring-up, using servers the
    operator controls and therefore knows to be honest.
    """
    honest = [s for s in world.servers.values() if s.behaviour is Behaviour.HONEST]
    if len(honest) < 2:
        raise ValueError("calibration needs at least two honest servers")

    collector = CalibrationCollector()
    rng = random.Random(99)

    for index in range(samples):
        left, right = rng.sample(honest, 2)
        overlap_start = max(left.block_range.start, right.block_range.start)
        overlap_end = min(left.block_range.end, right.block_range.end)
        if overlap_end <= overlap_start:
            continue

        block_range = BlockRange(world.model_id, overlap_start, overlap_end)
        hidden = world.fresh_hidden()
        nonce = index.to_bytes(16, "big")

        a = world.run_segment(left.info(), block_range, hidden, nonce=nonce, step=index)
        b = world.run_segment(right.info(), block_range, hidden, nonce=nonce, step=index)
        if not (a.ok and b.ok):
            continue

        comparison = compare_sketches(a.sketch, b.sketch, Tolerance())
        collector.add(
            comparison.relative_distance,
            comparison.cosine_distance,
            comparison.norm_deviation,
        )

    return calibrate(
        collector.sample(), target_false_positive_rate=target_false_positive_rate
    )


@dataclass
class RunMetrics:
    """What a scenario run observed. All counts are ground truth from the simulator."""

    requests: int = 0
    succeeded: int = 0
    failed: int = 0
    unroutable: int = 0

    total_hops: int = 0
    malicious_hops: int = 0

    verifications: int = 0
    verdict_match: int = 0
    verdict_mismatch: int = 0
    verdict_inconclusive: int = 0
    passthrough_detections: int = 0

    quarantined_malicious: dict[PeerId, int] = field(default_factory=dict)
    quarantined_honest: dict[PeerId, int] = field(default_factory=dict)
    load: dict[PeerId, int] = field(default_factory=dict)

    malicious_hops_first_half: int = 0
    malicious_hops_second_half: int = 0
    hop_counts: list[int] = field(default_factory=list)
    latency_ms_total: float = 0.0

    @property
    def mean_hops(self) -> float:
        return sum(self.hop_counts) / len(self.hop_counts) if self.hop_counts else 0.0

    @property
    def mean_latency_ms(self) -> float:
        """Mean end-to-end pipeline latency. Load balancing must not wreck this."""
        return self.latency_ms_total / self.succeeded if self.succeeded else 0.0

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.requests if self.requests else 0.0

    @property
    def malicious_hop_rate(self) -> float:
        return self.malicious_hops / self.total_hops if self.total_hops else 0.0

    @property
    def false_quarantine_rate(self) -> float:
        """Honest servers wrongly excluded, as a fraction of all quarantines."""
        total = len(self.quarantined_honest) + len(self.quarantined_malicious)
        return len(self.quarantined_honest) / total if total else 0.0

    def load_gini(self) -> float:
        """Inequality of request distribution. 0 = perfectly even, 1 = one server takes all."""
        values = sorted(self.load.values())
        if not values or sum(values) == 0:
            return 0.0
        n = len(values)
        cumulative = sum((i + 1) * v for i, v in enumerate(values))
        return (2 * cumulative) / (n * sum(values)) - (n + 1) / n


def run_swarm(
    world: SimWorld,
    *,
    requests: int = 400,
    tolerance: Optional[Tolerance] = None,
    seed: int = 42,
    step_seconds: float = 5.0,
    scorer_config: Optional[ScorerConfig] = None,
    routing_config: Optional[RoutingConfig] = None,
    sampler_config: Optional[SamplerConfig] = None,
    enable_verification: bool = True,
) -> RunMetrics:
    """Drive a full client loop against the simulated swarm."""
    tolerance = tolerance or Tolerance()
    clock = world.clock
    rng = random.Random(seed)

    scorer = ReputationScorer(scorer_config or ScorerConfig(), clock=clock)
    clusters = ClusterIndex()
    router = Router(scorer, routing_config or RoutingConfig())
    sampler = VerificationSampler(
        scorer, clusters, sampler_config or SamplerConfig(), rng=rng
    )

    metrics = RunMetrics()
    n_blocks = world.n_blocks(world.model_id)
    half = requests // 2

    # Seed every known server at zero. Without this, load inequality is computed only over
    # servers that received work, which hides the servers that received none -- the exact
    # population the metric exists to detect.
    for peer_id in world.servers:
        metrics.load[peer_id] = 0

    for index in range(requests):
        world.churn()
        clock.advance(step_seconds)
        metrics.requests += 1

        candidates = world.discover(world.model_id)
        route = router.plan(world.model_id, n_blocks, candidates)
        if route is None:
            metrics.unroutable += 1
            continue

        hidden = world.fresh_hidden()
        failed = False
        request_latency = 0.0
        metrics.hop_counts.append(route.hop_count)

        for hop_index, hop in enumerate(route.hops):
            attempted: set[PeerId] = set()
            server_info = hop.server
            result = None

            # One failover attempt per hop: the churn case the design assumes is normal.
            for _ in range(2):
                nonce = rng.getrandbits(128).to_bytes(16, "big")
                result = world.run_segment(
                    server_info, hop.covers, hidden, nonce=nonce, step=hop_index
                )
                metrics.total_hops += 1
                metrics.load[server_info.peer_id] = metrics.load.get(server_info.peer_id, 0) + 1
                router.note_assignment(server_info.peer_id, len(hop.covers))

                sim_server = world.servers[server_info.peer_id]
                if sim_server.is_malicious:
                    metrics.malicious_hops += 1
                    if index < half:
                        metrics.malicious_hops_first_half += 1
                    else:
                        metrics.malicious_hops_second_half += 1

                scorer.record(
                    Observation(
                        observer=CLIENT_PEER_ID,
                        subject=server_info.peer_id,
                        block_range=hop.covers,
                        outcome=result.outcome,
                        latency_ms=result.latency_ms,
                        at=clock.now(),
                    )
                )

                request_latency += result.latency_ms
                if result.ok:
                    break

                attempted.add(server_info.peer_id)
                replacements = router.alternates(hop.covers, candidates, exclude=attempted)
                if not replacements:
                    break
                server_info = replacements[0]

            if result is None or not result.ok:
                failed = True
                break

            if enable_verification:
                self_check = world.sketch_of(hidden, nonce, hop.covers, hop_index)
                if len(hop.covers) >= 2 and detect_passthrough(self_check, result.sketch):
                    # Witnessed directly by this client, so it needs no corroborating peer.
                    metrics.passthrough_detections += 1
                    scorer.record_direct_fault(server_info.peer_id)

                task = sampler.plan(server_info, hop.covers, candidates)
                if task is not None:
                    peer_result = world.run_segment(
                        task.verifier, hop.covers, hidden, nonce=nonce, step=hop_index
                    )
                    if peer_result.ok:
                        comparison = compare_sketches(result.sketch, peer_result.sketch, tolerance)
                        metrics.verifications += 1
                        if comparison.verdict is Verdict.MATCH:
                            metrics.verdict_match += 1
                        elif comparison.verdict is Verdict.MISMATCH:
                            metrics.verdict_mismatch += 1
                        else:
                            metrics.verdict_inconclusive += 1
                        sampler.record_outcome(
                            task, comparison.verdict, comparable=comparison.comparable
                        )

            hidden = result.hidden

        if failed:
            metrics.failed += 1
        else:
            metrics.succeeded += 1
            metrics.latency_ms_total += request_latency

        _record_quarantines(world, scorer, router, metrics, index)

    return metrics


def _record_quarantines(
    world: SimWorld,
    scorer: ReputationScorer,
    router: Router,
    metrics: RunMetrics,
    index: int,
) -> None:
    """Note the first request at which each server crossed the exclusion threshold."""
    for peer_id, server in world.servers.items():
        if not router.is_excluded(peer_id):
            continue
        target = (
            metrics.quarantined_malicious if server.is_malicious else metrics.quarantined_honest
        )
        target.setdefault(peer_id, index)


@dataclass
class GossipResult:
    """Outcome of an aggregation scenario."""

    honest_view: float
    sybil_target_score: float
    distinct_clusters: int
    total_weight: float
    rejected_records: int
    mismatch_confirmed: bool


def run_gossip(
    *,
    honest_observers: int = 5,
    sybil_observers: int = 50,
    sybil_asn: int = 64999,
    honest_opinion: float = 0.2,
    sybil_opinion: float = 1.0,
    seed: int = 5,
) -> GossipResult:
    """Can a sybil fleet overturn the honest consensus about one peer?

    Setup: a genuinely bad server that honest observers all rate poorly, and a large fleet
    of sybil observers -- all in one ASN, as a cheap fleet would be -- publishing glowing
    reports about it. The fleet outnumbers the honest observers ten to one.

    If aggregation counted observers, the fleet wins trivially. The result below is what
    happens when it counts network clusters and takes a weighted median instead.
    """
    clock = ManualClock()
    target = "smtarget00000"

    addresses: dict[PeerId, tuple[str, int]] = {}
    identities: list[Identity] = []

    for index in range(honest_observers):
        identity = Identity.from_seed(bytes([index + 1]) * 32)
        identities.append(identity)
        addresses[identity.peer_id] = (f"203.0.{index + 1}.10", 64500 + index)

    for index in range(sybil_observers):
        identity = Identity.from_seed(bytes([(index % 200) + 40]) * 31 + bytes([index % 251]))
        identities.append(identity)
        addresses[identity.peer_id] = (f"198.51.100.{index % 250 + 1}", sybil_asn)

    addresses[target] = ("192.0.2.5", 64444)

    def cluster_lookup(peer_id: PeerId) -> str:
        entry = addresses.get(peer_id)
        return f"asn:{entry[1]}" if entry else "cluster:unknown"

    aggregator = ReputationAggregator(cluster_lookup, AggregationConfig())
    epochs = EpochStore()
    rejected = 0

    for index, identity in enumerate(identities):
        is_honest = index < honest_observers
        opinion = honest_opinion if is_honest else sybil_opinion
        attempts = 40
        ok = round(attempts * opinion)
        report = PeerReport(
            subject=target,
            ok=float(ok),
            failed=float(attempts - ok),
            mismatches=1.0 if is_honest else 0.0,
            p50_ms=120.0 if is_honest else 20.0,
            p95_ms=400.0 if is_honest else 30.0,
        )
        raw = build_batch(identity, [report], epoch=1, clock=clock, ttl_s=900.0)
        try:
            aggregator.ingest(verify_batch(raw, clock=clock, epochs=epochs))
        except Exception:
            rejected += 1

    score = aggregator.aggregate(target)
    return GossipResult(
        honest_view=honest_opinion,
        sybil_target_score=score.reliability,
        distinct_clusters=score.distinct_clusters,
        total_weight=score.total_weight,
        rejected_records=rejected,
        mismatch_confirmed=score.mismatch_confirmed,
    )
