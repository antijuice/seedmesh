"""Drive a real Petals swarm through Seedmesh's trust layer.

The first time reputation and verification see real servers instead of the simulator.

Expects the three-server swarm from tools/churn_test.sh (the tail half is redundant, so
two different servers host the same blocks -- which is what makes a verification pair
possible at all).

    bash tools/churn_test.sh            # in one shell, leave it up
    ~/hmspike/bin/python3 tools/backend_demo.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from seedmesh.backends.petals_backend import PetalsBackend
from seedmesh.core.clock import SystemClock
from seedmesh.core.types import Observation
from seedmesh.reputation.diversity import ClusterIndex, StaticAsnResolver, TableAsnResolver
from seedmesh.reputation.routing_bias import Router
from seedmesh.reputation.scorer import ReputationScorer
from seedmesh.verification.calibrate import ToleranceTable
from seedmesh.verification.compare import Tolerance, compare_sketches
from seedmesh.verification.sampler import SamplerConfig, VerificationSampler

# Anchored to the repo, not the cwd: the swarm scripts run clients from /tmp, and a
# relative default would silently fall back to placeholder thresholds there.
REPO_ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_TABLE = Path(
    os.environ.get("TOLERANCE_TABLE", str(REPO_ROOT / "sessions" / "tolerance_table.json"))
)
ASN_TABLE = Path(
    os.environ.get("ASN_TABLE", str(REPO_ROOT / "data" / "ip2asn-combined.tsv.gz"))
)


def load_tolerances() -> ToleranceTable | None:
    """Measured thresholds if calibration has been run, else None.

    Without a table the sampler falls back to a single global `Tolerance()` -- placeholder
    numbers fitted to simulated noise. With one, every comparison is judged against a
    threshold measured from real honest disagreement between real GPUs.
    """
    if not TOLERANCE_TABLE.exists():
        return None
    return ToleranceTable.load(TOLERANCE_TABLE)

MODEL = os.environ.get("MODEL", "JackFram/llama-160m")
CLIENT_ID = "smdemoclient0"


def main() -> int:
    initial_peers = [Path("/tmp/srv1.addr").read_text().strip()]
    print(f"connecting to {initial_peers}\n")

    backend = PetalsBackend(MODEL, initial_peers)
    clock = SystemClock()
    scorer = ReputationScorer(clock=clock)
    router = Router(scorer)
    clusters = ClusterIndex()

    try:
        # --- 1. discovery ----------------------------------------------------
        n_blocks = backend.n_blocks(MODEL)
        servers = backend.discover(MODEL)
        print(f"model has {n_blocks} blocks; discovered {len(servers)} online server(s):")
        for server in servers:
            print(
                f"  ...{server.peer_id[-8:]}  blocks {server.block_range.start:2d}-"
                f"{server.block_range.end - 1:2d}  addr={server.address or '(unresolved)'}  "
                f"profile={server.profile.key}  cluster={clusters.coarse_of(server)}"
            )
        resolved = sum(1 for s in servers if s.address)
        print(f"  addresses resolved from the p2p layer: {resolved}/{len(servers)}")
        if not servers:
            print("\nno servers online -- is the swarm up?")
            return 1

        # --- 2. routing through the reputation layer -------------------------
        route = router.plan(MODEL, n_blocks, servers)
        if route is None:
            print("\nreputation-biased router could not cover the model")
            return 1
        print(f"\nSeedmesh route ({route.hop_count} hop(s), cost {route.total_cost:.3f}):")
        for hop in route.hops:
            print(
                f"  blocks {hop.covers.start:2d}-{hop.covers.end - 1:2d} via "
                f"...{hop.server.peer_id[-8:]}  score={hop.score:.3f}"
            )

        # --- 3. run the route, feeding outcomes into reputation --------------
        rng = np.random.default_rng(0)
        hidden = rng.standard_normal((4, backend.config.hidden_size)).astype(np.float32)
        nonce = b"\x2a" * 16

        print("\nrunning the pipeline:")
        state = hidden
        for step, hop in enumerate(route.hops):
            result = backend.run_segment(
                hop.server, hop.covers, state, nonce=nonce, step=step
            )
            scorer.record(
                Observation(
                    observer=CLIENT_ID,
                    subject=hop.server.peer_id,
                    block_range=hop.covers,
                    outcome=result.outcome,
                    latency_ms=result.latency_ms,
                    at=clock.now(),
                )
            )
            print(
                f"  ...{hop.server.peer_id[-8:]}  {result.outcome.value:8s} "
                f"{result.latency_ms:7.1f} ms" + (f"  {result.error}" if result.error else "")
            )
            if not result.ok:
                print("\npipeline failed")
                return 1
            state = result.hidden[0] if result.hidden.ndim == 3 else result.hidden

        print(f"  final hidden state: {state.shape}")

        # --- 4. redundant verification on real servers -----------------------
        # Find a block range served by two different peers -- the tail half, by design.
        by_range: dict[tuple[int, int], list] = {}
        for server in servers:
            key = (server.block_range.start, server.block_range.end)
            by_range.setdefault(key, []).append(server)
        pairs = [group for group in by_range.values() if len(group) >= 2]

        if not pairs:
            print("\nno block range is served by two peers; skipping verification")
            return 0

        group = pairs[0]
        subject, verifier = group[0], group[1]
        block_range = subject.block_range
        print(
            f"\nverifying blocks {block_range.start}-{block_range.end - 1}: "
            f"...{subject.peer_id[-8:]} vs ...{verifier.peer_id[-8:]}"
        )

        tolerances = load_tolerances()
        if tolerances is None:
            tolerance = Tolerance()
            print(f"  tolerance: PLACEHOLDER (no table at {TOLERANCE_TABLE}) "
                  f"match<={tolerance.match_distance}")
        else:
            tolerance = tolerances.get(subject.profile.key, verifier.profile.key)
            if tolerance is None:
                print(f"  no calibration for {subject.profile.key} vs {verifier.profile.key}")
                print("  -> this pair is unverifiable; refusing rather than guessing")
                return 0
            print(f"  tolerance: MEASURED, match<={tolerance.match_distance:.6f} "
                  f"mismatch>={tolerance.mismatch_distance:.6f}")

        probe = rng.standard_normal((4, backend.config.hidden_size)).astype(np.float32)
        check_nonce = b"\x99" * 16
        left = backend.run_segment(subject, block_range, probe, nonce=check_nonce, step=0)
        right = backend.run_segment(verifier, block_range, probe, nonce=check_nonce, step=0)

        if not (left.ok and right.ok):
            print("  one side failed; verification inconclusive")
            return 1

        comparison = compare_sketches(left.sketch, right.sketch, tolerance)
        print(f"  relative distance: {comparison.relative_distance:.6f}")
        print(f"  cosine distance:   {comparison.cosine_distance:.8f}")
        print(f"  verdict:           {comparison.verdict.value}  ({comparison.reason})")

        # --- 5. does the diversity constraint now have something to work with? ---
        print("\ndiversity check:")
        print(f"  prefix-only resolver: independent={clusters.are_independent(subject, verifier)}")
        print(f"    {subject.address} -> {clusters.coarse_of(subject)}")
        print(f"    {verifier.address} -> {clusters.coarse_of(verifier)}")
        print("    Both loopback addresses share a /16, so they are correctly judged the")
        print("    same network -- on one host, that is the truthful answer.")

        # The real offline table, if it has been fetched.
        if ASN_TABLE.exists():
            real = TableAsnResolver.from_file(ASN_TABLE)
            print(f"\n  real ASN table ({len(real):,} routed ranges) on public addresses:")
            for probe_ip, who in (("8.8.8.8", "Google"), ("1.1.1.1", "Cloudflare")):
                print(f"    {probe_ip:16s} -> AS{real.resolve(probe_ip)}  ({who})")
            print("  ...and on this swarm's peers:")
            for server in servers:
                print(f"    {server.address:16s} -> {real.resolve(server.address)}"
                      f"  (loopback is not routed, so no AS -- correct)")
        else:
            print(f"\n  real ASN table absent ({ASN_TABLE}); run tools/fetch_asn_table.py")

        # A localhost swarm has no routable addresses, so the real table cannot separate
        # these peers -- correctly. To exercise the rest of the path, stand in for what a
        # real deployment sees: peers in genuinely different autonomous systems. This is
        # the ONLY simulated input remaining, and only because the swarm is single-host.
        simulated_asns = {
            server.address: 64500 + index
            for index, server in enumerate(s for s in servers if s.address)
        }
        geo_clusters = ClusterIndex(StaticAsnResolver(simulated_asns))
        independent = geo_clusters.are_independent(subject, verifier)
        print(f"\n  simulated distinct-AS peers: independent={independent}")
        print(f"    {subject.address} -> {geo_clusters.coarse_of(subject)}")
        print(f"    {verifier.address} -> {geo_clusters.coarse_of(verifier)}")

        # --- 6. the sampler, end to end -----------------------------------------
        sampler = VerificationSampler(
            scorer, geo_clusters, SamplerConfig(min_first_seen_gap_s=0.0),
            rng=random.Random(0), tolerances=tolerances,
        )
        chosen = sampler.select_verifier(subject, block_range, [verifier])
        print(f"\n  sampler picks a verifier for ...{subject.peer_id[-8:]}: "
              f"{'...' + chosen.peer_id[-8:] if chosen else 'None (refused)'}")
        if chosen is not None:
            task = sampler.plan(subject, block_range, [verifier], force=True)
            left2 = backend.run_segment(subject, block_range, probe, nonce=task.nonce, step=0)
            right2 = backend.run_segment(chosen, block_range, probe, nonce=task.nonce, step=0)
            if left2.ok and right2.ok:
                verdict = compare_sketches(
                    left2.sketch, right2.sketch, task.tolerance or tolerance
                )
                sampler.record_outcome(task, verdict.verdict, comparable=verdict.comparable)
                print(f"  sampler-driven verification: {verdict.verdict.value} "
                      f"(distance {verdict.relative_distance:.6f})")
                print(f"  subject integrity after: "
                      f"{scorer.breakdown(subject.peer_id).integrity:.4f}")

        print("\nranked servers after this run:")
        for breakdown in scorer.ranked([s.peer_id for s in servers]):
            print(
                f"  ...{breakdown.peer_id[-8:]}  score={breakdown.score:.4f} "
                f"reliability={breakdown.reliability:.3f} obs={breakdown.observations:.0f} "
                f"p95={breakdown.p95_ms:.0f}ms"
            )

        print("\nBACKEND DEMO OK")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
