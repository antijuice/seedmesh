"""Human-readable rendering of scenario results."""

from __future__ import annotations

from seedmesh.sim.scenarios import GossipResult, RunMetrics
from seedmesh.verification.compare import Tolerance
from seedmesh.sim.world import SimWorld


def format_tolerance(tolerance: Tolerance) -> str:
    return (
        "calibrated tolerance\n"
        f"  match distance      <= {tolerance.match_distance:.5f}\n"
        f"  mismatch distance   >= {tolerance.mismatch_distance:.5f}\n"
        f"  inconclusive band      [{tolerance.match_distance:.5f}, "
        f"{tolerance.mismatch_distance:.5f})\n"
        f"  cosine distance     <= {tolerance.cosine_distance:.6f}\n"
        f"  norm deviation      <= {tolerance.norm_deviation:.5f}"
    )


def format_run(metrics: RunMetrics, world: SimWorld, title: str) -> str:
    malicious = sorted(p for p, s in world.servers.items() if s.is_malicious)
    honest_total = sum(1 for s in world.servers.values() if not s.is_malicious)

    detected = len(metrics.quarantined_malicious)
    lines = [
        f"=== {title} ===",
        f"  servers                {len(world.servers)} "
        f"({honest_total} honest / {len(malicious)} malicious)",
        f"  requests               {metrics.requests}",
        f"  succeeded              {metrics.succeeded} ({metrics.success_rate:.1%})",
        f"  failed                 {metrics.failed}",
        f"  unroutable             {metrics.unroutable}",
        "",
        f"  hops total             {metrics.total_hops}",
        f"  hops hitting a cheat   {metrics.malicious_hops} "
        f"({metrics.malicious_hop_rate:.1%})",
        f"    first half           {metrics.malicious_hops_first_half}",
        f"    second half          {metrics.malicious_hops_second_half}",
        "",
        f"  verifications run      {metrics.verifications}",
        f"    match                {metrics.verdict_match}",
        f"    mismatch             {metrics.verdict_mismatch}",
        f"    inconclusive         {metrics.verdict_inconclusive}",
        f"  passthrough detected   {metrics.passthrough_detections}",
        "",
        f"  malicious quarantined  {detected}/{len(malicious)}",
        f"  honest quarantined     {len(metrics.quarantined_honest)} "
        f"(false-quarantine rate {metrics.false_quarantine_rate:.1%})",
        "",
        f"  mean hops              {metrics.mean_hops:.2f}",
        f"  mean latency           {metrics.mean_latency_ms:.0f} ms",
        f"  load gini              {metrics.load_gini():.3f} "
        f"({sum(1 for v in metrics.load.values() if v == 0)} servers idle)",
    ]

    if metrics.quarantined_malicious:
        lines.append("")
        lines.append("  time to detect (request index at first exclusion):")
        for peer_id, index in sorted(metrics.quarantined_malicious.items(), key=lambda kv: kv[1]):
            behaviour = world.servers[peer_id].behaviour.value
            lines.append(f"    {peer_id:<16} {behaviour:<10} at request {index}")

    missed = [p for p in malicious if p not in metrics.quarantined_malicious]
    if missed:
        lines.append("")
        lines.append("  not excluded within the run:")
        for peer_id in missed:
            server = world.servers[peer_id]
            lines.append(
                f"    {peer_id:<16} {server.behaviour.value:<10} "
                f"served {server.requests_served} segments"
            )

    return "\n".join(lines)


def format_gossip(result: GossipResult) -> str:
    return "\n".join(
        [
            "=== sybil gossip resistance ===",
            f"  honest observers say       {result.honest_view:.2f}",
            "  sybil fleet claims         1.00",
            f"  aggregate settled at       {result.sybil_target_score:.3f}",
            f"  distinct clusters counted  {result.distinct_clusters}",
            f"  total weight               {result.total_weight:.2f}",
            f"  records rejected           {result.rejected_records}",
            f"  mismatch corroborated      {result.mismatch_confirmed}",
        ]
    )
