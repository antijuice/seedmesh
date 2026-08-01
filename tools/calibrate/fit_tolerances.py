"""Merge calibration sessions and fit a ToleranceTable.

Takes the per-GPU session files from collect_sketches.py and produces the thing Seedmesh
actually needs: a tolerance per *pair of compute profiles*, fitted to how far apart honest
servers really land.

The important design point is **which distances get pooled into which threshold**. The
tolerance for "fp16 vs fp16" must cover two honest fp16 servers on *different hardware*, so
cross-session same-mode pairs are exactly what fits it. Pooling those is what turns a
threshold measured on one machine (where same-mode distance is a meaningless 0.0) into one
that survives a heterogeneous swarm.

    python tools/calibrate/fit_tolerances.py sessions/*.json --out tolerance_table.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from seedmesh.core.types import ComputeProfile  # noqa: E402
from seedmesh.verification.calibrate import (  # noqa: E402
    MIN_CALIBRATION_SAMPLES,
    CalibrationCollector,
    ToleranceTable,
    calibrate,
)
from seedmesh.verification.compare import relative_l2  # noqa: E402

# How each collected mode maps onto a published ComputeProfile. `attn` is pinned to eager
# in the collector, and is now published by patched servers (tools/port_petals.py).
MODE_PROFILES = {
    "fp32": ComputeProfile(quant="none", dtype="float32", attn="eager"),
    "fp16": ComputeProfile(quant="none", dtype="float16", attn="eager"),
    "bf16": ComputeProfile(quant="none", dtype="bfloat16", attn="eager"),
    "int8": ComputeProfile(quant="int8", dtype="float16", attn="eager"),
    "nf4": ComputeProfile(quant="nf4", dtype="float16", attn="eager"),
}


def load_sessions(paths: list[Path]) -> list[dict]:
    sessions = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != 1:
            print(f"  skipping {path.name}: unsupported schema {data.get('schema')}")
            continue
        data["_path"] = path
        sessions.append(data)
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fpr", type=float, default=0.001, help="target false-positive rate")
    parser.add_argument("--mismatch-multiplier", type=float, default=6.0)
    args = parser.parse_args()

    sessions = load_sessions(args.sessions)
    if not sessions:
        print("no usable session files")
        return 2

    print(f"loaded {len(sessions)} session(s):")
    for session in sessions:
        available = ", ".join(sorted(session["modes"]))
        print(f"  {session['label']:28s} {session['gpu']:28s} modes: {available}")

    hardware = {s["gpu"] for s in sessions}
    if len(hardware) < 2:
        print(f"\nWARNING: only one GPU type ({sorted(hardware)}).")
        print("Same-mode pairs (fp16 vs fp16, ...) will be ABSENT from the table entirely:")
        print("comparing a run against itself is not a measurement, so it is skipped rather")
        print("than fitted to a meaningless zero. An absent pair makes the sampler refuse to")
        print("verify it, which is correct -- but it means same-profile servers cannot check")
        print("each other until a second GPU type is added.")

    # Verify sessions are actually comparable before pooling anything.
    seeds = {s["input_seed"] for s in sessions}
    models = {s["model"] for s in sessions}
    if len(seeds) > 1 or len(models) > 1:
        print(f"\nFAIL: sessions disagree on inputs (seeds={seeds}, models={models}).")
        print("They must share weights and inputs to be comparable.")
        return 1

    # Every (session, mode) x (session, mode) combination, pooled by profile pair.
    distances: dict[frozenset[str], list[float]] = {}
    pair_sources: dict[frozenset[str], set[str]] = {}

    entries = [
        (session, mode) for session in sessions for mode in session["modes"]
    ]
    for (sa, ma), (sb, mb) in itertools.combinations_with_replacement(entries, 2):
        if sa is sb and ma == mb:
            continue  # identical run against itself: distance is trivially zero
        left = np.asarray(sa["modes"][ma], dtype=np.float64)
        right = np.asarray(sb["modes"][mb], dtype=np.float64)
        count = min(len(left), len(right))
        if count == 0:
            continue

        key = frozenset((MODE_PROFILES[ma].key, MODE_PROFILES[mb].key))
        bucket = distances.setdefault(key, [])
        for index in range(count):
            bucket.append(relative_l2(left[index], right[index]))
        pair_sources.setdefault(key, set()).add(
            f"{sa['label']}/{ma} vs {sb['label']}/{mb}"
        )

    print(f"\n{'profile pair':<58} {'n':>6} {'p50':>9} {'p99.9':>9}")
    print("-" * 86)
    table = ToleranceTable()
    skipped = []

    for key, values in sorted(distances.items(), key=lambda kv: sorted(kv[0])):
        profiles = sorted(key)
        label = " vs ".join(p.split("/")[0] + "/" + p.split("/")[1] for p in profiles)
        array = np.asarray(values)
        p50, p999 = float(np.quantile(array, 0.5)), float(np.quantile(array, 0.999))
        print(f"  {label:<56} {len(values):>6} {p50:>9.5f} {p999:>9.5f}")

        if len(values) < MIN_CALIBRATION_SAMPLES:
            skipped.append((label, f"only {len(values)} samples"))
            continue

        collector = CalibrationCollector()
        for value in values:
            collector.add(value, value / 100.0, value / 10.0)
        tolerance = calibrate(
            collector.sample(),
            target_false_positive_rate=args.fpr,
            mismatch_multiplier=args.mismatch_multiplier,
        )
        left_profile, right_profile = (profiles * 2)[:2]
        table.set(left_profile, right_profile, tolerance)

    if skipped:
        print("\nnot fitted (insufficient samples):")
        for label, reason in skipped:
            print(f"  {label}: {reason}")

    notes = (
        f"fitted from {len(sessions)} session(s) across {len(hardware)} GPU type(s): "
        f"{sorted(hardware)}; model={sessions[0]['model']} block={sessions[0]['block_index']}; "
        f"target FPR={args.fpr}, mismatch multiplier={args.mismatch_multiplier}"
    )
    table.save(args.out, notes=notes)
    print(f"\nwrote {args.out}: {len(table)} profile pair(s)")
    print(f"provenance: {notes}")

    if len(hardware) < 2:
        print("\nThis table is NOT production-usable: single GPU type.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
