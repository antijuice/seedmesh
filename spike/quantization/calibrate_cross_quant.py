"""Measure the honest-disagreement distribution for every pair of quantization modes.

A single sample is not a calibration. This runs many independent inputs through the same
block held at different precisions, and reports the distribution of sketch distances for
each pair -- which is what a per-pair tolerance table has to be fitted to.

Honest limitation, stated because it bounds how far these numbers travel: this measures the
**quantization** component of honest disagreement on one GPU. The **hardware** component
(different architectures, different kernels) needs several real GPU models and cannot be
obtained here. Quantization dominates -- NF4 error is roughly an order of magnitude above
fp16-vs-bf16 -- so these are the right shape, but a deployment must re-fit with hardware
diversity folded in.

    .venv-quant/Scripts/python.exe spike/quantization/calibrate_cross_quant.py
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantize_block import quantize_module_petals  # noqa: E402
from test_quantization import run_block, test_config  # noqa: E402

from seedmesh.verification.calibrate import CalibrationCollector, calibrate  # noqa: E402
from seedmesh.verification.compare import relative_l2  # noqa: E402
from seedmesh.verification.sketch import derive_seed, sketch_hidden_state  # noqa: E402

SAMPLES = 60
MODES = ("fp16", "bf16", "int8", "nf4")


def build_variants(config: LlamaConfig, device: torch.device) -> dict:
    torch.manual_seed(0)
    reference = LlamaDecoderLayer(config, layer_idx=0).eval()
    reference_fp16 = copy.deepcopy(reference).to(torch.float16)

    variants = {
        "fp16": copy.deepcopy(reference_fp16).to(device),
        "bf16": copy.deepcopy(reference).to(torch.bfloat16).to(device),
        "int8": quantize_module_petals(copy.deepcopy(reference_fp16), quant_type="int8").to(device),
        "nf4": quantize_module_petals(copy.deepcopy(reference_fp16), quant_type="nf4").to(device),
    }
    return variants


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("needs CUDA: bitsandbytes int8/NF4 kernels are GPU-only")
        return 2

    config = test_config()
    variants = build_variants(config, device)

    print(f"collecting {SAMPLES} samples per pair on {torch.cuda.get_device_name(0)}\n")

    distances: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in itertools.combinations_with_replacement(MODES, 2)
    }

    generator = torch.Generator().manual_seed(4242)
    for index in range(SAMPLES):
        hidden_fp32 = torch.randn(1, 16, config.hidden_size, generator=generator)
        seed = derive_seed(index.to_bytes(16, "big"), "calib:0-1", 0)

        sketches = {}
        for mode, block in variants.items():
            dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            hidden = hidden_fp32.to(device=device, dtype=dtype)
            output = run_block(block, hidden, config, device)
            sketches[mode] = sketch_hidden_state(output.float().cpu().numpy()[0], seed=seed)

        for left, right in distances:
            distances[(left, right)].append(
                relative_l2(sketches[left].values, sketches[right].values)
            )

    print(f"{'pair':<14} {'p50':>10} {'p99':>10} {'p99.9':>10} {'max':>10}")
    print("-" * 58)
    table = {}
    for (left, right), values in distances.items():
        array = np.asarray(values)
        p50, p99, p999 = (float(np.quantile(array, q)) for q in (0.5, 0.99, 0.999))
        print(f"{left + ' vs ' + right:<14} {p50:>10.5f} {p99:>10.5f} {p999:>10.5f} "
              f"{array.max():>10.5f}")
        table[f"{left}|{right}"] = {"p50": p50, "p99": p99, "p999": p999}

    print()
    print("Fitted per-pair tolerances (target FPR 0.1%, 6x mismatch multiplier):")
    print(f"{'pair':<14} {'match<=':>10} {'mismatch>=':>12}")
    print("-" * 40)

    fitted = {}
    for (left, right), values in distances.items():
        collector = CalibrationCollector()
        for value in values:
            # Cosine and norm are not separately measured here; the relative distance is
            # what drives the MISMATCH decision, so it is what gets fitted.
            collector.add(value, value / 100.0, value / 10.0)
        tolerance = calibrate(collector.sample(), target_false_positive_rate=0.001)
        fitted[f"{left}|{right}"] = {
            "match_distance": tolerance.match_distance,
            "mismatch_distance": tolerance.mismatch_distance,
            "cosine_distance": tolerance.cosine_distance,
            "norm_deviation": tolerance.norm_deviation,
        }
        print(f"{left + ' vs ' + right:<14} {tolerance.match_distance:>10.5f} "
              f"{tolerance.mismatch_distance:>12.5f}")

    out = Path(__file__).with_name("cross_quant_observed.json")
    out.write_text(json.dumps({"observed": table, "fitted": fitted}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")

    # Also emit a directly loadable ToleranceTable, keyed by ComputeProfile keys, so the
    # measurement can be consumed rather than transcribed.
    from seedmesh.core.types import ComputeProfile
    from seedmesh.verification.calibrate import ToleranceTable

    def profile_key(mode: str) -> str:
        if mode in ("fp16", "bf16"):
            return ComputeProfile(quant="none", dtype=mode, attn="eager").key
        return ComputeProfile(quant=mode, dtype="fp16", attn="eager").key

    tolerance_table = ToleranceTable()
    for (left, right), values in distances.items():
        collector = CalibrationCollector()
        for value in values:
            collector.add(value, value / 100.0, value / 10.0)
        tolerance_table.set(
            profile_key(left), profile_key(right),
            calibrate(collector.sample(), target_false_positive_rate=0.001),
        )

    table_path = Path(__file__).with_name("tolerance_table.json")
    tolerance_table.save(
        table_path,
        notes=(
            f"spike/quantization, {torch.cuda.get_device_name(0)}, {SAMPLES} samples/pair. "
            "QUANTIZATION component only -- single GPU, so the hardware-diversity component "
            "of honest disagreement is NOT included. Re-fit before production use."
        ),
    )
    print(f"wrote {table_path.name} ({len(tolerance_table)} pairs, loadable via ToleranceTable.load)")

    single = 0.01707
    print(f"\nWith the single global threshold ({single}), these pairs would never MATCH:")
    for (left, right), values in distances.items():
        if float(np.quantile(np.asarray(values), 0.5)) > single:
            print(f"  {left} vs {right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
