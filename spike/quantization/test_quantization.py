"""Does Petals' quantization path survive nine minor versions of bitsandbytes?

Three questions, in order of how much they matter to Seedmesh:

1. **Does it still work?** Petals pins ``bitsandbytes==0.41.1`` (September 2023); current is
   0.50.0. The coupling is only four constructors, so this is a narrow test.

2. **What is the real memory footprint?** The onboarding wizard has to turn "you have 4GB of
   VRAM" into "host N blocks" before a volunteer will trust it. Petals carries an empirical
   constant (``4.25/8`` bytes per NF4 parameter) that is worth re-measuring rather than
   inheriting.

3. **Does quantization break verification?** This is the one that could invalidate the trust
   layer's design, and no upstream project would have thought to ask it. Seedmesh detects
   cheating by comparing two servers' outputs against a calibrated tolerance. If an int8
   server and an fp16 server -- both completely honest -- disagree by more than the mismatch
   threshold, then honest volunteers convict each other. That is the exact bug class already
   found and fixed twice in this project, and it would arrive the moment a mixed-precision
   swarm went live.

Run with the quant venv:
    .venv-quant/Scripts/python.exe spike/quantization/test_quantization.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantize_block import measure_module_bytes, quantize_module_petals  # noqa: E402

# Thresholds produced by `seedmesh simulate` calibration on the healthy preset. These are
# the numbers the trust layer would actually be using.
CALIBRATED_MATCH = 0.01707
CALIBRATED_MISMATCH = 0.10241


def test_config() -> LlamaConfig:
    """TinyLlama-1.1B-shaped block: small enough for 4GB, realistic GQA ratios."""
    return LlamaConfig(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=32000,
        max_position_embeddings=2048,
        _attn_implementation="eager",
    )


def block_param_count(config: LlamaConfig) -> int:
    with torch.device("meta"):
        block = LlamaDecoderLayer(config, layer_idx=0)
    return sum(p.numel() for p in block.parameters())


# ---- question 1: does the API still work -------------------------------------


def probe_api() -> dict:
    import bitsandbytes as bnb

    results = {}
    for path in ("nn.Linear8bitLt", "nn.Int8Params", "nn.LinearNF4", "nn.Params4bit"):
        obj = bnb
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            results[path] = "present"
        except AttributeError as exc:
            results[path] = f"GONE ({exc})"
    return results


def test_quantization_applies(config: LlamaConfig) -> dict:
    """Run Petals' unmodified quantize_module for each type; report what happened."""
    outcomes = {}
    for quant_type in ("int8", "nf4"):
        torch.manual_seed(0)
        block = LlamaDecoderLayer(config, layer_idx=0).eval()
        try:
            quantized = quantize_module_petals(copy.deepcopy(block), quant_type=quant_type)
            linear_types = {
                type(m).__name__
                for m in quantized.modules()
                if "Linear" in type(m).__name__
            }
            outcomes[quant_type] = {"ok": True, "layers": sorted(linear_types)}
        except Exception as exc:
            outcomes[quant_type] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return outcomes


# ---- question 2: memory footprint --------------------------------------------


def test_memory_footprint(config: LlamaConfig, device: torch.device) -> dict:
    n_params = block_param_count(config)
    results = {}

    torch.manual_seed(0)
    # Quantize from an fp16 block, as a real server would: Petals loads blocks in the
    # target dtype and quantizes after, which leaves biases and norms in fp16.
    reference = LlamaDecoderLayer(config, layer_idx=0).eval().to(torch.float16)

    fp16 = copy.deepcopy(reference).to(device)
    results["fp16"] = measure_module_bytes(fp16)
    del fp16
    torch.cuda.empty_cache() if device.type == "cuda" else None

    for quant_type in ("int8", "nf4"):
        block = quantize_module_petals(copy.deepcopy(reference), quant_type=quant_type)
        block = block.to(device)
        results[quant_type] = measure_module_bytes(block)
        del block
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return {"n_params": n_params, "bytes": results}


# ---- question 3: does quantization break verification -------------------------


def run_block(block, hidden, config, device):
    """Drive one block the way a hosted server would."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary = LlamaRotaryEmbedding(config=config).to(device)
    batch, seq, _ = hidden.shape
    position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary(hidden, position_ids)

    minimum = torch.finfo(hidden.dtype).min
    mask = torch.full((seq, seq), minimum, device=device, dtype=hidden.dtype).triu(1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq, seq)

    with torch.no_grad():
        out = block(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    return out[0] if isinstance(out, tuple) else out


def test_verification_impact(config: LlamaConfig, device: torch.device) -> dict:
    """Measure sketch distances between honest servers at different quantization levels."""
    from seedmesh.verification.compare import Tolerance, compare_sketches
    from seedmesh.verification.sketch import derive_seed, sketch_hidden_state

    torch.manual_seed(0)
    reference = LlamaDecoderLayer(config, layer_idx=0).eval()
    reference_fp16 = copy.deepcopy(reference).to(torch.float16)

    variants = {}
    variants["fp16"] = copy.deepcopy(reference_fp16).to(device)
    variants["fp16_b"] = copy.deepcopy(reference_fp16).to(device)
    variants["bf16"] = copy.deepcopy(reference).to(torch.bfloat16).to(device)
    for quant_type in ("int8", "nf4"):
        variants[quant_type] = quantize_module_petals(
            copy.deepcopy(reference_fp16), quant_type=quant_type
        ).to(device)

    torch.manual_seed(42)
    hidden_fp32 = torch.randn(1, 16, config.hidden_size)

    seed = derive_seed(b"\x11" * 16, "quant-test:0-1", 0)
    sketches = {}
    for name, block in variants.items():
        dtype = torch.bfloat16 if name == "bf16" else torch.float16
        hidden = hidden_fp32.to(device=device, dtype=dtype)
        output = run_block(block, hidden, config, device)
        sketches[name] = sketch_hidden_state(
            output.float().cpu().numpy()[0], seed=seed
        )

    tolerance = Tolerance(
        match_distance=CALIBRATED_MATCH,
        mismatch_distance=CALIBRATED_MISMATCH,
        cosine_distance=0.01,
        norm_deviation=0.05,
    )

    pairs = [
        ("fp16", "fp16_b"),
        ("fp16", "bf16"),
        ("fp16", "int8"),
        ("fp16", "nf4"),
        ("int8", "nf4"),
        ("bf16", "nf4"),
    ]

    results = {}
    for left, right in pairs:
        comparison = compare_sketches(sketches[left], sketches[right], tolerance)
        results[f"{left} vs {right}"] = {
            "distance": comparison.relative_distance,
            "verdict": comparison.verdict.value,
        }
    return results


# ---- main --------------------------------------------------------------------


def main() -> int:
    import bitsandbytes
    import transformers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"bitsandbytes {bitsandbytes.__version__} (Petals pins 0.41.1)")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    config = test_config()

    print("=" * 72)
    print("Q1. Does Petals' bitsandbytes API still exist?")
    print("=" * 72)
    for symbol, status in probe_api().items():
        print(f"  {'OK  ' if status == 'present' else 'GONE'}  bnb.{symbol:20s} {status if status != 'present' else ''}")
    print()

    print("  Applying Petals' unmodified quantize_module:")
    outcomes = test_quantization_applies(config)
    api_ok = True
    for quant_type, outcome in outcomes.items():
        if outcome["ok"]:
            print(f"    {quant_type:5s} OK   -> {', '.join(outcome['layers'])}")
        else:
            api_ok = False
            print(f"    {quant_type:5s} FAIL -> {outcome['error']}")
    print()

    if not api_ok:
        print("Quantization failed to apply; skipping downstream measurements.")
        return 1

    print("=" * 72)
    print("Q2. Real memory footprint (what the onboarding wizard must predict)")
    print("=" * 72)
    footprint = test_memory_footprint(config, device)
    n_params = footprint["n_params"]
    print(f"  block parameters: {n_params:,}")
    print(f"  {'mode':<6} {'MiB':>10} {'bytes/param':>12} {'vs fp16':>9}")
    fp16_bytes = footprint["bytes"]["fp16"]
    for mode in ("fp16", "int8", "nf4"):
        b = footprint["bytes"][mode]
        print(f"  {mode:<6} {b / 2**20:>10.1f} {b / n_params:>12.3f} {b / fp16_bytes:>8.1%}")
    print()
    print("  Petals' own constants for comparison: int8 = 1.0 bytes/param,")
    print("  nf4 = 4.25/8 = 0.531 bytes/param (its comment says 'measured empirically').")
    print()

    print("  Blocks that fit in common volunteer VRAM (Llama-3-8B-shaped block,")
    print("  ~218M params, leaving 1GiB headroom for activations and KV cache):")
    big = LlamaConfig(
        hidden_size=4096, intermediate_size=14336, num_hidden_layers=1,
        num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
    )
    big_params = block_param_count(big)
    print(f"    (block parameters: {big_params:,})")
    print(f"    {'VRAM':>6} {'fp16':>6} {'int8':>6} {'nf4':>6}")
    for vram_gb in (4, 8, 12, 24):
        usable = (vram_gb * 2**30) - 2**30
        counts = []
        for mode in ("fp16", "int8", "nf4"):
            per_param = footprint["bytes"][mode] / n_params
            counts.append(int(usable / (big_params * per_param)))
        print(f"    {vram_gb:>4}GB {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}")
    print()

    print("=" * 72)
    print("Q3. Does quantization break Seedmesh's verification? [THE IMPORTANT ONE]")
    print("=" * 72)
    print(f"  Calibrated thresholds: MATCH <= {CALIBRATED_MATCH}, "
          f"MISMATCH >= {CALIBRATED_MISMATCH}")
    print("  All pairs below are HONEST servers. Any 'mismatch' is a false accusation.")
    print()
    impact = test_verification_impact(config, device)
    print(f"  {'pair':<20} {'distance':>10}  verdict")
    print(f"  {'-' * 20} {'-' * 10}  {'-' * 12}")
    false_accusations = []
    for pair, result in impact.items():
        flag = ""
        if result["verdict"] == "mismatch":
            flag = "  <-- FALSE ACCUSATION"
            false_accusations.append(pair)
        elif result["verdict"] == "inconclusive":
            flag = "  <-- degraded"
        print(f"  {pair:<20} {result['distance']:>10.5f}  {result['verdict']}{flag}")
    print()

    if false_accusations:
        print("  RESULT: cross-quantization verification produces FALSE ACCUSATIONS between")
        print("  honest servers. Verification pairs MUST be matched on quantization type.")
        for pair in false_accusations:
            print(f"    - {pair}")
    else:
        print("  RESULT: no false accusations across quantization levels at these thresholds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
