"""Collect honest-pair calibration sketches on ONE machine. Run once per GPU type.

Every verification threshold in Seedmesh is currently fitted to *simulated* floating-point
noise. This is the script that replaces guesses with measurements.

The design constraint that shapes everything here: to compare two GPUs, both must run the
**same weights** on the **same inputs**, in sessions that may be hours apart. So nothing is
random at runtime -- weights come from a real published model, inputs come from a fixed
seed, and sketch projections come from fixed seeds. Two sessions on different hardware
produce directly comparable outputs.

What travels between sessions is tiny: a sketch is 64 floats, so a full session file is a
few hundred KB. That is what makes Colab viable — you never need two GPUs at once, just
several sessions, and the artefacts merge offline.

Uses the stock `LlamaDecoderLayer`, not Petals' wrapper. What is being measured is the
numerical noise of the same computation across hardware and precision; the ported block
delegates to this layer and was verified numerically exact against it
(`spike/transformers_port/`), so Petals is not needed in Colab.

    python tools/calibrate/collect_sketches.py --out session.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

# Allow running straight from a repo checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from seedmesh.verification.sketch import SketchSpec, derive_seed, sketch_hidden_state  # noqa: E402

MODEL = "JackFram/llama-160m"
BLOCK_INDEX = 0
N_SAMPLES = 64
SEQ_LEN = 16
INPUT_SEED = 20260731
SCHEMA = 1

# fp32 is the reference: it is the least-quantized thing every device can run, so it
# anchors the comparison rather than being just another mode.
MODES = ("fp32", "fp16", "bf16", "int8", "nf4")


def load_reference_block():
    """Block 0 of a real published model -- identical bytes in every session."""
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = AutoConfig.from_pretrained(MODEL)
    config._attn_implementation = "eager"  # pinned: the kernel changes the noise floor

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    block = model.model.layers[BLOCK_INDEX]
    assert isinstance(block, LlamaDecoderLayer)
    return config, block.eval()


def quantize(block, mode: str):
    import copy

    from bitsandbytes import nn as bnb_nn

    if mode in ("fp32", "fp16", "bf16"):
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[mode]
        return copy.deepcopy(block).to(dtype)

    # Petals' own quantize_module logic, applied to the same layer.
    target = copy.deepcopy(block).to(torch.float16)
    for name, module in list(target.named_children()):
        _quantize_children(module, mode, bnb_nn)
        if isinstance(module, torch.nn.Linear):
            setattr(target, name, _make_quantized(module, mode, bnb_nn))
    return target


def _quantize_children(module, mode, bnb_nn):
    for name, child in list(module.named_children()):
        _quantize_children(child, mode, bnb_nn)
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, _make_quantized(child, mode, bnb_nn))


def _make_quantized(linear: torch.nn.Linear, mode: str, bnb_nn):
    if mode == "int8":
        replacement = bnb_nn.Linear8bitLt(
            linear.in_features, linear.out_features, linear.bias is not None,
            has_fp16_weights=False, threshold=6.0,
        )
        replacement.weight = bnb_nn.Int8Params(
            linear.weight.data, requires_grad=False, has_fp16_weights=False
        ).to(linear.weight.dtype)
    elif mode == "nf4":
        replacement = bnb_nn.LinearNF4(
            linear.in_features, linear.out_features, linear.bias is not None,
            compress_statistics=True,
        )
        replacement.weight = bnb_nn.Params4bit(
            linear.weight.data, requires_grad=False, quant_type="nf4",
            blocksize=64, compress_statistics=True,
        ).to(linear.weight.dtype)
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    replacement.bias = linear.bias
    return replacement


def run_block(block, config, hidden, device):
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
            hidden, attention_mask=mask, position_ids=position_ids,
            past_key_values=None, use_cache=False, position_embeddings=position_embeddings,
        )
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--label", type=str, default=None, help="optional note, e.g. 'colab-t4'")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device. int8/NF4 kernels are GPU-only, and the point of this run is")
        print("to characterise real hardware -- a CPU session would measure the wrong thing.")
        return 2

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    capability = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    print(f"GPU: {gpu_name} (compute {capability})")

    config, block = load_reference_block()
    spec = SketchSpec()

    # Inputs are generated once, on CPU, from a fixed seed -- identical in every session.
    generator = torch.Generator().manual_seed(INPUT_SEED)
    inputs = [
        torch.randn(1, SEQ_LEN, config.hidden_size, generator=generator)
        for _ in range(args.samples)
    ]

    modes: dict[str, list[list[float]]] = {}
    norms: dict[str, list[float]] = {}
    failures: dict[str, str] = {}

    for mode in MODES:
        try:
            variant = quantize(block, mode).to(device)
        except Exception as exc:
            failures[mode] = f"{type(exc).__name__}: {exc}"
            print(f"  {mode:5s} unavailable: {failures[mode]}")
            continue

        dtype = torch.bfloat16 if mode == "bf16" else (
            torch.float32 if mode == "fp32" else torch.float16
        )
        vectors, mode_norms = [], []
        for index, sample in enumerate(inputs):
            hidden = sample.to(device=device, dtype=dtype)
            output = run_block(variant, config, hidden, device)
            seed = derive_seed(index.to_bytes(16, "big"), f"calib:{BLOCK_INDEX}", 0)
            sketch = sketch_hidden_state(
                output.float().cpu().numpy()[0], seed=seed, spec=spec
            )
            vectors.append([float(v) for v in sketch.values])
            mode_norms.append(float(sketch.norm))

        modes[mode] = vectors
        norms[mode] = mode_norms
        print(f"  {mode:5s} ok ({len(vectors)} samples)")
        del variant
        torch.cuda.empty_cache()

    import transformers

    payload = {
        "schema": SCHEMA,
        "label": args.label or gpu_name,
        "gpu": gpu_name,
        "compute_capability": capability,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "model": MODEL,
        "block_index": BLOCK_INDEX,
        "seq_len": SEQ_LEN,
        "input_seed": INPUT_SEED,
        "sketch_components": spec.components,
        "modes": modes,
        "norms": norms,
        "unavailable": failures,
    }
    try:
        import bitsandbytes

        payload["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        payload["bitsandbytes"] = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out} ({size_kb:.0f} KB, {len(modes)} mode(s))")
    print("Collect one of these per GPU type, then run fit_tolerances.py over all of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
