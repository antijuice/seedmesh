"""Is DeepSeek-V3 (and therefore Kimi K2) now hostable?

The same shape of probe as probe_mixtral.py, plus the two things unique to this
architecture: that both model types reach the registry, and that the MLA cache round-trips.

The cache check is the one that matters. K and V are stored at different widths here
(qk_nope + qk_rope vs v_head_dim), and a translation that mixes them up would not raise --
it would reshape one tensor into the other's dimensions and quietly corrupt attention. In a
swarm that judges servers by whether their outputs agree, silent corruption is
indistinguishable from cheating, so it is checked numerically rather than by shape alone.
"""

import sys
import traceback

import torch


def report(step, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {step}" + (f"\n         {detail}" if detail else ""),
          flush=True)
    return ok


print("=== 1. registry ===")
try:
    import petals  # noqa: F401  populates the registry
    from petals.utils.auto_config import _CLASS_MAPPING

    for name in ("llama", "mixtral", "deepseek_v3", "kimi_k2"):
        report(f"{name} registered", name in _CLASS_MAPPING)
    if "deepseek_v3" not in _CLASS_MAPPING:
        sys.exit(1)
except Exception as exc:
    report("import petals", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 2. construct a block ===")
from transformers import DeepseekV3Config  # noqa: E402

from petals.models.deepseek_v3 import WrappedDeepseekV3Block  # noqa: E402

config = DeepseekV3Config(
    hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    # DeepSeek routes experts in GROUPS: the gate does
    # `scores.view(-1, n_group, n_routed_experts // n_group)`, so n_group must divide the
    # expert count. The default n_group=8 against 4 experts gives a 0-sized dim and a
    # RuntimeError -- a property of the tiny test config, not of the port.
    n_group=2, topk_group=1,
    first_k_dense_replace=1, q_lora_rank=64, kv_lora_rank=32,
    qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    max_position_embeddings=64, vocab_size=128,
)
try:
    # layer 0 is dense (first_k_dense_replace=1), layer 2 is MoE -- both must work.
    dense_block = WrappedDeepseekV3Block(config, layer_idx=0).eval()
    moe_block = WrappedDeepseekV3Block(config, layer_idx=2).eval()
    report("construct dense (layer 0) and MoE (layer 2) blocks", True,
           f"dense {sum(p.numel() for p in dense_block.parameters()):,} params, "
           f"MoE {sum(p.numel() for p in moe_block.parameters()):,} params")
except Exception as exc:
    report("construct blocks", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 3. forward pass ===")
torch.manual_seed(0)
hidden = torch.randn(1, 6, config.hidden_size)
ok = True
for label, block in (("dense layer", dense_block), ("MoE layer", moe_block)):
    try:
        with torch.inference_mode():
            out = block(hidden)
        tensor = out[0]
        finite = bool(torch.isfinite(tensor).all())
        ok &= report(f"{label} forward", finite,
                     f"output {tuple(tensor.shape)}, finite={finite}")
    except Exception as exc:
        ok = False
        report(f"{label} forward", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
if not ok:
    sys.exit(1)

print("\n=== 4. MLA cache round-trip (the one that would corrupt silently) ===")
try:
    with torch.inference_mode():
        _, present = moe_block(hidden, use_cache=True)
    keys, values = present
    print(f"         petals-format key {tuple(keys.shape)}, value {tuple(values.shape)}")

    # Round-trip through the translation and require exact recovery.
    back_k, back_v = moe_block._cache_from_petals(present, batch_size=1)
    again_k, again_v = moe_block._cache_to_petals(back_k, back_v, batch_size=1)
    exact = torch.equal(keys, again_k) and torch.equal(values, again_v)
    report("translation is its own inverse", exact,
           f"unpacked key {tuple(back_k.shape)}, value {tuple(back_v.shape)}")

    expected_k = config.qk_nope_head_dim + config.qk_rope_head_dim
    widths_differ = back_k.shape[-1] == expected_k and back_v.shape[-1] == config.v_head_dim
    report("K and V keep their own widths", widths_differ,
           f"K={back_k.shape[-1]} (expected {expected_k}), "
           f"V={back_v.shape[-1]} (expected {config.v_head_dim})")
    if not (exact and widths_differ):
        sys.exit(1)
except Exception as exc:
    report("cache round-trip", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 5. incremental decode with that cache ===")
try:
    with torch.inference_mode():
        step_in = torch.randn(1, 1, config.hidden_size)
        out2 = moe_block(step_in, layer_past=present, use_cache=True)
    tensor2, present2 = out2
    grew = present2[0].shape[-1] == present[0].shape[-1] + 1
    report("cache grows by one position", grew and bool(torch.isfinite(tensor2).all()),
           f"seq {present[0].shape[-1]} -> {present2[0].shape[-1]}")
except Exception as exc:
    report("incremental decode", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\nDeepSeek-V3 blocks construct, run, and round-trip their MLA cache.")
