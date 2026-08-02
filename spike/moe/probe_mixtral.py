"""Does the Seedmesh port actually support Mixtral, or only Llama?

Petals registers four architectures, and `mixtral` is the only MoE among them -- so it is the
whole basis for any claim that this stack could carry a large mixture-of-experts model. But
the port was written and validated against Llama: `tools/port_petals.py` contains no mixtral
references, and `models/mixtral/block.py` is untouched upstream code while `config.py` and
`model.py` were rewritten by the codemod's generic import pass.

The Llama block needed 138 lines replacing 221 to work on current transformers. Mixtral's
block has had none of that. This checks, in increasing order of severity:

  1. does the module import at all?
  2. can a block be instantiated from a tiny config (no weights, no download)?
  3. does a forward pass produce finite numbers?
  4. does the inference path (incremental, with a cache) work?

A tiny random config is used deliberately -- Mixtral-8x7B is 46.7B parameters, and nothing
here needs real weights. Shape and API compatibility is what is in question.
"""

import sys
import traceback

import torch


def report(step, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {step}" + (f"\n         {detail}" if detail else ""), flush=True)
    return ok


print("=== 1. import ===")
try:
    from transformers import MixtralConfig

    from petals.models.mixtral import WrappedMixtralBlock
    report("import WrappedMixtralBlock", True)
except Exception as exc:
    report("import WrappedMixtralBlock", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 2. instantiate from a tiny config ===")
# Mixtral's shape knobs, shrunk. num_local_experts/num_experts_per_tok are the MoE-specific
# ones; everything else is standard decoder geometry.
config = MixtralConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_local_experts=4,
    num_experts_per_tok=2,
    max_position_embeddings=128,
    vocab_size=256,
)
try:
    block = WrappedMixtralBlock(config, layer_idx=0)
    block.eval()
    n_params = sum(p.numel() for p in block.parameters())
    report("construct block", True, f"{n_params:,} params, {config.num_local_experts} experts")
except Exception as exc:
    report("construct block", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 3. forward pass across SEEDS (one config is not evidence) ===")
# The first version of this probe tested a single unseeded initialisation and reported PASS.
# Rerunning it later reported FAIL with no code change -- because roughly half of random
# weight inits drive the MoE path to NaN. Testing one config proves one config.
finite_count = 0
for seed in range(8):
    torch.manual_seed(seed)
    seeded = WrappedMixtralBlock(config, layer_idx=0).eval()
    with torch.inference_mode():
        probe_out = seeded(torch.randn(1, 128, config.hidden_size))[0]
    finite_count += bool(torch.isfinite(probe_out).all())

report(
    f"finite output on {finite_count}/8 random initialisations",
    finite_count == 8,
    "" if finite_count == 8 else
    "KNOWN AND UNRESOLVED: the MoE path returns NaN for about half of random weight inits. "
    "Llama is 8/8 at every sequence length, so this is MoE-specific, not the shared causal "
    "mask. May be random-init conditioning rather than a port bug -- untested on trained "
    "weights. See docs/findings-moe.md.",
)

print("\n=== 3b. single-config forward (kept for the shape check) ===")
torch.manual_seed(0)
block = WrappedMixtralBlock(config, layer_idx=0).eval()
batch, seq = 1, 8
hidden = torch.randn(batch, seq, config.hidden_size)
try:
    with torch.inference_mode():
        out = block(hidden)
    tensor = out[0] if isinstance(out, tuple) else out
    finite = bool(torch.isfinite(tensor).all())
    report("prefill forward", finite,
           f"output {tuple(tensor.shape)}, all finite: {finite}")
    if not finite:
        sys.exit(1)
except Exception as exc:
    report("prefill forward", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== 4. incremental step with a cache (the inference path) ===")
try:
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    with torch.inference_mode():
        out = block(hidden, use_cache=True, past_key_value=cache)
        step_in = torch.randn(batch, 1, config.hidden_size)
        out2 = block(step_in, use_cache=True, past_key_value=cache)
    tensor2 = out2[0] if isinstance(out2, tuple) else out2
    report("incremental step", bool(torch.isfinite(tensor2).all()),
           f"output {tuple(tensor2.shape)}")
except Exception as exc:
    report("incremental step", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

print("\n=== what this means ===")
print("  All pass  -> mixtral is usable and the MoE path is worth measuring further.")
print("  Any fail  -> the port covers llama only; mixtral needs the same treatment the")
print("               llama block got (138 lines replacing 221), before any MoE claim.")
