"""Does honest numerical noise flip MoE expert selection -- and how much does that cost?

The question dense calibration cannot answer. A dense block turns small input noise into
small output noise, which is exactly what tolerance-based verification depends on. An MoE
block contains a discrete top-k over router scores: two honest servers whose logits differ
slightly can run DIFFERENT experts, and the output difference is then discontinuous.

Two false starts are worth recording, because both produced numbers that looked fine:

  1. Measuring `MixtralSparseMoeBlock` directly returns NaN at every token count. The
     `_experts_implementation` dispatch only takes effect through the wrapped block's
     constructor, so the submodule in isolation is not a valid subject. Everything here
     therefore goes through `WrappedMixtralBlock`.

  2. A single random gate gave "20 flips in 65536" on one run and "0" on the next, purely
     because module construction consumed RNG differently. Flip frequency is a property of
     how often two experts are nearly TIED, which depends entirely on the gate's weights --
     so one initialisation measures nothing. This sweeps seeds and reports the spread.

**The caveat that matters most:** these are randomly initialised gates. A trained router has
structured score gaps; a random one does not. These numbers bound the *mechanism*, not the
rate a real Mixtral would show. Settling that needs trained weights -- see docs/findings-moe.md.
"""

import torch
from transformers import MixtralConfig

from petals.models.mixtral import WrappedMixtralBlock

HIDDEN = 128
N_TOKENS = 256  # attention is quadratic; this is what a full block can afford
SEEDS = range(12)

config = MixtralConfig(
    hidden_size=HIDDEN, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4,
    num_local_experts=8, num_experts_per_tok=2,
    max_position_embeddings=512, vocab_size=1000,
)
config._attn_implementation = "eager"
config._experts_implementation = "grouped_mm"


def selected_experts(block, x):
    with torch.inference_mode():
        out = block.mlp.gate(x.reshape(-1, HIDDEN))
        if isinstance(out, tuple):
            for element in out:
                if element is not None and element.dtype in (torch.int64, torch.int32):
                    return element.sort(dim=-1).values
            out = out[0]
        weights = torch.softmax(out.float(), dim=-1)
        _, idx = torch.topk(weights, config.num_experts_per_tok, dim=-1)
    return idx.sort(dim=-1).values


print(f"{N_TOKENS} tokens per seed, {len(SEEDS)} seeds, perturbation = bf16 round-trip")
print(f"{'seed':>5} {'flips':>7} {'stable median':>15} {'flipped median':>16} {'ratio':>8}")
print("-" * 56)

total_flips = 0
ratios = []
for seed in SEEDS:
    torch.manual_seed(seed)
    block = WrappedMixtralBlock(config, layer_idx=0).eval()
    x = torch.randn(1, N_TOKENS, HIDDEN)
    x_pert = x.to(torch.bfloat16).to(torch.float32)

    flipped = (selected_experts(block, x) != selected_experts(block, x_pert)).any(dim=-1)
    with torch.inference_mode():
        a = block(x)[0][0]
        b = block(x_pert)[0][0]
    if not torch.isfinite(a).all():
        print(f"{seed:>5}  non-finite output -- skipped")
        continue

    per_token = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)
    n_flip = int(flipped.sum())
    total_flips += n_flip
    if n_flip == 0 or n_flip == len(per_token):
        print(f"{seed:>5} {n_flip:>7} {per_token.median():>15.2e} {'-':>16} {'-':>8}")
        continue
    stable = per_token[~flipped].median()
    flip = per_token[flipped].median()
    ratio = (flip / stable).item()
    ratios.append(ratio)
    print(f"{seed:>5} {n_flip:>7} {stable:>15.2e} {flip:>16.2e} {ratio:>7.1f}x")

print("-" * 56)
rate = total_flips / (len(SEEDS) * N_TOKENS)
print(f"flip rate across all seeds: {total_flips}/{len(SEEDS) * N_TOKENS} = {rate * 100:.3f}%")
if ratios:
    ratios.sort()
    print(f"cost ratio (flipped/stable): min {ratios[0]:.1f}x  "
          f"median {ratios[len(ratios) // 2]:.1f}x  max {ratios[-1]:.1f}x  (n={len(ratios)})")
else:
    print("no seed produced a mix of flipped and stable tokens -- cost ratio unmeasured")

print("""
=== how to read this ===
  ratio near 1x  -> the swapped experts were near-tied and therefore near-zero weighted, so
                    a flip barely moves the output. Tolerance verification extends to MoE
                    with a recalibration and nothing more.
  ratio large    -> an honest server that flips even rarely emits an output a
                    dense-calibrated threshold calls fraud. The fix is NOT a wider tolerance
                    (one wide enough to absorb an expert swap detects nothing) but comparing
                    router decisions separately from expert outputs.

  Either way these are RANDOM gates. The rate on a trained model is a different number.
""")
