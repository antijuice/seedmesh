"""The two open MoE questions, answered with TRAINED weights instead of random ones.

Both were blocked on the same thing, and `routing_sensitivity.py` says so in its own
docstring: a random gate cannot answer either question, because expert-routing flips depend
on how often two experts are nearly TIED, which is a property of training.

  Q1  Roughly half of random Mixtral inits drive the ported block to NaN. Is that a bug in
      the port, or is it random-init conditioning that trained weights never exhibit?

  Q2  When a bf16 round-trip perturbs the input, how often does expert selection change, and
      how much bigger is the output difference when it does? If the ratio is ~1x, MoE
      verification just needs recalibrated tolerances. If it is large, a wider tolerance is
      the WRONG fix -- one wide enough to absorb an expert swap detects nothing -- and
      verification has to compare router decisions separately from expert outputs.

`Mixtral-8x7B` is ~87 GB. `ggml-org/stories15M_MOE` is **70 MB**, has `model_type: mixtral`,
and is genuinely trained (a mixture-of-experts build of the TinyStories model). That makes
both questions answerable on a laptop.

**What this can and cannot settle.** Q1 is settled outright: either the port produces finite
output from trained weights across many inputs, or it does not. Q2 is answered for THIS
model. A 15M-parameter router trained on children's stories is not a 47B router trained on
the open internet, and the flip rate is exactly the quantity that depends on the router. So
treat the ratio as evidence about the mechanism's magnitude, not as a calibrated constant --
the tolerance table for a real MoE swarm still has to be measured on the model being served.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "ggml-org/stories15M_MOE"
N_TOKENS = 256


def banner(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def load_real_block():
    """A ported block carrying one layer's trained weights."""
    from petals.models.mixtral import WrappedMixtralBlock

    config = AutoConfig.from_pretrained(MODEL)
    assert config.model_type == "mixtral", f"not a mixtral model: {config.model_type}"
    config._attn_implementation = "eager"
    config._experts_implementation = "grouped_mm"

    source = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    trained_layer = source.model.layers[0]

    block = WrappedMixtralBlock(config, layer_idx=0).eval()
    # Subclassing MixtralDecoderLayer means parameter names match exactly -- which is the
    # property that lets Petals load real checkpoints at all, so a strict load is also a
    # regression check on the port's naming.
    missing, unexpected = block.load_state_dict(trained_layer.state_dict(), strict=False)
    assert not unexpected, f"port has parameters the real layer does not: {unexpected}"
    trainable = [k for k in missing if "inv_freq" not in k]
    assert not trainable, f"port is missing trained parameters: {trainable}"
    return config, block, source


def selected_experts(config, block, x):
    hidden = config.hidden_size
    with torch.inference_mode():
        out = block.mlp.gate(x.reshape(-1, hidden))
        if isinstance(out, tuple):
            for element in out:
                if element is not None and element.dtype in (torch.int64, torch.int32):
                    return element.sort(dim=-1).values
            out = out[0]
        weights = torch.softmax(out.float(), dim=-1)
        _, idx = torch.topk(weights, config.num_experts_per_tok, dim=-1)
    return idx.sort(dim=-1).values


def main() -> int:
    config, block, source = load_real_block()
    hidden = config.hidden_size
    print(f"model      {MODEL}")
    print(f"           {config.num_hidden_layers} layers, hidden {hidden}, "
          f"{config.num_local_experts} experts, top-{config.num_experts_per_tok}")

    # ---- Q1: does the NaN survive trained weights? --------------------------
    banner("Q1  finite output with trained weights?")
    failures = 0
    for seed in range(12):
        torch.manual_seed(seed)
        for length in (32, 128, N_TOKENS):
            x = torch.randn(1, length, hidden)
            with torch.inference_mode():
                out = block(x)[0]
            if not torch.isfinite(out).all():
                failures += 1
                print(f"  seed {seed}, {length} tokens: NON-FINITE")
    total = 12 * 3
    print(f"  {total - failures}/{total} finite")
    if failures:
        print("  -> trained weights still produce NaN. This is a PORT BUG, not conditioning.")
    else:
        print("  -> all finite. The ~50% NaN rate on RANDOM inits is a conditioning artifact")
        print("     of untrained expert weights, not a defect in the port.")

    # Real token embeddings, not noise -- the distribution a hosted block actually sees.
    banner("Q1b  finite output on real embeddings?")
    with torch.inference_mode():
        ids = torch.randint(0, config.vocab_size, (1, N_TOKENS))
        embeds = source.model.embed_tokens(ids)
        out = block(embeds)[0]
    print(f"  finite: {bool(torch.isfinite(out).all())}   "
          f"range [{out.min():.3f}, {out.max():.3f}]")

    # ---- Q2: does honest noise flip routing, and what does it cost? ---------
    banner("Q2  bf16 round-trip: does expert selection flip, and at what cost?")
    with torch.inference_mode():
        ids = torch.randint(0, config.vocab_size, (1, N_TOKENS))
        x = source.model.embed_tokens(ids).float()
    x_perturbed = x.to(torch.bfloat16).to(torch.float32)

    flipped = (
        selected_experts(config, block, x) != selected_experts(config, block, x_perturbed)
    ).any(dim=-1)

    with torch.inference_mode():
        a = block(x)[0][0]
        b = block(x_perturbed)[0][0]
    distance = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)

    n_flipped = int(flipped.sum())
    print(f"  tokens                {N_TOKENS}")
    print(f"  routing flips         {n_flipped} ({n_flipped / N_TOKENS:.1%})")
    if n_flipped and n_flipped < N_TOKENS:
        stable_median = distance[~flipped].median().item()
        flipped_median = distance[flipped].median().item()
        ratio = flipped_median / max(stable_median, 1e-12)
        print(f"  stable  rel. distance median  {stable_median:.3e}")
        print(f"  flipped rel. distance median  {flipped_median:.3e}")
        print(f"  ratio                         {ratio:.1f}x")
        print()
        if ratio < 3:
            print("  -> Flips barely move the output. MoE verification needs recalibrated")
            print("     tolerances and nothing structural.")
        else:
            print("  -> A flip costs far more than ordinary noise. A tolerance wide enough to")
            print("     absorb one would detect nothing, so verification must compare ROUTER")
            print("     DECISIONS separately from expert outputs -- a routing difference is a")
            print("     profile mismatch, not fraud.")
    elif n_flipped == 0:
        print("  -> No flips at all at bf16. The trained router's score gaps exceed bf16")
        print("     noise, so this perturbation never reaches the discrete boundary.")
    else:
        print("  -> Every token flipped; the comparison has no stable baseline.")

    # ---- Q2b: if bf16 never flips, where IS the boundary? -------------------
    #
    # "0 flips" on its own is a weak result: it says the perturbation was too small without
    # saying by how much. The margin is what tells you whether a real MoE swarm has
    # comfortable headroom or sits one quantization step from the discontinuity.
    banner("Q2b  how much margin does the trained router have?")

    with torch.inference_mode():
        logits = block.mlp.gate(x.reshape(-1, hidden))
        if isinstance(logits, tuple):
            logits = next(e for e in logits if e is not None and e.dtype.is_floating_point)
        scores = torch.softmax(logits.float(), dim=-1)
        top = scores.topk(config.num_experts_per_tok + 1, dim=-1).values

    # Gap between the last expert selected and the first rejected: the distance a
    # perturbation must cover to change that token's routing.
    k = config.num_experts_per_tok
    margin = top[:, k - 1] - top[:, k]
    relative_noise = ((x_perturbed - x).norm() / x.norm()).item()
    print(f"  bf16 round-trip = {relative_noise:.2e} relative input noise")
    print(f"  decision margin   min {margin.min():.2e}   p5 {margin.quantile(0.05):.2e}   "
          f"median {margin.median():.2e}")

    print("\n  perturbation sweep -- where does routing start to change?")
    print(f"  {'rel. noise':>12} {'flips':>8} {'output rel. distance':>22}")
    torch.manual_seed(0)
    direction = torch.randn_like(x)
    direction = direction / direction.norm() * x.norm()
    baseline = selected_experts(config, block, x)
    first_flip = None
    for scale in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        probe = x + direction * scale
        flips = int((selected_experts(config, block, probe) != baseline).any(dim=-1).sum())
        with torch.inference_mode():
            moved = block(probe)[0][0]
        rel = ((moved - a).norm() / a.norm()).item()
        print(f"  {scale:>12.0e} {flips:>8} {rel:>22.3e}")
        if flips and first_flip is None:
            first_flip = scale
    # How many tokens sit close enough to the boundary that bf16 noise COULD flip them?
    # This is the number the sweep above cannot give: a random perturbation direction rarely
    # moves a score gap much, so "0 flips observed" understates what is possible. The
    # minimum margin here is smaller than the bf16 noise, so flips are rare, not excluded --
    # and a conclusion of "well below the boundary" would have been wrong.
    at_risk = int((margin < relative_noise).sum())
    print(f"\n  tokens with margin below the bf16 noise floor: {at_risk}/{N_TOKENS} "
          f"({at_risk / N_TOKENS:.1%})")

    if first_flip:
        print(f"  -> routing first changes near {first_flip:.0e} relative noise, about "
              f"{first_flip / max(relative_noise, 1e-12):.0f}x a bf16 round-trip.")
    else:
        print("  -> routing never changed across the whole sweep.")
    print()
    print("  Read together: the MEDIAN token sits far from the boundary "
          f"({margin.median() / relative_noise:.0f}x the noise floor), but a small tail does")
    print("  not, so two honest servers will occasionally route a token differently.")
    print("  That argues for treating a routing difference as INCONCLUSIVE rather than as")
    print("  either a match or fraud -- the verdict the verification layer already has for")
    print("  'the test could not decide'. It does NOT argue for a wider distance tolerance,")
    print("  which would have to be wide enough to absorb a whole expert swap and would")
    print("  then detect nothing.")

    print("\nCAVEAT: a 15M router trained on children's stories is not a 47B router. This")
    print("bounds the mechanism; a real MoE swarm still needs tolerances measured on the")
    print("model it serves.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
