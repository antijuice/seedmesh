# MoE spike: can this stack carry a trillion-parameter mixture-of-experts?

Measured 2026-08-01 against `transformers 5.14.1` and the Seedmesh-ported Petals. Configs
read from the Hub (a few KB each); no weights downloaded. Scripts in [`spike/moe/`](../spike/moe/).

**Short answer: yes, and the binding constraint is not what it looks like.** An MoE is cheap
on the network and expensive on disk — the opposite of what the relay budget punishes. The
work is bounded and has a working template. Two of the three blockers are already solved.

## 1. Mixtral — Petals' only MoE — is broken in the port

`probe_mixtral.py`:

```
[PASS] import WrappedMixtralBlock
[PASS] construct block            110,976 params, 4 experts
[FAIL] prefill forward            TypeError: cannot unpack non-iterable NoneType object
```

```
petals/models/mixtral/block.py:73   outputs = super().forward(...)
transformers/models/mixtral/modeling_mixtral.py:327   cos, sin = position_embeddings
```

Current transformers computes RoPE in the *model* and passes `position_embeddings` down into
each decoder layer. `WrappedMixtralBlock` passes `position_ids` but never constructs
`position_embeddings`, so it is `None`.

This is the same failure class the Llama block had, and `tools/port_petals.py` contains **no
mixtral references** — the port was written and validated for Llama only. The evidence is in
the timestamps: `mixtral/block.py` is untouched upstream code, while `config.py` and
`model.py` were rewritten by the codemod's generic import pass.

**The fix has a working template.** The ported Llama block solves exactly this at line 106,
with the reason stated at line 15: *"own a rotary embedding, because a server hosts a slice
of layers with no `LlamaModel`"*. Llama's block is 244 lines against Mixtral's 113; the delta
is rotary ownership, the attention-mask API (Mixtral still calls the deprecated
`_prepare_4d_causal_attention_mask`), and cache handling (Mixtral still does Bloom-style
cache reordering).

### Ported, and it exposed a second bug

`spike/moe/ported_mixtral_block.py`, applied by `tools/port_petals.py` as patch 9. All four
probe steps now pass. The port is small because three of the four things it needs are
architecture-independent and already tested for Llama, so they are imported rather than
copied: `build_causal_mask`, `_SingleLayerCache`, and the Petals cache translation. MoE
replaces the MLP, not attention, so the KV cache layout is unchanged.

Fixing the rotary bug surfaced a worse one underneath it. transformers 5.x dispatches
mixture-of-experts through `config._experts_implementation`, exactly as it dispatches
attention — and a hosted block has no model wrapper to set it. Measured on CPU/fp32:

| `_experts_implementation` | result |
| --- | --- |
| unset (`None`) | **NaN** |
| `batched_mm` | **NaN** |
| `grouped_mm` | **finite** |
| `deepgemm` | requires bfloat16 |
| `sonicmoe` | requires CUDA |

So without pinning it, every Mixtral server silently returns NaN — the failure is in
correctness, not just performance. `grouped_mm` is now the default.

> ### CORRECTION: Mixtral is not reliably finite, and I reported it as working
>
> "All four probe steps pass" was true of **one unseeded initialisation**. Rerunning the
> identical probe later reported FAIL with no code change. A seed sweep gives the real
> picture: **the MoE path returns NaN for roughly half of random weight inits** —
> 3/8 in the current probe, 4/8 at most sequence lengths in a wider sweep.
>
> What this is *not*: the shared causal mask, attention, or sequence length. **Llama is
> 8/8 finite at every length from 32 to 1024**, so the live swarm is unaffected and the
> shared helpers are sound. It is MoE-specific.
>
> What it might be: random-init conditioning rather than a port bug. Random expert weights
> are badly scaled in a way trained ones are not, and `grouped_mm` was itself selected
> because it was the only implementation producing finite values on a single config —
> which, given the above, was weak evidence. **Untested on trained weights**, which is the
> only thing that would settle it, and needs a real Mixtral checkout (~87 GB).
>
> `probe_mixtral.py` now sweeps eight seeds so this cannot report a false green again.

**This is the "heterogeneity is not fraud" trap in a new place**, and it is why
`ComputeProfile` gained an `experts` field. Two honest servers on different expert kernels
will disagree numerically for the same reason two servers on different attention kernels do.
Unpublished, that disagreement is read as cheating. The key stays
`quant/dtype/attn` for dense models so every existing calibrated tolerance is untouched, and
becomes `quant/dtype/attn/experts` only when a model actually uses experts.

**Not yet measured: MoE tolerances.** The calibration table covers dense models across five
GPU architectures. Nothing has been measured for expert kernels, so an MoE swarm cannot
verify until `tools/calibrate/` is run against one.

## 2. Wire cost scales with hidden size, not parameters

This is the good news, and it is what makes the whole idea viable. Activations flow between
*block ranges*; expert routing happens **inside** a block. An MoE therefore costs the same
per token as a dense model of the same hidden size — its parameter count is nearly free on
the network.

| model | hidden | B/token (bf16) | 160-token request | fits 128 KiB relay? |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 4096 | 8,192 | 1,280 KB | **no** (10x over) |
| DeepSeek-V3 | 7168 | 14,336 | 2,240 KB | **no** (17x over) |
| Kimi K2 | 7168 | 14,336 | 2,240 KB | **no** (17x over) |

**This makes the four-peer requirement a precondition, not an optimisation.** A relayed
volunteer is cut off after 128 KiB, which is 1.8% of a single modest request at this scale.
Any volunteer who cannot get a direct connection is useless for a real model — which is
exactly what [NAT-AND-RELAYS.md](NAT-AND-RELAYS.md) now covers, and why it is already fixed.

## 3. DeepSeek-V3 is reachable; Kimi K2 needs one line

```
deepseek_v3    registered      DeepseekV3DecoderLayer imports OK
kimi_k2        NOT registered
mixtral        registered      MixtralDecoderLayer imports OK
```

`deepseek_v3` is native in transformers 5.14.1 and its decoder layer imports cleanly, so a
`petals/models/deepseek_v3/` subpackage following the Llama template is bounded work — no
`trust_remote_code`, which matters because Petals imports the layer class directly and
`trust_remote_code` would mean every volunteer executes code from a model repo.

**Kimi K2 declares `model_type: kimi_k2` but `architectures: ["DeepseekV3ForCausalLM"]`** — it
*is* the DeepSeek-V3 implementation under a different type string. That is a registry mapping,
not an architecture gap.

Two structural differences from Llama, both with consequences for code we have already written:

**Multi-head latent attention.** `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`,
`qk_rope_head_dim=64`, `v_head_dim=128`.

> **Correction.** An earlier version of this section said Petals would *over*-reserve here,
> "since MLA's whole point is a far smaller cache". That reasoned from the MLA design rather
> than from the implementation in use, and it was backwards. `probe_mla_cache.py` shows
> transformers decompresses the latent back to full K/V *before* `Cache.update`:
> `key_states = torch.cat((k_pass, k_rot), dim=-1)`. So the stored cache is standard-shaped
> and, with 128 heads at a 192-wide qk head, very large — Petals **under**-reserves. Two
> useful consequences: the Llama cache translation transfers to DeepSeek-V3 unchanged, and
> the sizing bug is the dangerous direction rather than the wasteful one.

Measured per token per layer, bf16 (`probe_cache_size.py`):

| model | attention | Petals reserves | actually stores | |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | GQA | 4,096 B | 4,096 B | correct |
| DeepSeek-V3 | MLA | 28,672 B | 98,304 B | **3.4x under** |
| Kimi K2 | MLA | 28,672 B | 49,152 B | **1.7x under** |

Petals computes `2 * hidden_size * tokens // num_key_value_groups`, which is exactly right
whenever `kv_heads * head_dim == hidden_size // groups` — true for MHA and GQA. MLA breaks
the identity: `num_key_value_groups` is 1 there, while the real width is
`heads * (qk_nope + qk_rope)`. **Fixed as codemod patch 11**, branching on the MLA config
fields and leaving the GQA path byte-identical.

Under-reserving is the dangerous direction: the server accepts sessions it cannot finish and
OOMs mid-request, which the swarm reads as an unreliable peer rather than a misconfigured
one — reputation lost for a bug the volunteer did not cause.

**Blocks are not uniform.** `first_k_dense_replace=3`: the first three layers are dense, the
remaining 58 are MoE — a 19x size difference between layer types.

**Both of these were bugs in `seedmesh/cli/hardware.py`, now fixed.** `params_per_block`
counted a single MLP, so it sized a Mixtral block at 0.22B instead of 1.45B — **6.6x under**.
It now counts `num_local_experts` / `n_routed_experts` copies plus shared experts and the
router. `params_per_layer()` and `is_moe_layer()` handle the dense prefix, and `plan_blocks`
sizes against the **largest** layer rather than an average, because a volunteer does not
choose which block range they are assigned — a plan that fits only the dense prefix is a plan
that OOMs on assignment.

Correctness check: summing the analytic per-layer figures over DeepSeek-V3's 61 layers
(58 MoE at 11.53B + 3 dense at 0.60B) gives **670B** against the published **671B**. Dense
models are bit-identical to before — the Llama block still computes 139,520 params.

### Built: `petals/models/deepseek_v3/`

Codemod patches 13 and 14. `probe_deepseek.py` passes end to end: both `deepseek_v3` and
`kimi_k2` reach the registry, dense (layer 0) and MoE (layer 2) blocks construct and run
finite, the MLA cache round-trips exactly, and incremental decode grows the cache by one
position.

`kimi_k2` is registered as an alias subclass — it declares `model_type: kimi_k2` while its
architectures field says `DeepseekV3ForCausalLM`, and Petals dispatches on `model_type`, so
without the alias a model that is byte-for-byte a DeepSeek-V3 is refused. The registration
is wrapped in `try/except AssertionError` because a future transformers release may add
`kimi_k2` natively and registering a duplicate raises.

**The cache translation did NOT transfer unchanged**, contrary to what this document said an
hour earlier. Measured before writing the block (`probe_mla_shapes.py`), a tiny config gives:

```
key   (1, 4, 6, 24)     <- qk_nope + qk_rope
value (1, 4, 6, 16)     <- v_head_dim
```

K and V are stored at **different widths**. Llama's translation does
`value_states.view(*key_states.shape)` — correct for MHA and GQA, silently wrong here. It
would not raise; it would reshape one tensor into the other's dimensions and corrupt
attention, which in this swarm is indistinguishable from cheating. The DeepSeek block
therefore carries `key_dim` and `value_dim` separately, and the probe asserts the round-trip
is exact rather than merely shaped right.

That discovery also required a **second Petals-side patch**.
`TransformerBackend.get_inference_cache_descriptors` derives one `head_dim` from
`hidden_size // num_attention_heads` and allocates K and V at that same width — for
DeepSeek-V3, 56 against real widths of 192 and 128. Patch 12 branches on the MLA config
fields. Patch 11's byte formula was corrected at the same time: it had assumed both tensors
were qk-wide, over-reserving ~1.2x.

**Regression checked**: llama-160m still reports `Attention cache for all blocks will consume
up to 0.14 GiB`, byte-identical to before patches 11–14, and still loads all 12 blocks. The
GQA path is untouched.

### Attempted: MoE tolerance calibration — blocked, and why

Dense calibration cannot answer the MoE-specific question. A dense block turns small input
noise into small output noise, which is exactly what tolerance verification relies on. An MoE
block contains a discrete top-k over router scores: two honest servers whose logits differ
slightly can run **different experts**, and the output difference is then discontinuous — the
one shape a threshold cannot model.

This could not be measured here, for three reasons found in order:

1. **`MixtralSparseMoeBlock` in isolation returns NaN at every token count.** The
   `_experts_implementation` dispatch only takes effect through the wrapped block's
   constructor, so the submodule is not a valid subject and early numbers taken from it were
   meaningless.
2. **A single random gate measures nothing.** One run gave "20 flips in 65536", the next gave
   0 — purely because module construction consumed RNG differently between versions. Flip
   frequency depends on how often two experts are nearly *tied*, which is a property of the
   gate's weights.
3. **Random gates are the wrong subject entirely.** A trained router has structured score
   gaps; a random one has arbitrary ones. Whatever rate a random gate shows is an artifact of
   initialisation, not a prediction about Mixtral.

And underneath all three sits the NaN problem above: half the initialisations cannot produce
a number at all.

**What this needs:** real trained MoE weights. Then the measurement is straightforward —
perturb the input by a bf16 round-trip, count tokens whose top-k expert set changes, and
compare the output distance for flipped tokens against stable ones. If the ratio is near 1x
(the swapped experts were near-tied, so near-zero weighted) tolerance verification extends to
MoE with a recalibration. If it is large, **a wider tolerance is the wrong fix** — one wide
enough to absorb an expert swap detects nothing — and verification would have to compare
router decisions separately from expert outputs, so a routing difference is reported as a
profile mismatch rather than as cheating.

Until that is measured, **an MoE swarm can serve but cannot verify**, and that is a stronger
statement than it sounds: turning verification off is not a neutral default for a project
whose entire pitch is trust.

## 4. Storage — how many volunteers a trillion parameters needs

| model | params | @ NF4 | avg/layer | MoE layer |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 47B | 22 GB | 0.7 GB | ~0.7 GB |
| DeepSeek-V3 | 671B | 322 GB | 5.3 GB | ~5.6 GB |
| Kimi K2 | 1000B | 480 GB | 7.9 GB | **~8.3 GB** |

**Correction to my own arithmetic.** The first pass reported "61 hosts with 8 GB each" for
Kimi K2, dividing total size by layer count. Finding 3 invalidates that: with only 3 of 61
layers dense, essentially all the weight sits in the 58 MoE layers, so a real MoE block is
**~8.3 GB, not 7.9** — it does *not* fit an 8 GB card. **12 GB is the realistic floor** for
hosting one Kimi K2 block at 4-bit, and Petals cannot split a block finer without changes.

So the honest figure: **~61 volunteers with 12 GB cards** covers Kimi K2 once. Real swarms
want each block range covered 2–3x, so **120–180 volunteers** for a swarm that survives
churn. That is a community, not a data centre — which is the point.

## What this queues up

| | work | status |
| --- | --- | --- |
| 1 | Port `WrappedMixtralBlock` | **done** — patch 9, all probe steps pass |
| 1b | Calibrate MoE tolerances | **unblocked 2026-08-02** — answered on a 70 MB trained MoE; see below |
| 2 | Add `petals/models/deepseek_v3/` | **done** — patch 13, probe passes |
| 3 | Map `kimi_k2` → `deepseek_v3` | **done** — patch 14 |
| 4 | Fix `attn_cache_tokens` for MLA | **done** — patch 11; it under-reserved, not over |
| 5 | Make `plan_blocks` per-layer, not uniform | **done** — and experts are now counted at all |
| 6 | Direct connections for volunteers | **already done** — four-peer requirement |

Nothing here needs a new transport, a fork, or a protocol change. The hard networking problem
was the relay budget, and that is solved.

## Both open questions, answered with trained weights (2026-08-02)

They were blocked on the same thing, and `routing_sensitivity.py` said so in its own
docstring: a random gate cannot answer either, because routing flips depend on how often two
experts are nearly **tied**, which is a property of training.

Mixtral-8x7B is ~87 GB. **`ggml-org/stories15M_MOE` is 70 MB**, declares
`model_type: mixtral`, and is genuinely trained. Both questions became laptop-sized.
`spike/moe/trained_weights.py`.

### Q1: the ~50% NaN rate is a random-init artifact, not a port bug

**36/36 finite** across 12 seeds x 3 sequence lengths, plus real token embeddings. So the
earlier finding stands as a warning about *probes*, not about the port: untrained expert
weights are badly conditioned in a way trained ones are not.

A bonus regression check fell out of it. Loading the real layer's `state_dict` into
`WrappedMixtralBlock` succeeded with **no missing trained parameters and no unexpected
ones** — which is exactly the property that lets Petals load real MoE checkpoints, and it had
never been tested against a real one.

### Q2: honest noise rarely flips routing — but "rarely" is not "never"

| | |
| --- | --- |
| bf16 round-trip | 1.48e-3 relative input noise |
| routing flips observed at bf16 | **0 / 256 tokens** |
| routing first flips under swept noise | ~1e-2, about **7x** a bf16 round-trip |
| decision margin (top-k vs first rejected) | min 1.35e-4, p5 5.1e-3, median 4.1e-2 |
| tokens with margin *below* the bf16 noise floor | **8 / 256 (3.1%)** |

The last two rows are the ones that matter, and they corrected a conclusion this file nearly
recorded. "Zero flips observed" invites "honest noise never changes routing" — but the
**minimum margin is smaller than the bf16 noise floor**. No flips occurred because a random
perturbation direction rarely moves a score gap, not because it cannot. The observed rate (0%)
is a lower bound and the margin-based rate (3.1%) an upper one; the truth sits between and
depends on direction.

### What that means for verification

The median token sits 28x the noise floor from the boundary, so MoE verification is not
fundamentally broken — but a small tail means **two honest servers will occasionally route a
token differently**.

The right response is not a wider distance tolerance. One wide enough to absorb an expert
swap would detect nothing at all. It is to treat a routing difference as **INCONCLUSIVE** —
the verdict the verification layer already has for "the test could not decide" — rather than
as MISMATCH.

**Still open, and now sharper:** a client sees only output hidden states, so it cannot
currently *distinguish* an expert swap from a genuine fault. Until it can, a flipped token
looks like a mismatch. That is a verification-layer design question, not a porting one.

**Caveat.** A 15M router trained on children's stories is not a 47B router trained on the
open internet, and flip frequency is precisely the quantity that depends on the router. These
numbers bound the mechanism; a real MoE swarm still needs tolerances measured on the model it
serves.
