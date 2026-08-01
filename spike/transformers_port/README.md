# Spike: porting Petals' Llama block to modern transformers

**Question:** how expensive is it to make Petals' block wrappers work against current
transformers, and is that the right backend to build on at all?

**Answer:** cheaper than the audit predicted, and the reason is that most of the code
should be deleted rather than ported. A working, numerically-exact Llama block took **138
lines** against transformers 5.14.1, replacing 221 lines of upstream code. But the port is
only the visible part of the cost — see "What this does not measure" before treating it as
a decision.

Measured 2026-07-31 with torch 2.13.0+cpu, transformers 5.14.1, Python 3.13.2.

---

## Reproducing

```bash
python -m venv .venv-spike
.venv-spike/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-spike/Scripts/python.exe -m pip install "transformers==5.14.1" accelerate
git clone --depth 1 https://github.com/bigscience-workshop/petals.git /tmp/petals

.venv-spike/Scripts/python.exe spike/transformers_port/probe_api.py
.venv-spike/Scripts/python.exe spike/transformers_port/test_original_imports.py /tmp/petals
.venv-spike/Scripts/python.exe spike/transformers_port/test_equivalence.py
```

No GPU, no model weights, no downloads beyond the packages.

## Finding 1 — the audit was wrong about *how* it breaks

`docs/findings-upstream-audit.md` originally claimed `_prepare_4d_causal_attention_mask`
and the other imports were removed by the transformers 4.48 attention refactor. **That was
incorrect and has been corrected.** Every one of the ten module-scope imports in
`petals/models/llama/block.py` still resolves on transformers 5.14.1 — HuggingFace kept
back-compatible shims (`_prepare_4d_causal_attention_mask` now emits a `FutureWarning`
pointing at `transformers.masking_utils`, but it works).

The breakage is **attribute- and signature-level**, which is a worse failure mode than a
missing import, because the module loads cleanly and a naive "does it import?" check says
yes. Loading Petals' unmodified block against 5.14.1:

```
module import:              OK
construct WrappedLlamaBlock: OK
first forward pass:          AttributeError: 'OptimizedLlamaAttention' object
                             has no attribute 'num_heads'
```

What actually changed on `LlamaAttention`:

| Petals uses | Status in 5.14.1 |
| --- | --- |
| `self.num_heads` | **gone** |
| `self.hidden_size` | **gone** |
| `self.num_key_value_heads` | **gone** |
| `self.rotary_emb` | **gone** — moved up to `LlamaModel` in 4.48 |
| `self.head_dim`, `self.num_key_value_groups` | present |
| `forward(..., position_ids=...)` | **gone** — now takes `position_embeddings` |
| `past_key_value=` (tuple) | **renamed** to `past_key_values=`, must be a `Cache` |
| `LlamaDecoderLayer.forward` → tuple | **now returns a bare tensor** |

That last one is the quiet one. Code doing `outputs[0]` on a returned tensor gets the first
*row* rather than the hidden states, with no error.

## Finding 2 — the fix is deletion, not translation

Petals' 300-line Llama file is mostly `OptimizedLlamaAttention`: a hand-copied fork of
transformers' attention body, carried so two CUDA-graph optimizations could be spliced in
(graphed rotary application, graphed RMSNorm — both firing only for single-token decode on
CUDA). That copied body is exactly what tracks transformers' internals, so it is exactly
what broke.

A block-hosting server does not need it. It needs four things, none requiring a fork:

1. own a rotary embedding (no `LlamaModel` around it to supply position embeddings);
2. build its own causal mask (no model-level mask plumbing either);
3. translate between Petals' flattened cache layout and transformers' `Cache`;
4. keep parameter names identical so checkpoints still load.

Everything else delegates to stock `LlamaDecoderLayer`, which HuggingFace maintains.

| | total lines | code lines |
| --- | --- | --- |
| upstream `llama/block.py` | 300 | 221 |
| **ported equivalent** | 244 | **138** |

## Finding 3 — Petals already agreed

This is the strongest signal in the spike. Comparing all four architectures upstream:

| architecture | code lines | strategy |
| --- | --- | --- |
| `bloom` | 34 | delegates to stock `BloomBlock` |
| `mixtral` | 92 | delegates to stock `MixtralDecoderLayer` + `DynamicCache` |
| `llama` | 221 | **hand-forked attention** |
| `falcon` | 361 | **hand-forked attention** |

Mixtral is the most recently added architecture, and it already uses the delegate-to-stock
pattern with the modern `Cache` API. So the port strategy above is not an invention — it is
the pattern Petals itself converged on before development stopped. The forks are the legacy.

## Finding 4 — it is numerically exact

"It runs" is not a port. A subtly wrong block produces plausible text and would be
**invisible to Seedmesh's own verification layer**, because redundant execution checks
*agreement*, not correctness — if every server runs the same wrong port, they all agree.

```
single forward pass vs stock LlamaDecoderLayer   max abs diff  0.000e+00   PASS
incremental decode (4 tokens + 1 cached)         max abs diff  2.384e-07   PASS
cache layout roundtrip                           max abs diff  0.000e+00   PASS
causal mask structure (prefill/decode/padding)                             PASS
state_dict compatibility                         0 missing, 0 unexpected
```

`2.384e-07` is float32 epsilon — the cached path is arithmetically identical, just
differently ordered. The incremental test is the one that matters: a transposed cache
reshape passes a single forward pass and fails from the second token onward.

## Finding 5 — attention dispatch must be pinned explicitly

Hosting a bare layer leaves `config._attn_implementation` unset, and transformers warns and
falls back. For Seedmesh this is not cosmetic: **the kernel choice determines the numerical
noise floor**, and the verification thresholds are calibrated against that floor. Two
servers silently choosing different attention kernels would widen honest disagreement and
push comparisons into the inconclusive band.

The backend adapter must pin `_attn_implementation` and **publish it as part of the block
announcement**, so clients never compare servers running different kernels. This is a real
protocol requirement that the spec did not anticipate.

## What this does not measure

The honest boundary of this spike. It ports **one block wrapper** and proves it numerically
equivalent. That is roughly 5% of the work of a Petals backend. Not covered:

- **falcon** (361 code lines, the same fork problem, and a more idiosyncratic attention).
- **The rest of Petals**: server, client, DHT integration, `RemoteSequential`, routing,
  `tensor_parallel==1.0.23` (unmaintained), `bitsandbytes==0.41.1` (September 2023 — the
  quantization path is three years stale and is a separate port).
- **`hivemind` at a pinned SHA**, with its own compatibility surface.
- **Real hardware.** CPU float32 only. The quantized int8/int4 paths that make the memory
  math work are untested here, and they are where the remaining risk concentrates.

So: the *architecture* port is tractable and now has a proven recipe. The *dependency
stack* around it is still two-to-three years stale, and `bitsandbytes` is the piece most
likely to be genuinely painful.

## Recommendation

**The Petals block-wrapper port is no longer a reason to avoid Petals.** It is a solved,
recipe-shaped problem: delete the forked attention, delegate to stock layers, own the mask
and rotary, translate the cache, verify numerically. One day per architecture, with a test
harness that already exists in this directory.

**But do not decide on this evidence alone.** Nothing here touched quantization, and
`bitsandbytes==0.41.1` plus `tensor_parallel==1.0.23` are the stale dependencies that
actually gate a working server. The next spike should be quantization, not another
architecture — it is where the unknown risk now sits.

**And this does not settle Petals vs llama.cpp.** That choice should turn on hardware reach,
not port cost, now that port cost is bounded. Most volunteers will not have an NVIDIA GPU,
and llama.cpp reaches Apple Silicon, AMD and plain CPU. Seedmesh's trust layer is
deliberately indifferent to which is chosen — that is why the backend seam exists.
