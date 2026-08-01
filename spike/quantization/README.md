# Spike: Petals' quantization path on current bitsandbytes

**Question:** Petals pins `bitsandbytes==0.41.1` (September 2023) and
`tensor_parallel==1.0.23`. The transformers port spike concluded these were where the
remaining backend risk sat. Is it a port or a replacement — and does quantization break
Seedmesh's verification design?

Measured 2026-07-31. bitsandbytes 0.50.0, transformers 5.14.1, torch 2.13 + CUDA,
RTX 3050 Laptop (4GB, compute 8.6).

---

## Reproducing

```bash
python -m venv .venv-quant
.venv-quant/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv-quant/Scripts/python.exe -m pip install "transformers==5.14.1" "bitsandbytes==0.50.0" accelerate
.venv-quant/Scripts/python.exe -m pip install -e .

.venv-quant/Scripts/python.exe spike/quantization/test_quantization.py
```

Needs a CUDA GPU — bitsandbytes' int8 and NF4 kernels are GPU-only, and measuring them on
CPU would characterise a fallback path no volunteer would actually run.

## Source analysis (independent of the measurements below)

### The bitsandbytes surface is tiny

All of Petals' quantization coupling is **four constructors** in
`petals/utils/convert_block.py::quantize_module`:

```python
bnb.nn.Linear8bitLt(in, out, bias, has_fp16_weights=False, threshold=6.0)
bnb.nn.Int8Params(data, requires_grad=False, has_fp16_weights=False)
bnb.nn.LinearNF4(in, out, bias, compress_statistics=True)
bnb.nn.Params4bit(data, requires_grad=False, quant_type="nf4", blocksize=64,
                  compress_statistics=True)
```

That is a completely different risk profile from the transformers coupling, which forked an
entire attention implementation. Four constructors either kept their signatures or did not.

### `tensor_parallel` is on the single-GPU path — but is deletable

An earlier note in this project claimed `tensor_parallel` was only reachable on a multi-GPU
path. **That was wrong**, and reading the source corrects it. `convert_block` calls
`make_tensor_parallel` **unconditionally** (line 53), and the docstring is explicit:

> *"if there is only a single device, model will still be wrapped with TensorParallel (for
> uniformity)"*

So every Petals server, including a one-GPU volunteer, goes through `tp.TensorParallel`.

It also breaks for a **second, independent reason**. `make_tensor_parallel` does:

```python
for submodule in tp_shard.modules():
    if isinstance(submodule, model_config.attn_class):
        total_heads += submodule.num_heads
assert total_heads == model_config.num_attention_heads
```

`submodule.num_heads` is the *same removed attribute* that broke the Llama block port. So
`tensor_parallel` fails on current transformers regardless of its own maintenance state.

**But it is doing nothing functional for a single device** — the docstring says so, and
Petals separately warns that "tensor parallelism is not tested for models other than BLOOM."
A port simply skips the wrapper when `len(devices) == 1`, which:

- drops the `tensor_parallel==1.0.23` dependency entirely for the volunteer case;
- removes the second `num_heads` breakage;
- costs nothing, because the volunteer case is one GPU.

Multi-GPU tensor parallelism would need real work. It is also a minority path that upstream
already flagged as under-tested, and no volunteer network should depend on it.

---

## Q1 — the bitsandbytes API is completely intact

All four constructors exist on 0.50.0, and Petals' **unmodified** `quantize_module` applies
cleanly for both int8 and NF4:

```
OK  bnb.nn.Linear8bitLt      OK  bnb.nn.LinearNF4
OK  bnb.nn.Int8Params        OK  bnb.nn.Params4bit

int8  OK -> Linear8bitLt
nf4   OK -> LinearNF4
```

**Quantization is not a port. It is a no-op.** Nine minor versions and the four signatures
Petals depends on did not move. This is the opposite of the transformers result, and the
reason is structural: Petals *used* bitsandbytes' public API and *forked* transformers'
internals. Public APIs keep their promises; copied internals do not.

One configuration detail worth carrying into the adapter: bitsandbytes warns that
`bnb_4bit_compute_dtype` defaults to fp32 while inputs are fp16. Petals never sets it. That
is slower but *more* accurate — and since it changes the numerical noise floor, it belongs
in the published compute profile (see Q3).

## Q2 — memory footprint, and Petals' constant is slightly conservative

Measured on a real block (45,092,864 parameters), walking actual storage including NF4's
out-of-band quantization statistics:

| mode | MiB | bytes/param | vs fp16 | Petals' constant |
| --- | --- | --- | --- | --- |
| fp16 | 86.0 | 2.000 | 100% | 2.0 |
| int8 | 43.0 | 1.000 | 50% | 1.0 ✓ exact |
| nf4 | 22.2 | **0.516** | 25.8% | 0.531 (over-estimates by 3%) |

Petals' `4.25/8` comment says "measured empirically" and it is nearly right, erring on the
*safe* side — a wizard using it under-promises block counts, which is the correct direction
to be wrong in.

Blocks that fit, for a Llama-3-8B-shaped block (218,112,000 params), reserving 1GiB for
activations and KV cache:

| VRAM | fp16 | int8 | nf4 |
| --- | --- | --- | --- |
| 4GB | 7 | 14 | **28** |
| 8GB | 17 | 34 | 66 |
| 12GB | 27 | 54 | 104 |
| 24GB | 56 | 113 | 219 |

This is the table the onboarding wizard needs. Note what it means for the volunteer pool: a
4GB laptop GPU contributes 28 NF4 blocks — roughly **⅞ of a 32-layer model** — versus 7 at
fp16. NF4 is what makes ordinary hardware useful, which is exactly why the Q3 result matters
so much.

## Q3 — quantization broke Seedmesh's verification, and the spike caught it

Honest-pair sketch distances, 60 samples per pair, identical weights:

| pair | p50 | p99.9 |
| --- | --- | --- |
| fp16 vs fp16 | 0.00000 | 0.00000 |
| fp16 vs bf16 | 0.00312 | 0.00435 |
| fp16 vs int8 | 0.00508 | 0.00652 |
| bf16 vs int8 | 0.00602 | 0.00770 |
| **fp16 vs nf4** | **0.05544** | **0.06935** |
| **bf16 vs nf4** | **0.05556** | **0.06982** |
| **int8 vs nf4** | **0.05505** | **0.07001** |

LLM.int8() is genuinely near-lossless — fp16-vs-int8 (0.005) is barely worse than
fp16-vs-bf16 (0.003). **NF4 is an order of magnitude further out**, and consistently so:
every NF4 pairing lands at ~0.055 regardless of what it is compared against, because NF4's
own error dominates the measurement.

Against the calibrated global thresholds the trust layer was using
(`MATCH ≤ 0.01707`, `MISMATCH ≥ 0.10241`), that puts **every NF4 comparison in the ambiguous
band, every time**.

That is not merely degraded — it is fatal, via a mechanism added earlier to catch a
*different* attack. The ambiguous-rate gate treats a peer whose ambiguous rate exceeds 25%,
across two independent clusters, as accumulating dispute weight. An honest NF4 server is
ambiguous 100% of the time. Traced through the real scorer:

```
verification    cluster  integrity  accusers  QUARANTINED
           4    asn:400      1.000         0        False
           5    asn:100      0.211         1        False
           6    asn:200      0.032         2         True   <-- honest server evicted
```

**An honest NF4 volunteer is quarantined after six verifications.** It looks exactly like
the threshold-hugging attacker the gate was built to catch, because "consistently just
inside the ambiguous band, across many independent clusters" describes both.

This is the fourth time in this project that honest heterogeneity has been mistaken for
fraud, after: hash comparison across GPUs, charging both parties of a mismatch, and counting
sybil accusers by peer id. The pattern is now unmistakable enough to state as a rule:

> **Any verification threshold that is not a function of the participants' declared compute
> configuration will eventually convict the honest servers that differ most from the
> majority — and those are exactly the volunteers with the cheapest hardware.**

### The fix, implemented

- `ComputeProfile` (quant / dtype / attention kernel) is now part of `ServerInfo` and must
  be published in the block announcement.
- `ToleranceTable` replaces the single global `Tolerance`, keyed by the **unordered pair**
  of profiles, so a verdict cannot depend on which server the client called the subject.
- **An uncalibrated pair returns `None`, meaning "do not verify this pair."** Same call the
  sampler already makes when no independent verifier exists: a check judged against a
  made-up threshold manufactures evidence the reputation layer will act on.
- The sampler prefers same-profile verifiers (~10x tighter, so far more discriminating) and
  falls back to calibrated cross-profile pairs.

`tolerance_table.json` in this directory is the measured output, loadable directly via
`ToleranceTable.load`.

Lying about your own profile is self-punishing rather than profitable: it selects the wrong
tolerance for your own comparisons. Claiming NF4 while running fp16 gets you judged against
a uselessly wide band; claiming fp16 while running NF4 fails every check you take part in.

### What these numbers do NOT cover

**Only one GPU was available, so this is the *quantization* component of honest
disagreement, not the whole of it.** Every same-mode pair measured exactly 0.00000 because
it is literally the same weights on the same silicon — real fp16-vs-fp16 across an A100 and
a 3090 is nonzero, and the fitted same-mode tolerances in `tolerance_table.json` are floor
artefacts, not measurements. They must be re-fitted with real hardware diversity before any
production use.

That is precisely the calibration run that needs several GPU models — and per
`docs/outreach-drafts.md`, a month of Colab supplies exactly that, because sketches are 64
floats and can be compared across sessions.

## Verdict

**The Petals backend is in far better shape than the audit feared.**

| component | status |
| --- | --- |
| block wrappers | port required — recipe proven, 138 lines, numerically exact |
| **bitsandbytes** | **works unmodified** |
| **tensor_parallel** | **deletable for single-GPU, which is the volunteer case** |
| hivemind | maintained (last commit 2026-01-11) |
| server / client / DHT plumbing | not yet assessed — the remaining unknown |

Both spikes have now come back cheaper than predicted, and the stale-dependency risk that
motivated the backend seam has largely evaporated. The seam still earned its place: it is
what let the trust layer be built, tested and *corrected* without waiting on any of this.

The next unknown is no longer numerical. It is Petals' server/client/DHT plumbing against
current hivemind — plus the real question the port cost was never going to settle, which is
whether PyTorch-only hardware reach is acceptable for a volunteer network where most
contributors will not own an NVIDIA GPU.
