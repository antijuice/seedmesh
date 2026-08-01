# Calibration: replacing simulated thresholds with measured ones

Every verification threshold in Seedmesh is currently fitted to **simulated** floating-point
noise. Until that changes, verification runs but does not mean much — the numbers it
compares against were invented by `seedmesh/sim/world.py`, not measured on hardware anyone
will actually donate.

This directory turns that around. It needs **several different GPU models**, not powerful
ones, which is exactly what Colab hands you for free: you cannot choose which GPU a session
gets, and here that variability is the feature.

## Why this works on Colab specifically

The thing calibration measures is *how far apart two honest servers land*. That needs two
different GPUs computing the same thing — but crucially **not at the same time**. Each
session writes a small file; the comparison happens offline afterwards.

A sketch is 64 floats, so a session file with 64 samples × 5 precision modes is a few
hundred KB. You are carrying kilobytes between sessions, not model weights.

Everything that could vary at runtime is pinned so sessions are comparable:

* **weights** — block 0 of `JackFram/llama-160m`, a real published model, identical bytes
  every time (not randomly initialised);
* **inputs** — generated from a fixed seed on CPU;
* **sketch projections** — fixed derived seeds;
* **attention kernel** — pinned to `eager`, since the kernel changes the noise floor.

`fit_tolerances.py` refuses to merge sessions that disagree on model or input seed.

## Step 1 — collect, once per GPU type

In a Colab notebook with a GPU runtime (Runtime → Change runtime type → GPU):

```python
!nvidia-smi --query-gpu=name --format=csv,noheader     # note which GPU you got
!pip install -q transformers bitsandbytes accelerate

# Get the repo into the session. Either upload a zip and unzip it, or:
!git clone https://github.com/<you>/seedmesh.git /content/seedmesh
%cd /content/seedmesh

!python tools/calibrate/collect_sketches.py --out /content/session_t4.json --label colab-t4
```

Then download `session_t4.json` (Files pane, or `google.colab.files.download`).

**Repeat until you have at least two different GPUs.** Colab commonly serves T4, L4 and
A100 depending on tier and load; reconnecting the runtime is usually enough to get a
different one. Three types is meaningfully better than two — the threshold is a quantile of
a distribution, and two points is a thin distribution.

Modes that a given GPU cannot run (int8 and NF4 need bitsandbytes and a supported card) are
recorded as unavailable rather than failing the session.

## Step 2 — fit, locally

```bash
python tools/calibrate/fit_tolerances.py sessions/*.json --out tolerance_table.json
```

It prints the pooled distance distribution per profile pair and writes a `ToleranceTable`
loadable directly by `seedmesh.verification.calibrate.ToleranceTable.load`.

**It exits non-zero if all sessions came from one GPU type.** With a single machine, a
same-mode comparison is a run against itself, which is not a measurement — so those pairs are
**skipped entirely** rather than fitted to a meaningless zero. They end up absent from the
table, and an absent pair makes the sampler refuse to verify rather than guess. Correct, but
it means same-profile servers cannot check each other until a second GPU type exists.

## What gets pooled into what

The part worth understanding before trusting the output.

A tolerance is keyed by an **unordered pair of compute profiles**. The entry for
`fp16 vs fp16` is fitted from every *cross-session* same-mode comparison — two honest fp16
servers on different hardware. That pooling is the whole point: it is what makes the
threshold cover honest hardware diversity rather than one machine's idea of zero.

Cross-mode pairs (`fp16 vs nf4`, and so on) pool every session combination, so they capture
quantization error *and* hardware error together — which is what a real mixed swarm
presents.

## Results from a real 5-GPU run

Collected 2026-07-31 across five architectures — Tesla T4 (Turing, cc 7.5), A100 (Ampere,
8.0), RTX 3050 (Ampere, 8.6), L4 (Ada, 8.9), RTX PRO 6000 Blackwell (12.0) — 64 samples each.

**Same mode, different hardware.** This is the number a single machine cannot produce, and
the reason the exercise exists:

| mode | max distance across GPUs |
| --- | --- |
| fp32 | 0.00000019 |
| int8 | 0.0000063 |
| nf4 | 0.00021 |
| fp16 | 0.00016 |
| **bf16** | **0.00143** |

**Different modes** (p50, pooled across all GPU pairs):

| pair | p50 | p99.9 |
| --- | --- | --- |
| fp16 vs fp32 | 0.00020 | 0.00028 |
| bf16 vs fp16 | 0.00165 | 0.00247 |
| int8 vs fp16 | 0.00528 | 0.00767 |
| **nf4 vs fp16** | **0.03339** | 0.04515 |

Three things worth taking from this:

**Quantization dominates hardware by ~23x.** The worst honest hardware disagreement (bf16,
0.0014) is far below the smallest quantization gap that matters (nf4 vs fp16, 0.033). The
per-pair tolerance design was necessary; a single global threshold could never have covered
both.

**fp32 is effectively bit-reproducible across five architectures** (1.9e-7), and int8 nearly
so (6.3e-6) — int8 quantization being deterministic given identical weights. bf16 is the
outlier at ~9x fp16, which follows from its shorter mantissa.

**The earlier single-GPU warning was right in principle, overstated in detail.** The
quantization spike said same-mode zeros were floor artefacts needing replacement. True for
bf16 (0.0 → 0.0014) and fp16 (0.0 → 0.00016); for fp32 and int8 the single-GPU zero was
nearly correct.

Note these differ from the ~0.055 nf4 figure in `spike/quantization/`, which used synthetic
weights and a different sequence length: **absolute distances are model- and shape-dependent,
so calibrate for the model you actually serve.**

### The floor is now the binding constraint for the tightest modes

fp32 and int8 hit the `--floor` (1e-4) rather than their measured p99.9 (1.8e-7 and 6.3e-6).
The fitter reports which pairs are floor-bound. The floor errs *permissive*, never hostile —
but it costs sensitivity: a cheat perturbing fp32 output by under 1e-4 hides inside it, when
the data would support a threshold ~500x tighter.

Left at 1e-4 by default deliberately. Tightening on the evidence of five GPUs risks fitting
a thin hardware sample, and a false accusation is the expensive error. Lower it with
`--floor` once the sample is broader.

## Step 3 — use it

```python
from pathlib import Path
from seedmesh.verification.calibrate import ToleranceTable

tolerances = ToleranceTable.load(Path("tolerance_table.json"))
sampler = VerificationSampler(scorer, clusters, tolerances=tolerances)
```

An uncalibrated profile pair returns `None`, and the sampler refuses to verify it rather
than guessing a threshold — see `seedmesh/verification/calibrate.py`.

## Honest limits

* One block of one small model. Deeper blocks and larger hidden sizes may differ; the same
  scripts extend to those by changing `MODEL` and `BLOCK_INDEX`.
* Colab's GPU pool is not the volunteer population. It skews modern-datacenter; a swarm will
  have consumer cards this never sees.
* Thresholds are a snapshot. A swarm whose hardware mix changes needs re-calibration, and
  re-fitting on a population that already contains cheats would widen the tolerance until
  they fit inside it.
