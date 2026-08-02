# Moving the swarm to Llama-3.1-8B

Everything here is measured against `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated, same
weights as `meta-llama/*` without the licence gate) on a 4 GiB RTX 3050.

## Before anything: fix torch

The single blocker. `pip install petals` pulls the **CPU-only** torch wheel by default, and
Petals then runs the whole model on the CPU with nothing saying so — `probe` reads the GPU
via nvidia-smi, Petals reads it via `torch.cuda`, and those disagree silently.

`seedmesh probe` now catches this and prints the fix. Check both machines:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it says `+cpu` or `False`:

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0+cu126
```

`cu126` carries torch 2.13.0 — the same version as the CPU wheel — so this is a swap, not an
upgrade that drags dependencies with it.

A fresh `seedmesh setup` on a machine with a GPU already does the right thing: it installs
plain PyPI torch, which on Linux bundles CUDA. The CPU wheel only appears if you installed by
hand or on a GPU-less box.

## What it costs each person

| | |
| --- | --- |
| weights on disk | **11–15 GiB** — Petals downloads whole safetensors shards, so a 15-block range still pulls 3 of 4 shards |
| VRAM, 15 blocks at nf4 | 1.6 GiB weights + 0.94 GiB attention cache = **2.5 GiB** |
| extra for anyone running `chat` | **1.96 GiB** (embeddings + lm_head, downloaded once) |
| wire cost | **16 KiB per token per hop** |

The disk figure is the one that surprises people: quantization happens *after* download, so
`--quant nf4` saves VRAM and saves you nothing on the download.

## The block arithmetic, and the lever that makes it work

Two 4 GiB cards at defaults give **15 blocks each = 30**, which is two short of 32. The model
would not be usable.

The fix is the attention cache. Petals reserves 16384 tokens per block for grouped-query
models like this one, which is 64 MiB per block — more than half the size of the *weights* at
nf4. That budget bounds how much concurrent session length a server can hold. It is **not**
the context window and not model quality:

```
  per block         218.1M params, 171 MiB (107 weights + 64 attention cache)
  cache is large    --attn-cache-tokens 4096 would fit 21 blocks instead of 15
```

At 4096 tokens the cache drops to 16 MiB/block and each card fits **21 blocks — 42 slots for
32 blocks**, so the model is covered with redundancy on ten of them. That is the configuration
to use.

## Running it

The model now lives in `seedmesh/swarm.json`, so neither of you passes `--model`:

```bash
git pull && pip install -e .
seedmesh probe                      # confirm the GPU warning is gone
seedmesh serve --attn-cache-tokens 4096
```

Auto-sizing accounts for the smaller cache and passes it to the server too, so the plan and
the allocation agree. First start downloads ~11–15 GiB; subsequent starts are cached.

Then, from either machine:

```bash
seedmesh monitor        # is every block covered?
seedmesh chat
```

**The four droplets need no changes at all.** They run `run_dht`, which is model-agnostic —
one set of bootstrap peers carries any number of swarms, and llama-160m stays reachable with
`--model JackFram/llama-160m` for quick connectivity checks.

## Order of operations

1. Both machines: fix torch, `git pull`, `pip install -e .`.
2. Prove the network still works on the small model:
   `seedmesh monitor --model JackFram/llama-160m`.
3. Both start serving 8B. Expect the first start to be slow — it is downloading.
4. `seedmesh monitor` until coverage shows every block hosted.
5. `seedmesh chat`. The first prompt may wait out the routing warm-up (up to ~5 minutes after
   a server starts); that is normal and `chat` waits rather than failing.

## What to watch for

**"Out of memory shortly after starting"** should now be much less likely — the block plan
charges the attention cache per block rather than hoping a flat reserve covers it. If it still
happens, drop `--num-blocks` a little; auto-sizing uses free VRAM at the moment it runs, and
a browser can move that number by a gigabyte.

**Coverage gaps if one of you drops.** With 21 + 21 the model survives one server leaving only
for the ten doubly-covered blocks. Two people is genuinely thin; a third volunteer is worth
more than either of you adding blocks.

**Relay budget.** At 16 KiB/token, a relayed connection would be severed after ~8 tokens. This
swarm's paths are direct — verified at 6.2× the relay budget with zero failures — but if you
ever see requests dying at a consistent size, that is the cause. See
[NAT-AND-RELAYS.md](NAT-AND-RELAYS.md).
