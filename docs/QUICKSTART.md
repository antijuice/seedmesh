# Quickstart

How to join or run a Seedmesh swarm. Written for someone who has not read anything else in
this repo.

> **Read this first.** Seedmesh is pre-alpha. There is no public swarm to join — these
> instructions are for running a *private* one with people you know. Anything you send
> through a swarm is visible in intermediate form to the machines processing it; see
> [security-privacy.md](security-privacy.md).

## What you need

| | |
| --- | --- |
| **To use a swarm** (send prompts) | Any machine. No GPU. |
| **To host blocks** (donate compute) | Linux or **WSL2** on Windows, an NVIDIA GPU, Python ≥3.10 |

The backend (Petals + hivemind) does not run natively on Windows. Seedmesh's own trust layer
does, which is why `seedmesh probe` and `seedmesh simulate` work anywhere.

## 1. Install

```bash
git clone <this repo> seedmesh && cd seedmesh
pip install -e .
seedmesh probe --model NousResearch/Meta-Llama-3.1-8B-Instruct
```

`probe` needs no backend and no downloads — it reads the model's config and your GPU, and
tells you what you could host:

```
GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU  4.0 GiB (3.9 GiB free, compute 8.6)

[nf4]
  model             NousResearch/Meta-Llama-3.1-8B-Instruct (32 blocks)
  quantization      nf4 (0.516 bytes/param)
  per block         218.1M params, 107 MiB
  usable VRAM       2.9 GiB (after reserve)
  recommendation    27 blocks (84% of the model)
```

## Choosing a model

**Petals implements four architectures — `llama`, `mixtral`, `falcon`, `bloom` — and
nothing else.** This is an architecture limit, not a licensing one: Qwen, Gemma, Phi and
dense Mistral are all refused no matter how permissive their weights. `probe` and `serve`
check before doing any work; a model outside the list fails with one sentence rather than a
traceback from inside the server.

Two verified-reachable, ungated options:

| model | arch | blocks | use |
| --- | --- | --- | --- |
| `JackFram/llama-160m` | llama | 12 | first connectivity test — seconds to download, runs on CPU |
| `NousResearch/Meta-Llama-3.1-8B-Instruct` | llama | 32 | a real swarm |

`meta-llama/*` and `google/gemma-*` are **gated** — anonymous fetches are refused, so every
volunteer would need a Hugging Face account and accepted licence terms. The `NousResearch`
mirror is the same Llama 3.1 weights without that friction.

**Everyone in a swarm must use the same model string.** It determines the DHT prefix, so a
peer on a different model silently joins a different swarm and sees nobody. Prove the network
works on `llama-160m` first, then have everyone restart on the larger model together.

## 2. Install the backend (hosting only)

```bash
seedmesh setup
```

This clones Petals, applies Seedmesh's port, installs the dependencies, and verifies the
result. It takes a few minutes, mostly downloading torch.

The port is necessary, not optional: Petals has been unmaintained since 2024-09-07 and does
not run on current transformers, hivemind or numpy without it. Details in
[petals-port.md](petals-port.md). `setup` is idempotent — re-run it any time.

## 3. Join a swarm

Ask whoever runs the swarm for its **bootstrap address**. It looks like:

```
/ip4/203.0.113.10/tcp/31337/p2p/QmXh3hVojQEJ72bP1ZCe4UTMLSD9Cje9KF7SDjeT5cQjh7
```

Then:

```bash
seedmesh serve --model JackFram/llama-160m --initial-peers <that address> --public-name "your-name"
```

Block count is auto-sized from your GPU. Override with `--num-blocks N` if you want to
donate less.

## 4. Use a swarm

```bash
seedmesh chat --model JackFram/llama-160m --initial-peers <that address>
```

## 5. Run your own swarm

Someone needs a **publicly reachable** machine. This is the part people underestimate:
neither a home PC behind NAT nor a Colab notebook can be dialled from outside, so a swarm of
those alone cannot form. A cheap VPS (~$5/month, **no GPU needed** — a bootstrap peer only
relays discovery metadata) is the usual answer.

On the VPS:

```bash
seedmesh bootstrap --port 31337 --announce-ip <the VPS's public IPv4>
```

A bootstrap peer is a DHT node, not a server with zero blocks — it takes no `--model`, and
one bootstrap serves any swarm. (`serve --num-blocks 0` crashes inside Petals about a minute
in, after looking healthy; it now refuses up front and points here.)

It prints a `/ip4/.../tcp/31337/p2p/Qm...` address. That is what everyone else passes to
`--initial-peers`. Open port 31337.

Everyone else then runs step 3 or 4 against it.

## Troubleshooting

**`hivemind does not run natively on Windows`** — expected. Use WSL2:
`wsl --install -d Ubuntu`, then run everything inside the Ubuntu shell.

**`<model> is gated`** — the model needs a Hugging Face account and accepted licence terms.
Prefer permissively-licensed models; see `seedmesh/models/registry.yaml`.

**`Petals has no block implementation for model type X`** — the architecture is unsupported.
Only `llama`, `mixtral`, `falcon` and `bloom` exist; see *Choosing a model* above. If you get
the raw form of this, `ValueError: Petals does not support model type X` from inside a
traceback, you are on an older checkout that pre-dates the up-front check.

**`could not reach huggingface.co`** — usually transient. Hugging Face resets anonymous
connections under load; `probe` and `serve` already retry three times, so a persistent
failure is more likely a real network problem than a bad model name.

**Connected, but the swarm looks empty** — check every peer is using the *identical* model
string. It sets the DHT prefix, so a mismatch puts people in separate swarms with no error.

**`routing: not found` right after someone starts serving** — normal, and temporary. A
server's *block* records reach the DHT before its *peer routing* record does, so for a few
minutes the swarm advertises a server nobody can dial yet. Measured on a real two-host
swarm: a NAT'd laptop showed `ONLINE` with all 12 blocks and was unreachable at one attempt,
then served the identical query ~5 minutes later with nothing changed. Wait and retry. If it
persists past ~10 minutes, see [NAT-AND-RELAYS.md](NAT-AND-RELAYS.md).

**`No GPU detected and --num-blocks not given`** — `probe` found no CUDA device. You can
still use a swarm as a client.

**Server starts, then nobody can reach it** — you are behind NAT. Either forward the port,
or join an existing swarm rather than hosting the bootstrap.

**Out of memory shortly after starting** — the auto-size reserve is a heuristic, not a
measurement. Re-run with a lower `--num-blocks`.

## What does not work yet

Being explicit, so nothing here surprises you:

- **No public swarm exists.** Private swarms only.
- **Verification does not run automatically during inference.** It works, and is driven in
  `tools/backend_demo.py`, but a serving client does not yet sample its own requests inline.
- **Multi-host now works, with caveats.** Verified 2026-08-01 across two real hosts on the
  public internet: a VPS bootstrap peer and a laptop behind home NAT hosting all 12 blocks
  of a 160M model, with a client routing to it and getting coherent tokens back. What is
  still unproven: more than two hosts, more than one server, GPU inference, models above
  160M, and cross-host verification distances.
- **Only clients gossip reputation.** Servers observe nothing (they receive requests, they
  do not route them), so in a swarm where one person runs the client, there is only one
  observer and nothing to exchange. Two or more people using the swarm is what makes
  reputation collective.

Reputation *is* now shared and persisted — see below.
