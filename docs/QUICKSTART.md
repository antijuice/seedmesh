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
seedmesh probe --model Qwen/Qwen3-8B
```

`probe` needs no backend and no downloads — it reads the model's config and your GPU, and
tells you what you could host:

```
GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU  4.0 GiB (3.9 GiB free, compute 8.6)

[nf4]
  model             Qwen/Qwen3-8B (36 blocks)
  per block         192.9M params, 95 MiB
  recommendation    30 blocks (83% of the model)
```

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
seedmesh serve --model Qwen/Qwen3-8B --initial-peers <that address> --public-name "your-name"
```

Block count is auto-sized from your GPU. Override with `--num-blocks N` if you want to
donate less.

## 4. Use a swarm

```bash
seedmesh chat --model Qwen/Qwen3-8B --initial-peers <that address>
```

## 5. Run your own swarm

Someone needs a **publicly reachable** machine. This is the part people underestimate:
neither a home PC behind NAT nor a Colab notebook can be dialled from outside, so a swarm of
those alone cannot form. A cheap VPS (~$5/month, **no GPU needed** — a bootstrap peer only
relays discovery metadata) is the usual answer.

On the VPS:

```bash
seedmesh serve --model Qwen/Qwen3-8B --num-blocks 0 \
  --host-maddrs /ip4/0.0.0.0/tcp/31337
```

It prints `Running a server on ['/ip4/.../tcp/31337/p2p/Qm...']`. That address is what
everyone else passes to `--initial-peers`. Open port 31337.

Everyone else then runs step 3 or 4 against it.

## Troubleshooting

**`hivemind does not run natively on Windows`** — expected. Use WSL2:
`wsl --install -d Ubuntu`, then run everything inside the Ubuntu shell.

**`<model> is gated`** — the model needs a Hugging Face account and accepted licence terms.
Prefer permissively-licensed models; see `seedmesh/models/registry.yaml`.

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
- **Multi-host has never been tested.** Everything so far ran on one machine, so NAT
  traversal and relays are unexercised — the most likely thing to break first, and the main
  reason to run this with friends.
- **Only clients gossip reputation.** Servers observe nothing (they receive requests, they
  do not route them), so in a swarm where one person runs the client, there is only one
  observer and nothing to exchange. Two or more people using the swarm is what makes
  reputation collective.

Reputation *is* now shared and persisted — see below.
