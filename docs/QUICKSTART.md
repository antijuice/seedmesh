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
| `NousResearch/Meta-Llama-3.1-8B-Instruct` | llama | 32 | a real swarm — this is what `swarm.json` names |

Running the 8B model on two laptop GPUs needs one non-obvious setting; see
[SWITCHING-TO-8B.md](SWITCHING-TO-8B.md).

`meta-llama/*` and `google/gemma-*` are **gated** — anonymous fetches are refused, so every
volunteer would need a Hugging Face account and accepted licence terms. The `NousResearch`
mirror is the same Llama 3.1 weights without that friction.

**Everyone in a swarm must use the same model string.** It determines the DHT prefix, so a
peer on a different model silently joins a different swarm and sees nobody. That is why the
model lives in `swarm.json` -- the file you hand people -- rather than in each person's
command line:

```json
{ "name": "your-swarm", "model": "NousResearch/Meta-Llama-3.1-8B-Instruct", "bootstrap_peers": [...] }
```

`--model` still overrides it when you want to test something. Prove the network works on
`llama-160m` first, then change the file and have everyone restart together.

**Switching models needs no change to the bootstrap peers.** They run `run_dht`, which is
model-agnostic — one set of four droplets carries any number of swarms. Two models can even
run at once on the same peers, since a different model means a different DHT prefix and
therefore a separate namespace; `seedmesh monitor --model X` shows each one.

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

```bash
seedmesh serve
```

That is the whole command. The bootstrap peers and the model come from `seedmesh/swarm.json`,
which ships with the package, so there is nothing to paste. Block count is auto-sized from
your GPU; override with `--num-blocks N` to donate less.

To join a *different* swarm, point at its definition:

```bash
seedmesh serve --swarm-file ./their-swarm.json
```

or pass addresses directly with `--initial-peers`, which overrides everything else.

## 4. Use a swarm

```bash
seedmesh chat
```

Same story — no arguments. It waits out the routing warm-up (a few minutes after a server
starts, its blocks reach the DHT before its address does) rather than failing your first
prompt, and retries a stalled request with a fresh session instead of hanging.

## 5. See what the swarm is doing

```bash
seedmesh monitor
```

Who is hosting which blocks, whether the model is fully covered, and what other people's
clients have measured about each server. Add `--watch 30` to refresh, or
`--html swarm.html` to write a self-contained page you can serve anywhere.

Read the coverage bar first. **A model is usable only if every block has a host** -- twelve
servers all hosting blocks 0-3 serve nothing at all, so "how many volunteers" is the wrong
question and the report does not lead with it.

Two columns that look equally authoritative are not:

| column | where it comes from |
| --- | --- |
| **self-reported** | the server's own announcement -- name, throughput, compute profile. Checked by nothing. |
| **observed by others** | signature-verified reputation records from clients that actually sent it requests |

A server with a high self-reported throughput and no observations is one nobody has tested
yet. Real output from this swarm, showing exactly that:

```
  server                blocks  profile                self-rep.  observed
  ...jSUDSNRi            0-12   nf4/bf16/eager+relay   1091 tok/s  not yet measured
  hewitt                 0-12   none/bf16/eager+relay   173 tok/s  0.992 (3 obs, 1 clust)
```

To publish this page for everyone, see [PUBLIC-MONITOR.md](PUBLIC-MONITOR.md) -- it runs on
a bootstrap droplet with no backend installed, because monitoring only reads DHT keys.

The profile column is quantisation / dtype / attention kernel. Volunteers differ here and
that is normal -- it is why verification compares within a profile instead of treating the
difference as a fault.

## 6. Run your own swarm

You need **at least four publicly reachable peers**. This is not a redundancy
recommendation — it is a hard requirement of go-libp2p, and getting it wrong fails in a way
that looks like everything is fine.

go-libp2p accepts an observed public address for itself only once **four distinct peers**
have independently reported it (`identify/obsaddr.go`, `ActivationThresh = 4`). Below four, a
volunteer behind NAT never learns its own public address, so it advertises nothing dialable,
so every connection falls back to a circuit relay — and go-libp2p resets a relayed connection
after **128 KiB**, which is less than one request of any real model.

Measured on this swarm: with one bootstrap peer, a 117 KB request failed **15/15**. With
four, the same request from a Colab client on the other side of the internet succeeded
**12/12**, directly, with no relay involved.

Stand up four cheap VPSs (~$5/month each, no GPU) following
[BOOTSTRAP.md](BOOTSTRAP.md), then list all four in your own `swarm.json`:

```json
{
  "name": "your-swarm",
  "model": "JackFram/llama-160m",
  "bootstrap_peers": ["/ip4/.../tcp/31337/p2p/Qm...", "...", "...", "..."]
}
```

Hand that file to your volunteers, or set `SEEDMESH_SWARM=/path/to/swarm.json`.

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

**A prompt takes an extra second and prints `(attempt 1 stalled, retrying)`** — working as
intended. `chat` uses a short 10s per-attempt timeout and retries with a *fresh* session,
because a stalled request never recovers on its own; a long timeout only delays the retry
that works. Tune with `--timeout` / `--attempts`.

**Known limitation: a server dying mid-generation loses that request.** Petals' own recovery
path is broken (`_update_sequence()` copies `history` into the replacement session but leaves
its `_position` at 0, so the next step asserts `0 and N`). Seedmesh sets `max_retries=1` to
keep Petals off that path and retries a level up instead, but an in-flight request cannot be
resumed. Send the prompt again.

**`No GPU detected and --num-blocks not given`** — `probe` found no CUDA device. You can
still use a swarm as a client.

**Server starts, then nobody can reach it** — run `seedmesh doctor`, which answers this
directly instead of leaving you to infer it. Otherwise: count your bootstrap peers first. Fewer than
four and your machine never learns its own public address, so it advertises nothing dialable.
That is the cause far more often than the router is. If you do have four and it still fails,
your NAT is probably symmetric or you are behind CGNAT; forward the port. See
[NAT-AND-RELAYS.md](NAT-AND-RELAYS.md).

**Out of memory shortly after starting** — much less likely than it was: the plan now
charges the attention cache per block instead of hoping a flat reserve covers it. If it still
happens, re-run with a lower `--num-blocks`; auto-sizing uses free VRAM at the moment it runs,
and a browser can move that by a gigabyte.

**`cache is large` in probe output** — the attention cache can exceed half the size of a
block's weights, because quantization shrinks weights and leaves the cache untouched.
`--attn-cache-tokens 4096` trades concurrent session capacity for blocks; on an 8B model it is
the difference between two 4 GiB cards covering the model and falling two blocks short.

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
