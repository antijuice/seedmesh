# Seedmesh

**A trust layer for public peer-to-peer inference swarms.**

No company should be able to price-gate or switch off access to open models. The way to
prevent that is infrastructure nobody owns — a swarm where anyone can donate a slice of a
large model and anyone can use it.

That idea has been built before. [Petals](https://github.com/bigscience-workshop/petals)
proved the hard part works: split a 70B+ model across volunteer GPUs, stream activations
peer to peer, get tokens back. The protocol was never the problem.

The problems were **maintenance** and **trust**. Petals has had no commits since
**2024-09-07**, and its public swarm is gone — `health.petals.dev` refuses connections, and
`chat.petals.dev` reports it is out of capacity while offering a Llama-2 fine-tune from
2023. Meanwhile it never had a way to answer *which servers are reliable* beyond throughput
each server self-reported about itself, and no way at all to answer *did this server
actually do the work*.

Seedmesh builds the missing half.

> **Status: pre-alpha. There is no public swarm.** The trust layer is built and tested, and
> a *private* swarm now runs real inference through it end to end — discovery, routing,
> per-server execution, and sampler-driven verification — see
> [docs/petals-port.md](docs/petals-port.md). What that does not mean: no public network
> exists, and inference has only run on CPU at toy model size on a single host. Verification
> thresholds are measured across five real GPU architectures, and network clustering uses a
> real offline ASN table — but a single-host swarm has no routable addresses, so ASN
> separation itself is still only exercised against a simulated mapping. Said plainly because
> this project's pitch is trust, and overstating readiness is a bad way to start.

---

## What's here

```
seedmesh/
├── core/            self-certifying identity, canonical signing, domain types
├── reputation/      scoring, signed records, sybil-resistant aggregation, routing
├── verification/    hidden-state sketches, tolerance comparison, calibration, sampling
├── backends/        the three-method seam a real transport plugs into
└── sim/             deterministic swarm simulator with adversarial scenarios
```

Try it:

```bash
pip install -e . && seedmesh probe --model Qwen/Qwen3-8B
```

`probe` needs no GPU-side install and no weight download — it reads the model config and
your hardware and tells you what you could host. On a 4 GB laptop GPU that is 30 of an 8B
model's 36 blocks at NF4.

To run or join a swarm, see **[docs/QUICKSTART.md](docs/QUICKSTART.md)**:

```bash
seedmesh setup                                        # install + patch the backend
seedmesh serve --model <m> --initial-peers <addr>     # donate compute
seedmesh chat  --model <m> --initial-peers <addr>     # use the swarm
seedmesh simulate                                     # adversarial scenarios, no backend
```

`simulate` runs three scenarios end to end — healthy, mixed-threat, and a sybil fleet — and
prints calibrated thresholds, detection outcomes and load distribution.

## Two corrections that shaped the design

**Hidden states cannot be compared by hash.** The obvious way to check that two servers did
the same work is to hash their outputs. It cannot work: floating-point addition is not
associative, so an A100 in bf16 and a 3090 in fp16 agree to a few decimals and differ in the
low bits, every time. Under hash comparison *every honest heterogeneous pair* reads as a
mismatch — and heterogeneous hardware is the entire premise of a volunteer network. Seedmesh
compares random-projection sketches against thresholds **calibrated from measured honest
disagreement**.

**A mismatch does not say who was wrong.** It proves one of two peers is wrong. Charging both
at full weight evicts the honest node that just caught a cheat — in simulation, 50% of
quarantines were honest servers, which makes verifying actively dangerous. And counting
accusers by peer id lets a 12-identity fleet manufacture accusers: it convicted four honest
servers and none of its own members. Seedmesh attributes disagreements to the partner's
**network cluster** and requires corroboration from two independent clusters before excluding
anyone.

Both defects were found by the simulator, not by inspection. That is what it is for.

## Measured behaviour

`seedmesh simulate`, 400 requests per preset. Real arrays, real sketches, real comparisons —
only the network and GPUs are simulated.

| Preset | Cheats caught | Honest wrongly excluded | Malicious traffic, 2nd half |
| --- | --- | --- | --- |
| healthy (mixed hardware) | — | **0** | 0 |
| mixed threats | **3/3** | **0** | 0 |
| sybil fleet (12 identities, one ASN) | **12/12** | **0** | 0 |

A 50-strong sybil fleet insisting a bad peer is perfect, against 5 honest observers rating
it 0.20, moves the aggregate only to **0.255**.

These are simulator results on a synthetic topology. They show the design is internally
sound and that specific attacks fail against it. They are **not** measurements of a real
swarm.

## Design commitments

**No token, no payment, ever.** Reputation is the only currency: donate reliably, get
preferred service. Every token-based "decentralized AI" project studied for this hits the
same wall — emissions outrun real usage, price falls, suppliers switch off, capacity
spirals. Payouts tied to speculation rather than demand do not durably work.

**The public swarm is not private.** Prompts pass through strangers' machines as
activations. Activations are not plaintext, but recovering input from hidden states is an
active research area, so treating them as safe would be wrong. For sensitive data, run a
private swarm. See [docs/security-privacy.md](docs/security-privacy.md).

**Verification checks agreement, not correctness.** Redundant sampling raises the cost of
returning garbage. It cannot certify any answer is right, and it is probabilistic and
delayed — a cheat is caught after some requests, not before its first.

**Honest about what is not defended.** An attacker with real network diversity across many
ASNs defeats the anti-sybil rules. A caught peer can discard its identity and rejoin. See
[docs/threat-model.md](docs/threat-model.md), which lists the gaps.

## Documentation

| | |
| --- | --- |
| [QUICKSTART.md](docs/QUICKSTART.md) | Install, join a swarm, run one — and what doesn't work yet |
| [BOOTSTRAP.md](docs/BOOTSTRAP.md) | Standing up the always-on rendezvous peer a swarm needs |
| [NAT-AND-RELAYS.md](docs/NAT-AND-RELAYS.md) | Hosting from a laptop behind home wifi |
| [architecture.md](docs/architecture.md) | How the layers fit, and why each defence exists |
| [threat-model.md](docs/threat-model.md) | Defended, not defended, open questions |
| [security-privacy.md](docs/security-privacy.md) | For volunteers and for users |
| [findings-upstream-audit.md](docs/findings-upstream-audit.md) | Verified state of Petals and hivemind |
| [GOVERNANCE.md](GOVERNANCE.md) | Who decides what, and the non-monetization pledge |

## Backend status

The trust layer imports no ML framework. A backend satisfies three methods
(`discover`, `n_blocks`, `run_segment`) and plugs in underneath.

| Backend | Status |
| --- | --- |
| Simulator | Implemented — what the trust layer is tested against |
| **Petals + hivemind** | **Chosen for v1, and working.** A private swarm generates real tokens; `PetalsBackend` drives it through the trust layer. See [docs/petals-port.md](docs/petals-port.md) |
| llama.cpp RPC | Planned as a second backend for **private/LAN meshes only**. Far broader hardware reach, but its RPC is unauthenticated by design — measured: an anonymous socket enumerated the host and allocated 256 MiB on it. See [llamacpp_rpc](spike/llamacpp_rpc/) |

The backend decision turned on architecture, not port cost. llama.cpp RPC inverts the
volunteer safety guarantee: in Petals the server owns the weights and the graph and the
client sends only a tensor, so "hosting a block can't run arbitrary code on your machine"
is structurally true. In llama.cpp RPC the *client* submits the compute graph. That is right
for a trusted LAN — which is what its README says — and wrong for a public swarm.

Petals compatibility, measured rather than guessed:

| component | result | spike |
| --- | --- | --- |
| Llama block wrapper | port required — 138 lines replacing 221, numerically exact | [transformers_port](spike/transformers_port/) |
| bitsandbytes int8/NF4 | **works unmodified** across 9 minor versions | [quantization](spike/quantization/) |
| tensor_parallel | **deletable** for single-GPU, which is the volunteer case | [quantization](spike/quantization/) |
| hivemind DHT/p2p | **7 import statements across 22 files** — mechanical | [hivemind_dht](spike/hivemind_dht/) |
| server/client logic above the imports | still unexercised end-to-end | — |

## Requirements

Python ≥3.10, numpy, cryptography, PyYAML. No GPU, no torch, runs on Windows.

The eventual Petals backend does not: it needs Linux (or WSL2) and Python 3.10/3.11, since
its `numpy<2` pin has no wheels for 3.13.

## License

MIT, matching upstream Petals so fixes can flow either way.
