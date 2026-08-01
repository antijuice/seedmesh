# Seedmesh: MVP Technical Spec

*A publicly-shared, BitTorrent-style inference network for open models*

> **Revision 2 — 2026-07-31.** This document has been rewritten against verified evidence.
> The original draft was written from research; several of its load-bearing premises turned
> out to be wrong when checked directly, and two of its design proposals could not have
> worked as written. Corrections are marked **[CORRECTED]** inline with what was believed,
> what is actually true, and how it was verified — the reasoning is kept rather than
> silently overwritten, because knowing *why* a premise failed is what stops it recurring.
>
> Everything unmarked is unchanged from the original and still stands.
>
> **Current status:** the reputation and verification layer is **built and tested** (138
> tests, 3 adversarial simulation scenarios). No backend adapter exists, so **no real
> inference runs yet**. There is no swarm.

---

## 0. The premise: fork, don't rebuild

**[CORRECTED — the fork target changed.]**

*Originally believed:* "Petals is not dead, it's just quiet." Checked via GitHub showing
recent activity, 10.3k stars, and a live `health.petals.dev`.

*Actually true, verified 2026-07-31 against the GitHub API and the live endpoints:*

| Signal | Reality |
| --- | --- |
| Petals last commit (`pushed_at`) | **2024-09-07** — 23 months ago |
| Petals `updated_at` | 2026-07-31 (bumps on *stars/watches*, not code — this is what misled the original draft) |
| `health.petals.dev` | **Connection refused** |
| `chat.petals.dev` | Loads; *"out of capacity — attention caches of existing servers are full"*, offering only Stable Beluga 2 (70B), a Llama-2 finetune from 2023 |
| **hivemind** last commit | **2026-01-11 — alive and maintained** |

So Petals is not quiet, it is **abandoned**, and the public swarm is **gone**. But the
original insight — *don't rebuild the hard 20%* — survives intact. It just points at
**hivemind** (the Kademlia DHT, peer discovery, tensor transport) rather than at Petals.

Two consequences that change the plan:

1. **There is no upstream to merge fixes from.** Vendoring Petals "as a subtree so upstream
   changes can still be pulled in" is inheriting a maintenance burden, not sharing one.
2. **Do not pitch this as "reviving Petals."** The swarm is offline and the repo is two
   years stale. The audience most likely to care (r/LocalLLaMA, HN) will check within
   minutes, and being corrected in public on your launch post is expensive.

The honest framing is **rebuild the network, reuse the protocol, and add the trust layer
nobody built.** That is a better story anyway, and it is true.

Full evidence: `docs/findings-upstream-audit.md`.

---

## 0.5 Your actual hardware

**[CORRECTED — better than assumed.]**

*Originally believed:* laptop, phone, tablet; no usable GPU; client-role only.

*Actually true:* the development machine has an **RTX 3050 Laptop GPU (4GB VRAM, compute
8.6), 32GB RAM, 20 logical cores**, and WSL2 Ubuntu installed. That is enough to host blocks
of a small quantized model, not just to act as a client.

Three practical constraints the original spec did not mention:

- **Petals/hivemind do not run natively on Windows.** WSL2 is mandatory for the server role,
  not optional.
- **Python 3.13 will not work for the Petals side.** Its `numpy<2` pin has no cp313 wheels
  (NumPy 1.26.4 caps at 3.12), so that environment needs Python 3.10 or 3.11.
- **The Seedmesh trust layer has neither constraint** — it runs natively on Windows, on
  Python 3.13, with no GPU and no torch. That is by design (see §2.1).

**MIT cluster access** remains a genuine development upgrade, with the same two caveats as
the original draft (policy first; shared partitions are wrong for always-on roles). But the
*ask* has narrowed usefully — see §10.

**The practical split** (unchanged, still correct):
- **Bootstrap/seed peers**: cheap always-on VPS, ~$5–6/month, **no GPU needed** — a bootstrap
  peer only relays DHT metadata. This is what decouples the network from your student status.
  *Do this when the first backend lands, not before — a bootstrap peer for an empty swarm is
  just a bill.*
- **Server role during development**: short-lived Slurm jobs, treated as disposable.
- **Long-term capacity**: actual volunteers. That is the whole point.

---

## 1. Goals / non-goals

**Goals (v1):** unchanged.
- Anyone can `pip install` a client and either query the swarm or donate capacity to host a shard.
- No payment, no token. Contributing earns priority/reliability, not money.
- Public, permissionless: anyone can join without an invite.

**[CORRECTED]** The original goal "works today on the existing Petals/Hivemind protocol
against a *revived* swarm" is not achievable as stated — there is no swarm to revive, and no
backend adapter exists yet. Restated: *works on the hivemind protocol against a swarm we
stand up.*

**Non-goals (v1):** unchanged — no training/fine-tuning focus, no cryptographic
proof-of-computation, no mobile or browser node.

---

## 2. System architecture

The swarm topology is unchanged from the original draft: bootstrap peers for rendezvous,
server peers hosting block ranges, client peers assembling pipelines, gRPC activation
transport, DHT block announcements with TTLs handling churn.

```
                        ┌─────────────────────────┐
                        │  Bootstrap / DHT seed     │   (always-on, public IP,
                        │  peers (you run these)    │    no GPU — entry points,
                        └────────────┬─────────────┘    not authority)
                                     │
                     Kademlia DHT (hivemind), gossiped across all peers
                                     │
        ┌────────────────────────────┼────────────────────────────┐
┌───────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
│  Server node A   │          │  Server node B   │          │  Server node C   │
│  blocks 0-19     │◄────────►│  blocks 20-39   │◄────────►│  blocks 40-59   │
└───────▲────────┘  activa-  └────────▲────────┘  activa-  └────────▲────────┘
        │            tions             │           tions             │
        └─────────────── inference request ───────────────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │       Client node         │
                        │  + Seedmesh trust layer   │
                        └───────────────────────────┘
```

### 2.1 The backend seam **[NEW]**

Given §0, binding the project's novel work to an abandoned codebase would mean the trust
layer could not be developed, tested or trusted until an unbounded porting problem was
solved first. So the backend sits behind a **three-method interface**
(`discover`, `n_blocks`, `run_segment`) and nothing above it imports torch, transformers,
hivemind or gRPC.

```
   Seedmesh trust layer  (reputation + verification — pure Python/numpy)
             │  backends/base.py
   ┌─────────┼──────────────┬─────────────────┐
   │ Petals  │ llama.cpp RPC │  simulator      │
   │ adapter │   adapter     │  (implemented)  │
   └─────────┴───────────────┴─────────────────┘
```

This is why the trust layer is finished while the backend decision is still open. The
interface deliberately omits **prompt text** — servers see activations only, which makes
that a structural property rather than a convention.

---

## 3. What to actually add on top (the real v1 scope)

### 3.1 Reputation layer — **BUILT**

The original design (rolling reliability score, stored in the DHT, biasing routing) was
right in outline and had one hole:

**[CORRECTED]** *Originally proposed:* "Store scores in the DHT alongside block
announcements so any client benefits from the swarm's collective experience."

*Problem:* a DHT is an **open write surface**. Kademlia stores whatever it is handed, so an
unsigned reputation record there is worth nothing — anyone can publish "peer X is terrible"
or "peer Y (me) is perfect".

*As built:*
- Records are **Ed25519-signed**, with peer ids derived as hashes of public keys, so
  authorship is checkable with no registry or trusted introducer.
- Domain-separated, monotonic per-observer `epoch` (anti-replay/rollback), bounded TTL,
  self-reports dropped, size-capped.
- Aggregation applies six rules: one-observer-one-vote, **per-network-cluster weight caps**,
  weighted **median** (not mean — 50% breakdown point), collusion discount, corroboration
  required for accusations, and first-hand experience always dominating hearsay.

The key insight: **identities are free, network diversity is not.** Every anti-sybil rule
prices influence in clusters (ASN / address prefix), never in peer count.

*Measured:* 50 sybil observers insisting a bad peer is perfect, against 5 honest observers
rating it 0.20, move the aggregate only to **0.255**.

Scoring is `integrity × reliability × latency_factor`, where latency is bounded so it can
modulate but never dominate — **a fast liar must always rank below a slow honest node.**

### 3.2 Redundant-verification sampling — **BUILT, with the core method changed**

**[CORRECTED — this is the most important correction in the document.]**

*Originally proposed:* "route the same sub-computation to two servers and diff the outputs
(or a hash/fingerprint of the hidden state, not the full tensor, to keep overhead low)."

*Why that cannot work:* **floating-point addition is not associative.** Results depend on
reduction order, which depends on kernel choice, GPU architecture, dtype and batch shape.
An A100 in bf16 and a 3090 in fp16 computing the identical layer agree to a few decimal
places and differ in the low bits — **every single time.** Under exact hash comparison,
*every honest heterogeneous pair reads as a mismatch.* The check would have had a
~100% false-positive rate and evicted precisely the heterogeneous volunteers the network
exists to aggregate. This would not have surfaced until real mixed hardware joined.

*As built:* the bandwidth instinct was right, the hash was wrong.
- Project the hidden state onto `k` pseudo-random unit vectors and compare **those** floats.
  Johnson–Lindenstrauss preserves relative distances, so honest pairs stay close and
  fabricated output lands far away. A 64-component sketch replaces ~2M floats — the
  bandwidth saving is preserved.
- The seed derives from a **client nonce** bound to block range and step, so a server must
  commit to its output before it can learn the projection basis.
- Three verdicts, not two: `MATCH` / `MISMATCH` / **`INCONCLUSIVE`**. The third matters
  because the error costs are asymmetric — wrongly evicting a volunteer loses capacity and
  goodwill permanently; missing one cheat costs one request that sampling catches later.
- Thresholds are **calibrated from measured honest disagreement**, not guessed. Choosing the
  quantile *is* choosing the false-positive rate.
- A separate **passthrough check** catches a server echoing its input to skip computation —
  which redundant sampling alone would miss, because it looks like a *fast success*.

**[NEW — a protocol requirement the original spec did not anticipate.]** A server's
**compute profile** (quantization mode, dtype, attention kernel) determines the numerical
noise floor that thresholds are calibrated against, so it must be **published in the block
announcement** and the verification tolerance looked up per *pair* of profiles.

This is not a refinement — without it the design evicts honest volunteers. Measured
(§3.3a): honest servers with identical weights disagree by ~0.005 across fp16/bf16/int8 but
by **~0.055 when one is NF4**. Against a single global threshold, every NF4 comparison lands
in the ambiguous band, and the ambiguous-rate gate then quarantines the honest NF4 server
after **six** verifications — it is indistinguishable from a threshold-hugging attacker.

That matters most for the cheapest hardware: a 4GB laptop GPU hosts 28 NF4 blocks of an
8B-class model versus 7 at fp16. NF4 is what makes ordinary machines useful, so this bug
would have silently evicted precisely the volunteers the network depends on.

Implemented: `ComputeProfile` on `ServerInfo`, `ToleranceTable` keyed by unordered profile
pair, and an uncalibrated pair **refuses to verify** rather than guessing a threshold.

### 3.3 Modernized model catalog — **[CORRECTED] a port, not config**

*Originally believed:* "primarily: confirm architecture compatibility, publish
block-conversion/quantization config, update the tracked model list."

*Actually true:* Petals hard-pins `transformers==4.43.1` and hand-forks transformers'
attention body for Llama and Falcon. Current transformers is **5.14.1** — a major-version
boundary. Qwen3 and Llama 4 need ≥4.51.

A spike was run to measure this (`spike/transformers_port/`, reproducible, no GPU needed):

**Sub-correction:** an earlier draft of the audit said the imports had been *removed*. They
have not — all ten still resolve behind back-compat shims. The breakage is
**attribute- and signature-level**, which is worse: the module imports cleanly and dies on
the first forward pass with `AttributeError: 'OptimizedLlamaAttention' object has no
attribute 'num_heads'`.

**Spike result — better than feared.** The fix is *deletion*, not translation. Petals'
`OptimizedLlamaAttention` is a copied fork of transformers' attention carried only to splice
in two CUDA-graph optimizations that fire for single-token CUDA decode. A hosted block does
not need it — it needs to own its rotary embedding and causal mask, translate the cache, and
keep parameter names stable. Everything else delegates to stock `LlamaDecoderLayer`.

| | code lines |
| --- | --- |
| upstream `llama/block.py` | 221 |
| **ported equivalent** | **138** |
| verification | exact match on prefill (0.0), float32-epsilon on cached decode (2.4e-07), `state_dict` 0 missing / 0 unexpected |

**And Petals already agreed.** Its two most recently added architectures already delegate:
`mixtral` (92 lines, stock `MixtralDecoderLayer` + `DynamicCache`) and `bloom` (34 lines).
Only `llama` (221) and `falcon` (361) are legacy hand-forks. The port strategy is the one
Petals itself converged on before development stopped.

### 3.3a Quantization — **[RESOLVED] it works unmodified**

A second spike (`spike/quantization/`) tested the stale quantization stack. Result: better
than the first.

**bitsandbytes needs no port at all.** All four constructors Petals uses
(`Linear8bitLt`, `Int8Params`, `LinearNF4`, `Params4bit`) survive nine minor versions, and
Petals' *unmodified* `quantize_module` applies cleanly on 0.50.0. The reason is structural:
Petals **used** bitsandbytes' public API and **forked** transformers' internals. Public APIs
kept their promises; copied internals did not.

**`tensor_parallel` is deletable for the volunteer case.** Correcting an earlier claim in
this project: it is *not* multi-GPU-only — `convert_block` wraps every server in
`TensorParallel`, and the docstring says a single device is wrapped "for uniformity." It
also breaks independently, because `make_tensor_parallel` reads `submodule.num_heads`, the
same removed attribute that broke the Llama block. But since it is decorative for one
device, skipping it when `len(devices) == 1` drops the `tensor_parallel==1.0.23` pin
entirely and removes that breakage at zero cost.

**Memory footprint, measured** (validates the onboarding wizard's math):

| mode | bytes/param | Petals' constant |
| --- | --- | --- |
| fp16 | 2.000 | 2.0 |
| int8 | 1.000 | 1.0 — exact |
| nf4 | 0.516 | 0.531 — over-estimates by 3%, i.e. errs safe |

Blocks of an 8B-class model per volunteer GPU: **4GB → 7 fp16 / 14 int8 / 28 nf4**;
24GB → 56 / 113 / 219.

**And it broke verification** — see §3.2. That finding is the most valuable output of either
spike, and no upstream project would have looked for it, because only Seedmesh compares two
servers' numbers against each other.

### 3.3b DHT plumbing — **[RESOLVED] 7 import statements**

A third spike (`spike/hivemind_dht/`) tested Petals' hivemind coupling on a real 3-node DHT
in WSL. hivemind is the **healthiest dependency in the stack**: 1.1.12 shipped 2026-01-03,
and Petals pins a ~1.1.10-era SHA — two minor releases over 2.5 years, versus a
major-version boundary for transformers.

**The breakage is import paths and nothing else.** hivemind narrowed its top-level
namespace, so `from hivemind import PeerID` fails while `hivemind.p2p.PeerID` works. Seven
broken statements across **22 files**, purely mechanical. Every deep reach I expected to
break — `hivemind.moe.*` (which Petals repurposes for block hosting),
`p2p_daemon_bindings.control`, the module global Petals assigns to in `compression.base` —
**survives 6/6**. Exactly the opposite of the prediction.

**The DHT storage model already fits the reputation design.** hivemind gives each subkey its
own value *and its own expiration* under a shared key — one slot per observer, newest-wins,
independently TTL-expiring, which is precisely what §3.1 needs and required inventing
nothing. TTL expiry works, so churn handling is free. And 512-report batches (57KB) publish
and roundtrip with signatures still verifying — `MAX_REPORTS_PER_BATCH = 512` was assumed
without checking, and it holds.

**Confirmed by measurement: the DHT has no write protection by default.** Any peer can
overwrite any other peer's subkey, silently. `records.py` argued this to justify Ed25519
signing; it is now tested rather than asserted, and it means those signatures carry 100% of
the attribution guarantee. hivemind's own `RSASignatureValidator` *does* block the hijack
when enabled — use both, since they protect different things: hivemind secures the DHT
*slot*, Seedmesh's signature secures the *content* and travels with it, verifiable offline,
carrying the anti-rollback epoch.

**What is still unassessed:** whether Petals' server/client *logic* — `RemoteSequential`,
the sequence manager, the inference session protocol — works end-to-end once the imports are
fixed. That needs a running two-node swarm, not another compatibility probe.

### 3.3c Backend decision — **[RESOLVED] Petals for v1, llama.cpp for private meshes**

**[CORRECTED]** *Originally believed (§9 of revision 1, and repeated in revision 2):*
llama.cpp RPC's broader hardware reach probably outweighs Petals' PyTorch-only limitation,
because "most volunteers will not have an NVIDIA GPU."

*That claim was wrong as stated.* The Steam Hardware Survey (June 2026) puts NVIDIA at ~72%
of surveyed GPUs. Among gaming PCs with discrete GPUs, most volunteers **would** have NVIDIA.
The real reach argument is different and still real: llama.cpp opens pools Petals cannot
reach *at all* — Apple Silicon (unified memory makes an ordinary Mac a good inference host),
AMD, Intel, mobile, and plain CPU.

But a fourth spike (`spike/llamacpp_rpc/`) measured the thing that actually decides it. An
anonymous client with a raw TCP socket, no credential of any kind:

```
HELLO                -> protocol v5.0.0
DEVICE_COUNT         -> 1 device(s) enumerated
GET_DEVICE_MEMORY[0] -> free 15.62 GiB / total 15.62 GiB
ALLOC_BUFFER(256MiB) -> granted, remote ptr 0x609bd8e6a2e0
```

The unauthenticated command set also includes `SET_TENSOR` / `GET_TENSOR` (write and read
host memory) and `GRAPH_COMPUTE` (**execute a client-supplied compute graph**). There is no
authentication, no authorization, and `GET_MAX_SIZE` returns effectively `SIZE_MAX`, so
uncapped allocation is a one-line DoS against a volunteer.

**This breaks §7's core promise to volunteers.** "Hosting a block does not let peers run
arbitrary code on your machine" is true for Petals *because of an architectural property* —
the server owns the weights and the graph, the client supplies only an input tensor.
llama.cpp RPC inverts that: the client owns the model and submits the graph, and the server
executes what it is sent. A volunteer on a public network would not have donated a shard;
they would have donated a programmable accelerator to strangers.

None of this is a criticism of llama.cpp, whose README says exactly this and which is
maintained far better than Petals (commits dated today, versus 23 months dead). It is a
statement about fit.

**Decision.** The split maps onto the product question the research doc flagged as
unresolved — shared public swarm, or exo-style "mesh my own devices":

- **Public permissionless swarm → Petals + hivemind.** The only architecturally public-safe
  option, and all three revival spikes showed the cost is bounded and mostly mechanical.
- **Private / LAN mesh → llama.cpp RPC**, as a second backend behind the same seam. Trusted
  network, any hardware, single binary, no discovery needed — and the security model that
  disqualifies it publicly does not apply there. This is also the lowest-risk path to Mac
  and CPU volunteers later.

**Launch model set** — pick 2–3, bias toward unambiguously permissive licenses. See
`seedmesh/models/registry.yaml`, which deliberately excludes Llama-family models as default
network citizens pending a real license read, and notes that `qwen2.5-7b-instruct` is the
pragmatic option because it is Apache-2.0 *and* within the existing transformers pin.

### 3.4 Onboarding polish — not started
One-command install with hardware auto-detection, and a public leaderboard/thank-you page.
`seedmesh probe` exists as a stub; block-count recommendation needs the backend adapter to
know per-block memory footprint.

### 3.5 Governance & comms — **BUILT (docs)**
`GOVERNANCE.md` exists with the explicit non-monetization pledge. `CONTRIBUTING.md` exists.
Discord/Matrix not yet created — correctly, since there is nothing to run yet.

---

## 4. Repo structure

**[CORRECTED — reflects what exists.]** The original structure assumed a vendored Petals
subtree at the centre. Actual structure inverts that: the trust layer is the centre and
backends are pluggable.

```
seedmesh/
├── README.md, GOVERNANCE.md, CONTRIBUTING.md, LICENSE (MIT), pyproject.toml
├── seedmesh/
│   ├── core/            identity (Ed25519, canonical signing), types, injectable clock
│   ├── reputation/      scorer, records, aggregate, diversity, routing_bias
│   ├── verification/    sketch, compare, calibrate, sampler
│   ├── backends/        base.py — the three-method seam
│   ├── sim/             world, scenarios, presets, report
│   ├── cli/             simulate, probe
│   └── models/          registry.yaml
├── spike/transformers_port/   port spike: probe, ported block, equivalence harness
├── tests/               138 tests
└── docs/
    ├── architecture.md            layers, and why each defence exists
    ├── threat-model.md            defended / not defended / open
    ├── security-privacy.md        for volunteers and for users
    ├── findings-upstream-audit.md verified state of Petals and hivemind
    └── outreach-drafts.md         ORCD + TLO email drafts
```

---

## 5. Milestones

**[CORRECTED — reordered.]** The original order was M0 fork → M1 trust layer → M2 gateway.
That order is now impossible (nothing to fork against, no swarm to test on) and was also
backwards: the trust layer is the novel work and it does not depend on the backend.

**M0 — Trust layer against a simulator — ✅ DONE**
Reputation, verification, routing, signed records, sybil-resistant aggregation, and a
deterministic adversarial simulator. 138 tests. Measured:

| scenario | cheats caught | honest wrongly excluded | malicious traffic, 2nd half |
| --- | --- | --- | --- |
| healthy (mixed hardware) | — | 0 | 0 |
| mixed threats | 3/3 | 0 | 0 |
| sybil fleet (12 identities, 1 ASN) | 12/12 | 0 | 0 |

*Simulator numbers on a synthetic topology — they prove the design is internally sound and
that specific attacks fail. They are not measurements of a real swarm.*

**M0.5 — Port spike — ✅ DONE**
Llama block ported to transformers 5.14.1, numerically exact. Recipe and harness in
`spike/transformers_port/`.

**M1 — Quantization spike — ✅ DONE**
bitsandbytes works unmodified; `tensor_parallel` is deletable for single-GPU; memory
constants validated. Also found and fixed a defect that would have quarantined honest NF4
volunteers (§3.2, §3.3a).

**M2 — Hardware calibration (~1 week, needs several GPU models)**
Re-fit the tolerance table with real hardware diversity. The current table measures the
*quantization* component only — it was collected on one GPU, so every same-mode pair reads
exactly 0.0, which is a floor artefact rather than a measurement. **This is the one
remaining task that genuinely needs GPU access**, and a month of Colab supplies it: session
variability hands you hardware diversity for free, and Seedmesh fingerprints are 64 floats,
so cross-session comparison is a few KB of JSON. No cluster required.

**M3 — First adapter — ✅ DONE**

Petals ported (`tools/port_petals.py`, 7 patches), a private two-server swarm generates real
tokens, churn survived (SIGKILL a serving node → identical output via the redundant one),
and `seedmesh/backends/petals_backend.py` drives it all through the trust layer: discovery,
reputation-biased routing, per-server segment execution, and a real verification comparison.
Full detail in `docs/petals-port.md`.

**Peer addresses now resolve**, via hivemind's `P2P.list_peers()`, so the diversity
constraint has real input and the sampler drives verification end to end on live servers.
On a single-host swarm it still (correctly) judges peers non-independent — they share a
/16 — and separates them once an ASN table is supplied, which is what a deployment's offline
GeoIP lookup provides.

**The attention kernel is now published too** (patch 8), so a server's full compute profile
— quant type, dtype, kernel — travels in its DHT record, which is exactly what §3.2 requires
for per-pair tolerance selection.

**Remaining gaps, in priority order:**
- **Every threshold is still fitted to simulated noise.** `tools/calibrate/` is built and
  smoke-tested on a real GPU; it needs Colab sessions across several GPU types (M2). This is
  now the only thing between "verification runs" and "verification means something".
- **No real ASN table** is wired into `ClusterIndex` yet — the demo uses a simulated one.

**llama.cpp RPC as a second backend later**, behind the same seam, for private/LAN meshes —
where it is nearly ideal and where its security model is not disqualifying.

**M3.5 — Private multi-node swarm (2–3 weeks)**
Real inference end-to-end across separate processes/nodes, with deliberate mid-request
kills to confirm rerouting. Needs **several network endpoints, not GPU power** — runs as
local processes or $5 VPSs.

**M4 — Onboarding, gateway, monitor (2–3 weeks)**
Hardware auto-detection wizard, HTTP/WebSocket gateway, public monitor showing reputation.

**M5 — Public alpha**
Bootstrap peers public. Post to r/LocalLLaMA, HN, Petals issues. **Not before M2 at the
earliest** — launching with no working backend spends the one-time attention of exactly the
audience you need, on a repo they cannot use.

---

## 6. Coding-agent kickoff prompt

**[CORRECTED — superseded twice.]** The original prompt ("fork Petals, get a 3-node local
swarm running, don't build reputation yet") is obsolete, and so is its quantization-spike
replacement — both spikes are done. The next task is hardware calibration:

> Collect honest-pair calibration data across **several different GPU models**, using
> Google Colab sessions (each session gives whatever GPU is available; record it with
> `nvidia-smi`). For each session, run the fixed reference inputs through a transformer
> block held at fp16, bf16, int8 and NF4, and save the resulting sketches — they are 64
> floats each, so the artefact is a few KB of JSON per session. Reuse
> `spike/quantization/calibrate_cross_quant.py`, which already does this for one GPU;
> generalise it to merge sketch files across sessions and fit a `ToleranceTable` over the
> full (profile, hardware) cross-product. The current table's same-mode entries read
> exactly 0.0 because they were collected on one GPU — those are floor artefacts and are
> the specific thing this run must replace. Save with provenance via
> `ToleranceTable.save(..., notes=...)`.

---

## 7. Security/privacy notes

Both original points still stand and are now documented in `docs/security-privacy.md`:

- Hosting a block does **not** let peers run arbitrary code on your machine — they send
  tensors through layers you host. Note the caveat the original spec omitted: this does not
  cover bugs in the **deserialization path**, which is untrusted-input attack surface owned
  by the backend adapter and needs review before launch.
- Prompts are visible in intermediate form to servers. Say this **loudly**. Activations are
  not plaintext, but recovering input from hidden states is an active research area, so
  treating them as safe would be wrong.

**[NEW]** A third point the original missed: **verification checks agreement, not
correctness.** A server running genuine weights that produce subtly biased output is
indistinguishable from an honest one by construction. And if every server ran the same
subtly-wrong port, they would all agree — which is why §3.3's numerical equivalence testing
is a *security* requirement, not just a quality one.

---

## 8. Where forking doesn't cover you

Unchanged and confirmed. The reputation/verification layer was genuinely new work — and
building it surfaced three defects that only appeared under adversarial simulation:

1. **Charging both parties of a mismatch at full weight** evicted the honest node that
   caught the cheat (50% of quarantines were honest). Verifying became dangerous.
2. **Counting accusers by peer id** let a 12-identity fleet manufacture accusers — it
   convicted 4 honest servers and 0 of its own. Fixed by attributing disagreements to
   *network cluster*.
3. **Absolute accumulation of ambiguous results** created a self-fulfilling loop: suspicion
   raised the sampling rate, extra sampling turned up rare honest ambiguity, the node
   convicted itself. Fixed by gating on *rate*, not count.

None were visible by inspection. This is the strongest argument for the simulator existing.

---

## 9. Open ends

**RESOLVED since revision 1:**
- ~~Sybil/collusion resistance in verification pair selection~~ — implemented (different
  network cluster + first-seen gap + rotating pairings). Was flagged in the original as
  "add before this ships"; it shipped with it.
- ~~Backend choice: Hivemind/Petals isn't the only option~~ — **decided** (§3.3c). Petals +
  hivemind for the public swarm; llama.cpp RPC as a second backend for private/LAN meshes.
  The original spec was right to flag llama.cpp as worth a real evaluation, and right that
  hardware reach is its advantage. It was wrong to treat that as the deciding axis: the RPC
  protocol is unauthenticated by design and inverts the volunteer safety guarantee.

**STILL OPEN, unchanged from revision 1:**
- **Model licensing.** Lead with unambiguously permissive licenses. `registry.yaml` excludes
  Llama-family models as defaults pending a real license read.
- **Node-operator liability and abuse content.** The architecture's saving grace — servers
  see activations, never plaintext — is now structurally enforced by the backend interface
  omitting text. The exposed point remains the **gateway**, which does see plaintext.
- **Export controls.** A clear AUP is cheap insurance; enforcement in P2P is inherently weak.
- **EU AI Act** — hosting an unmodified model does not make an operator its "provider".
- **MIT IP policy** — see §10, now with a draft email.
- **Browser/WebGPU nodes** — v2 growth lever. Note they will have different noise
  characteristics and no stable network identity, so both **calibration and clustering** need
  rethinking before admitting them.
- **Launch-day graceful degradation** — honest queueing and wait-time estimates, not silent
  failure.

**NEWLY OPEN:**
- **Whitewashing.** A caught peer can discard its identity and rejoin fresh; identities are
  free. The prior (0.80, below a proven peer) makes this mildly unattractive but does not
  prevent it. Real defences are proof-of-work or stake at identity creation, both of which
  this project has deliberately refused. **Probably the most exploitable gap for a determined
  individual attacker.**
- **Calibration drift.** Thresholds are fitted to a hardware mix that changes as the swarm
  grows. Recalibration must be periodic, and must run on a population *known to be honest* —
  fitting on a population containing cheats widens tolerance until they fit inside it.
- **Bootstrapping the first honest observers.** Aggregation assumes some exist. At launch
  none do. Whether early operator-run nodes should be weighted more is a governance question
  with real centralization cost.
- **Attention-kernel publication** (§3.2) — a protocol addition, not yet in any DHT record
  schema.

---

## 10. Actions requiring you personally **[NEW]**

Drafts and full framing in `docs/outreach-drafts.md`.

**1. Colab instead of the cluster — likely the whole answer.** The two things the cluster
was wanted for split cleanly, and neither needs MIT:

| need | actually requires | option |
| --- | --- | --- |
| verification calibration (M2) | *diversity* of GPU models | **Colab Pro, ~$10/mo** |
| multi-node churn testing (M3.5) | several network endpoints, **no GPU** | local processes or $5 VPSs |

Colab's session-to-session GPU variability — normally its weakness — supplies exactly the
hardware diversity calibration needs, and Seedmesh fingerprints are 64 floats, so you never
need two GPUs at once. Caveat: keep it to short interactive runs; nothing long-lived should
live on Colab.

**2. MIT ORCD — `orcd-help-engaging@mit.edu`.** Now *optional*. Send only if you want
cluster access for convenience. If you do, the ask is much narrower than originally planned:
cluster-internal, non-persistent, no outside traffic.

**3. MIT TLO — `tlo@mit.edu`.** Much lower stakes if you go the Colab route, and arguably
unnecessary — no MIT facilities used means the question largely evaporates. If you want it
on record anyway, send the *confirmation* framing ("I intend to use only personally-funded
commercial compute") rather than a judgement call about thresholds; it is far easier to
answer quickly. In your favour either way: the entire trust layer was written on personal
hardware, and the git history shows it. Drafts in `docs/outreach-drafts.md`.

**4. Bootstrap VPS** — ~$5–6/month, no GPU. Defer until M3; a bootstrap peer for an empty
swarm is just a bill.

**5. Do not launch publicly yet.** Not before M3. There is no backend adapter, so nobody can
run anything.

---

## Sources

Verified 2026-07-31 unless noted.

- [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals) — `pushed_at` 2024-09-07
- [petals `setup.cfg`](https://raw.githubusercontent.com/bigscience-workshop/petals/main/setup.cfg) — dependency pins
- [petals `llama/block.py`](https://raw.githubusercontent.com/bigscience-workshop/petals/main/src/petals/models/llama/block.py) — forked attention
- [learning-at-home/hivemind](https://github.com/learning-at-home/hivemind) — `pushed_at` 2026-01-11
- [hivemind DHT docs](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html)
- `health.petals.dev` — connection refused; `chat.petals.dev` — capacity notice
- [transformers v5.14.1 `modeling_llama.py`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/llama/modeling_llama.py) — current signatures
- [About the Engaging Cluster — MIT ORCD](https://orcd.mit.edu/resources/about-engaging-cluster)
- [MIT Policies 13.1 — Intellectual Property](https://policies.mit.edu/policies-procedures/130-information-policies/131-intellectual-property)
- [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)
- [llama.cpp RPC backend README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- [General-Purpose AI Models in the AI Act — European Commission](https://digital-strategy.ec.europa.eu/en/faqs/general-purpose-ai-models-ai-act-questions-answers)
