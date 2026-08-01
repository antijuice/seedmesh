# Seedmesh architecture

Seedmesh is a **trust layer for public peer-to-peer inference swarms**. It answers two
questions that the underlying P2P transport does not:

1. *Which servers should I send work to?* (reputation)
2. *Did they actually do it?* (verification)

Everything else — DHT, block sharding, activation transport — is the backend's job, and
Seedmesh deliberately does not reimplement any of it.

---

## 1. Where this sits

```
        ┌───────────────────────────────────────────────────────────┐
        │                      Seedmesh trust layer                   │
        │                                                             │
        │   reputation/            verification/                      │
        │     scorer      ◄──────►   sampler    (who checks whom)     │
        │     records                sketch     (cheap fingerprints)  │
        │     aggregate              compare    (tolerance, verdicts) │
        │     diversity    ◄───────  calibrate  (thresholds from data)│
        │     routing_bias                                            │
        └────────────────────────────┬──────────────────────────────┘
                                     │  backends/base.py
                     discover() · n_blocks() · run_segment()
                                     │
        ┌────────────────────────────┼──────────────────────────────┐
        │                            │                              │
   ┌────▼─────┐              ┌───────▼──────┐              ┌────────▼──────┐
   │ Petals    │              │ llama.cpp RPC │              │  simulator     │
   │ adapter   │              │ adapter       │              │  (implemented) │
   │ (todo)    │              │ (evaluating)  │              │                │
   └───────────┘              └───────────────┘              └────────────────┘
```

The swarm topology itself is unchanged from the spec: bootstrap peers for rendezvous,
server peers hosting block ranges, client peers assembling pipelines. Seedmesh changes what
the *client* does when choosing among servers, and what it does afterwards to check them.

## 2. The backend seam

`seedmesh/backends/base.py` is three methods wide: `discover`, `n_blocks`, `run_segment`.
Nothing above it imports torch, transformers, hivemind or gRPC.

This is a direct response to evidence, not a stylistic preference. Petals has been
unmaintained since 2024-09-07, hard-pins `transformers==4.43.1`, and subclasses transformers
internals that the 4.48+ attention refactor rewrote (see
[findings-upstream-audit.md](findings-upstream-audit.md)). Binding the project's only novel
work to that codebase would mean the trust layer could not be developed, tested or trusted
until an unbounded porting problem was solved first.

With the seam, the porting decision stays open, and the trust layer is testable today
against a simulator that exercises it far harder than an early real swarm would.

One thing the interface deliberately omits: **prompt text.** Servers see activations only.
Keeping text out of the signature makes that a structural property rather than a convention.

## 3. Reputation

### 3.1 Local scoring — `reputation/scorer.py`

```
score = integrity × reliability × latency_factor
```

* **reliability** — Beta posterior over successes and failures, exponentially decayed
  (6h half-life). A prior means one lucky success does not outrank a long clean record, and
  confidence is tracked separately from score.
* **integrity** — verification history, decayed ~120× slower (30d half-life). It
  *multiplies*, so it acts as a gate: a caught cheat cannot launder a proven mismatch by
  serving a burst of cheap successful requests.
* **latency_factor** — bounded to `[1 − latency_weight, 1]`. Latency modulates ranking but
  can never dominate it, which encodes the ordering a trust layer needs: **a fast liar must
  always rank below a slow honest node.**

`Outcome.REFUSED` is scored as neither success nor failure. A server that cleanly refuses
because it is full is behaving correctly; punishing that teaches operators to accept work
they cannot do.

### 3.2 Sharing — `reputation/records.py`

Reputation travels as **signed observation batches**. The spec proposed putting scores in
the DHT; a DHT is an open write surface, so unsigned records there are worth nothing.

Every batch is Ed25519-signed, domain-separated, carries a monotonic per-observer `epoch`,
has a bounded TTL, drops self-reports, and is size-capped. Peer ids are hashes of public
keys, so authorship is checkable with no registry.

**Verified against a real DHT** (`spike/hivemind_dht/`): on a default hivemind DHT any peer
can silently overwrite any other peer's record, so these signatures carry the whole
attribution guarantee. hivemind's `RSASignatureValidator` blocks the hijack at the storage
layer and should also be enabled — it secures the *slot*, while the Ed25519 signature
secures the *content* and stays meaningful once a record is relayed or cached elsewhere.

The storage shape needed no invention: hivemind gives each subkey its own value and its own
expiration under a shared key, which is exactly one slot per observer, newest-wins,
independently expiring. 512-report batches (57KB) publish and roundtrip intact.

Attribution is not truth. A verified batch means *this observer really said this* — nothing
more.

### 3.3 Aggregation — `reputation/aggregate.py`

Turning strangers' claims into one number is where a naive design gets stuffed. Six rules:

1. One observer, one vote — influence is capped per observer, and only the newest batch counts.
2. Per-cluster weight caps — since identities are free but network diversity is not. A
   peer's cluster comes from an ASN resolved *locally*, never from anything the peer
   asserts: `TableAsnResolver` answers from an offline ip2asn table (573k routed ranges,
   ~3.4s to load, sub-microsecond cached lookups) fetched by `tools/fetch_asn_table.py`.
   Offline is a requirement, not a convenience — a DNS or whois lookup on the routing path
   would add latency to every decision and let an attacker stall routing by stalling the
   lookups. Addresses with no AS (private, loopback, unrouted) fall back to prefix
   clustering.
3. Weighted **median**, not mean — 50% breakdown point, so an attacker must control a
   majority of *weight*, which rule 2 has already made expensive.
4. Collusion discount for observers vouching for peers in their own cluster.
5. Accusations need corroboration from independent clusters — defaming an honest node must
   not be cheaper than praising yourself.
6. First-hand experience dominates hearsay, in proportion to how much you have measured.

### 3.4 Routing — `reputation/routing_bias.py`

Pipeline selection is a shortest-path problem, solved exactly rather than greedily, because
the two goals the spec asks for — fewer hops, higher reputation — trade against each other
and greedy per-range selection cannot express the trade.

Since hop cost is linear in blocks covered, an optimal segmentation always exists with
breakpoints at some server's start or end, so the search over that finite boundary set is
exact.

Two terms exist because the simulator showed the design failing without them:

* **Exploration bonus** (UCB-style, decays as `1/√n`). Without it, routing always prefers
  proven servers, so a new volunteer never gets traffic, never accumulates history, and
  stays unknown forever. The network cannot absorb new capacity.
* **Load term.** Without it the swarm collapses onto whichever few servers offer the
  shortest path (measured Gini 0.83, two donors carrying nearly everything). That burns out
  the people the network depends on — and starves verification, because a server that is
  never routed to is never checked.

The load term costs latency. That trade is measured and documented in `RoutingConfig`.

## 4. Verification

### 4.1 Fingerprints, not hashes — `verification/sketch.py`

The spec proposes comparing "a hash/fingerprint of the hidden state". **Exact hashing cannot
work.** Floating-point addition is not associative, so results depend on reduction order,
which depends on kernel, architecture, dtype and batch shape. Two honest servers — an A100
in bf16 and a 3090 in fp16 — agree to a few decimals and differ in the low bits, every
time. Under hash comparison every honest heterogeneous pair reads as a mismatch, and
heterogeneous hardware is the entire premise of a volunteer swarm.

Instead: project the hidden state onto `k` pseudo-random unit vectors and send those `k`
floats. Johnson–Lindenstrauss preserves relative distances, so honest pairs stay close and
fabricated output lands far away. A 64-component sketch replaces ~2M floats.

The seed is derived from a client nonce and bound to the block range and step, so a server
must commit to its output before it can learn the projection basis.

### 4.2 Three verdicts — `verification/compare.py`

`MATCH` / `MISMATCH` / `INCONCLUSIVE`. The third is the important one. Binary decisions on a
noisy measurement charge every borderline case as fraud, and the error costs are asymmetric:
wrongly evicting a volunteer loses capacity and goodwill permanently, while missing one
cheat costs one request that sampling will catch again.

`MISMATCH` rests on relative distance alone. Earlier revisions also convicted on multiples
of the cosine and norm thresholds, which let samples inside the ambiguous band be charged
anyway — the band stopped meaning what its name says.

### 4.3 Calibration — `verification/calibrate.py`

Thresholds are **measured, not guessed**. Collect distances from pairs known to be honest,
set the match threshold at a high quantile, and the choice of quantile *is* the choice of
false-positive rate. A threshold fitted on one hardware mix is not valid on another, so
calibrations are saved with provenance.

Calibration must run on an all-honest population. Fitting on a population containing cheats
widens the tolerance until the cheats fit inside it.

**Tolerance is per compute-profile-pair, not global.** Measured across five real GPU
architectures (`tools/calibrate/`): honest servers holding identical weights disagree by at
most **0.0014** when hardware differs but precision matches, and by **0.033** when one of
them is NF4 — roughly 23x more. Quantization dominates hardware, and no single global
threshold covers both.

Against a single global threshold fitted to the tighter modes, every NF4 comparison lands in
the ambiguous band, and the ambiguous-rate gate (§5) then quarantines the honest NF4 server
after six checks. It looks exactly like a threshold-hugging attacker, because "consistently
just inside the band, across independent clusters" describes both.

So `ToleranceTable` is keyed by the unordered pair of `ComputeProfile`s, servers publish
their profile in the block announcement, and **an uncalibrated pair refuses to verify** —
the same call the sampler makes when no independent verifier exists, for the same reason: a
check judged against an invented threshold manufactures evidence the reputation layer will
act on.

This matters most for the cheapest hardware. A 4GB laptop GPU hosts 28 NF4 blocks of an
8B-class model versus 7 at fp16 — NF4 is what makes ordinary machines useful, so a design
that quietly evicts NF4 servers evicts the volunteers the network most needs.

Thresholds are no longer placeholders: `sessions/tolerance_table.json` is fitted from real
measurements across T4, A100, RTX 3050, L4 and Blackwell hardware.

### 4.4 Sampling — `verification/sampler.py`

Adaptive rate (high for unproven or suspect peers, ~3% at steady state), and — closing the
spec's own §9 open question — verifiers must be in a **different network cluster** and have
a first-seen gap. Pairings rotate so no relationship becomes predictable.

When no independent verifier exists, the answer is *don't verify*. Checking against a
possible sock puppet is worse than not checking, because it manufactures evidence of
integrity that the reputation layer will then trust.

## 5. How the two layers meet: the corroboration rule

A mismatch proves **one of two peers is wrong**. It does not say which. This is the single
subtlest part of the design, and simulation caught two ways of getting it wrong:

* **Charging both parties at full weight** evicted the honest node that caught the cheat
  (measured: 50% of quarantines were honest). That makes verifying dangerous, which is worse
  than not verifying.
* **Counting distinct accusers by peer id** let a 12-identity fleet manufacture twelve
  accusers, convicting four honest servers and none of its own members.

The rule that works: disagreements are recorded against both parties, **attributed to the
partner's network cluster**, and exclusion requires accusation by *two independent clusters*
— or a fault the client witnessed unilaterally, such as a server echoing its input.
A cheat disagrees with every cluster it meets; its accuser disagrees only with the cheat's.

Ambiguous-band results accumulate as fractional evidence, but only once a peer's ambiguous
*rate* is anomalous. Gating on rate rather than count matters: absolute accumulation created
a self-fulfilling loop where extra scrutiny convicted honest peers.

## 6. Measured behaviour

`seedmesh simulate`, 400 requests per preset. Real arrays, real sketches, real comparisons —
only the network and GPUs are simulated.

| Preset | Cheats caught | Honest wrongly excluded | Malicious traffic, 2nd half | Load Gini |
| --- | --- | --- | --- | --- |
| healthy (14 honest, mixed hardware) | — | **0** | 0 | 0.54 |
| mixed threats (lazy/byzantine/subtle/flaky) | **3/3** | **0** | 0 | 0.71 |
| sybil fleet (12 identities, one ASN) | **12/12** | **0** | 0 | 0.70 |

Gossip resistance: 50 sybil observers claiming a bad peer is perfect, against 5 honest
observers rating it 0.20, settle at **0.255** — the fleet does not move it.

These are simulator numbers on a synthetic topology. They demonstrate the design is
internally sound and that specific attacks fail against it. They are **not** measurements of
a real swarm, and the constants they justify need re-deriving once one exists.

## 7. What is not built

* No backend adapter — no real inference runs yet.
* No DHT integration — records are designed for it, not yet published to it.
* No gateway, monitor, or onboarding wizard.
* No training or fine-tuning (a spec non-goal).
