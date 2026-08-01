# Threat model

What Seedmesh's trust layer defends against, what it does not, and what remains open. The
"not" and "open" sections are the ones worth reading — a threat model that only lists
victories is marketing.

Scope: the reputation and verification layers. Transport-level attacks (DHT eclipse,
routing table poisoning, NAT abuse) belong to hivemind and are noted but not addressed here.

---

## Adversary model

Assumed capabilities:

* Generate unlimited peer identities (keypairs are free).
* Run arbitrary code on their own nodes; return anything they like.
* Observe everything they receive, including activations routed through them.
* Coordinate freely among nodes they control.
* Acquire moderate network diversity (a few /16s, a handful of ASNs, some cloud regions).

Assumed limits:

* Cannot forge Ed25519 signatures or find SHA-256 preimages.
* Cannot cheaply acquire *large-scale* network diversity — many hundreds of unrelated ASNs.
* Cannot control the client's own code or its first-hand measurements.

That middle limit is the one everything rests on. **Identities are free; network diversity
is not.** Every anti-sybil rule prices influence in clusters rather than peers.

---

## Defended

### T1 — Forged reputation records
Publishing "peer X is terrible" or "peer Y (me) is perfect" as someone else.

*Defence:* Ed25519 signatures with peer ids derived from public keys, so authorship is
self-certifying. The peer-id binding check is explicit, catching the classic near-miss where
a record carries the attacker's key and a valid signature while claiming another author.
Domain separation stops a signature being replayed as a different record type.

*Measured, not assumed:* on a default hivemind DHT, any peer can silently overwrite any
other peer's record — verified on a real 3-node DHT (`spike/hivemind_dht/`). So these
signatures carry **100% of the attribution guarantee**; there is no storage-layer fallback
underneath them.

*Defence in depth:* hivemind's `RSASignatureValidator` does block the hijack when enabled,
and should be. The two are complementary rather than redundant — it secures the DHT *slot*,
while the Ed25519 signature secures the *content*, travels with the record, is verifiable
offline by anyone who received it through any channel, and carries the anti-rollback epoch
that slot ownership does not address.

### T2 — Replay and rollback
Re-publishing a captured batch, or reverting to an older, more flattering one.

*Defence:* monotonic per-observer `epoch`, bounded TTL, rejection of future-dated records.
A rejected batch does not consume epoch space, so a malformed record cannot lock its author
out of publishing valid ones.

### T3 — Sybil vote-stuffing
Thousands of identities flooding favourable reports about a colluding peer.

*Defence:* per-observer caps, per-cluster weight caps, weighted median. Measured: 50 sybils
against 5 honest observers moved the aggregate from 0.20 to 0.255.

### T4 — Self-verification
An operator running two nodes and having one "verify" the other.

*Defence:* verification pairs must be in different network clusters with a first-seen gap.
When no independent verifier exists, no verification happens — checking against a possible
sock puppet manufactures evidence of integrity, which is worse than no evidence.

### T5 — Fabricated or lazy computation
Returning garbage, or echoing the input to skip the forward pass entirely.

*Defence:* redundant sampling with tolerance-based sketch comparison, plus a dedicated
passthrough check. Passthrough matters separately: redundant sampling alone would miss it,
because a server that echoes its input looks like a *fast success*. Being witnessed
directly by the client, it needs no corroboration.

### T6 — Defaming honest competitors
Using verification to evict good nodes rather than bad ones.

*Defence:* the corroboration rule. Exclusion requires accusation from two independent
clusters. Both simulator-caught failures here were real: charging both parties at full
weight made 50% of quarantines honest, and counting accusers by peer id let a 12-identity
fleet convict four honest servers.

### T7 — Threshold-hugging corruption
Corrupting output by just under the mismatch bound, which evades any single-sample test.

*Defence:* ambiguous-band results accumulate as fractional evidence, gated on the peer's
ambiguous *rate*. Calibration puts an honest peer near ~0.1%; a peer above 25% is not
unlucky. Gating on rate rather than count is essential — absolute accumulation created a
self-fulfilling loop where extra scrutiny convicted honest peers.

### T8 — Misdeclared compute profile
Claiming a different quantization mode, dtype or attention kernel than you actually run, to
get judged against a more favourable tolerance.

*Defence:* the incentive points the wrong way for the liar. The declared profile selects the
tolerance for **their own** comparisons, so claiming NF4 while running fp16 buys a uselessly
wide band that catches nothing they wanted hidden, and claiming fp16 while running NF4 fails
every check they take part in. Misdeclaration is self-punishing rather than profitable,
which is a stronger position than trying to detect it.

*Residual:* an attacker could declare NF4 (widest tolerance) and then corrupt output by an
amount that hides inside NF4's honest band — roughly 12x more room than the fp16 band
allows. Cross-quantization verification is genuinely weaker verification. Preferring
same-profile verifiers limits the exposure; it does not remove it.

### T9 — Cold-start starvation and load collapse
Not an attack, but a failure mode that silently makes the network fragile and unverifiable.

*Defence:* exploration bonus and load-aware routing. Without them, traffic collapses onto a
few servers (Gini 0.83) and the rest are never sampled.

---

## Not defended

### N1 — Colluding operators with genuine network diversity
An adversary with nodes in many unrelated ASNs defeats every cluster-based rule here. The
constraints make sybil attacks *expensive*, not impossible; a well-resourced attacker buys
past them.

### N2 — Nonce-correlating collusion
Two colluding servers with a side channel can notice they received the same nonce for the
same block range and agree on a fabricated answer. The diversity constraint makes forming
such a pair expensive; nothing here makes it impossible. This is inherent to redundant
sampling and is what cryptographic proof-of-computation (TOPLOC-style) would eventually
address.

### N3 — Prompt privacy
Servers see intermediate activations. Activations are not plaintext, but they are not
nothing either — inversion attacks against hidden states are an active research area. The
honest statement is **the swarm is not private**, and the README says so rather than burying
it. Sensitive data belongs on a private swarm.

### N4 — Gateway-level abuse
The client-facing gateway sees plaintext and is the natural point for abuse reporting and
policy. Out of scope for this layer, and a genuine open question for the project (spec §9).

### N5 — Transport attacks
DHT eclipse, routing-table poisoning, resource exhaustion at the network layer. hivemind's
problem. Worth a separate review before public launch.

### N6 — Model-level attacks
A server hosting genuine weights that produce subtly biased output within numerical
tolerance is indistinguishable from an honest server, by construction. Redundant execution
verifies *agreement*, never *correctness*.

---

## Open questions

**Cluster-wide punishment.** When several members of one cluster are proven bad, should the
whole cluster be down-weighted? Tempting and dangerous — a legitimate shared host or
university network would be collateral damage. Deliberately not implemented.

**Calibration drift.** Thresholds are fitted to a hardware mix that changes as the swarm
grows. Stale thresholds silently become either permissive or hostile. Recalibration needs to
be periodic, and running it on a population that already contains cheats would widen the
tolerance until they fit inside it.

**Bootstrapping the first honest observers.** Aggregation assumes some honest observers
exist. At launch there are none, so early clients run on first-hand experience alone. This
is correct but slow; whether early operator-run nodes should be weighted more is a
governance question with real centralization cost.

**Whitewashing.** A caught peer can discard its identity and rejoin fresh, since identities
are free. The prior (0.80, below a proven peer's score) makes this mildly unattractive but
does not prevent it. The real defence is proof-of-work or stake at identity creation, both
of which carry costs this project has deliberately refused. Currently unaddressed and
probably the most exploitable gap for a determined individual attacker.

**Browser/WebGPU nodes.** Spec §9 wants these later. They will have different noise
characteristics and probably no stable network identity, so both calibration and clustering
need rethinking before they are admitted.
