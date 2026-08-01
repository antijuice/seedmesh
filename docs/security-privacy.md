# Security and privacy

Written to be read by a volunteer deciding whether to donate their GPU, and by a user
deciding what to send through the swarm. Both deserve the unflattering version.

---

## For people hosting a server

### Other peers cannot run code on your machine
You load a fixed set of transformer blocks and apply them to tensors you receive. Peers send
activations, not programs. There is no code path that lets a client choose what your GPU
executes — only what it executes *on*. This guarantee is inherited from Petals and is core
to why volunteering is reasonable at all.

**This is a property of the backend, not a policy, and it is why the backend was chosen.**
It holds because the *server* owns the weights and the computation graph while the client
supplies only an input tensor. A backend that inverted that would not have it. llama.cpp's
RPC backend does invert it — the client submits the compute graph and the server executes
what it is sent — which is correct for the trusted LAN it targets and disqualifying for a
public swarm. Measured in `spike/llamacpp_rpc/`: an anonymous socket with no credential
enumerated the host's devices, read its free memory, and allocated 256 MiB on it.

If Seedmesh ever ships a llama.cpp backend it will be for **private meshes only**, and this
page will say so at the point of use rather than in a footnote.

What that does **not** cover: bugs in the deserialization path. Anything that parses
untrusted bytes is attack surface, and the backend adapter owns that. It should be reviewed
before public launch and is currently unwritten.

### What you can see
You see intermediate activations passing through your blocks. You cannot see prompt text
directly. You *can* see something correlated with it, and if you host the final blocks you
are closest to the output distribution. Do not assume hosting is anonymous with respect to
content.

### What you are exposing
Your IP address is visible to peers you talk to — unavoidable in a P2P network, and the
same exposure BitTorrent has always had. Seedmesh additionally uses coarse network location
(address prefix, and ASN where resolvable) for anti-sybil clustering. That is computed
locally by whoever is talking to you, from information they already have; it is not
published as a claim about you.

Peers with no resolvable address are all placed in a single shared bucket, not given
individual anonymity — otherwise omitting your address would be a way to mint unlimited
sybil identities.

## For people using the swarm

### The public swarm is not private
Prompts flow through strangers' machines as activations. Activations are not plaintext, but
treating them as safe would be wrong: recovering input from hidden states is an active
research area, and the practical difficulty varies by model and layer depth.

**Do not send anything through the public swarm you would not send to an untrusted third
party.** For sensitive data, run a private swarm — the same code, a closed peer set. This is
stated here, in the README, and in the CLI rather than buried, because a volunteer network
runs on trust and losing it once is permanent.

### Verification checks agreement, not correctness
Redundant sampling detects servers whose output disagrees with an independent peer's. It
cannot detect a server running genuine weights that produce subtly biased results, because
that is indistinguishable from honesty by construction. Seedmesh raises the cost of
returning garbage; it does not certify that any answer is right.

### Detection is probabilistic and delayed
Verification samples a fraction of requests. A cheating server is caught after some number
of requests, not before its first. In simulation, detection took tens to low hundreds of
requests depending on attack type. Requests served in the meantime were served by a cheat.

---

## Cryptographic details

* **Identity:** Ed25519. Peer id = `sm` + base32(sha256(public key)[:20]).
* **Signing:** deterministic canonical JSON — sorted keys, compact separators, floats
  quantized to 6 decimals, non-finite values rejected. Every signature is domain-separated
  by a context string.
* **Sketch seeds:** SHA-256 over a client nonce (≥16 bytes), the block range key, and the
  step index. Servers cannot predict the projection basis before committing to output.
* **Record limits:** ≤512 reports per batch, ≤1h TTL, ≤5min clock skew, monotonic epochs.

Note the honest limit of identity: it makes records **attributable**, not **true**. Anyone
can mint unlimited keypairs, so a peer id is a pseudonym, not a scarce resource. Sybil
resistance comes from the aggregation rules, not from cryptography.

## Reporting a vulnerability

Do not open a public issue for anything exploitable. See `SECURITY.md` (to be added before
public launch — currently there is no swarm to attack).

## Known gaps

Tracked honestly in [threat-model.md](threat-model.md). The ones that most affect a
prospective volunteer:

* An attacker with genuine network diversity across many ASNs defeats the cluster-based
  anti-sybil rules.
* A caught peer can discard its identity and rejoin fresh; identities are free.
* Colluding servers with a side channel can coordinate to defeat redundant sampling.
* Transport-layer attacks (DHT eclipse, resource exhaustion) are hivemind's surface and
  have not been reviewed here.
