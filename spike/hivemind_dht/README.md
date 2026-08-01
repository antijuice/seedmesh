# Spike: Petals' DHT plumbing on current hivemind

**Question:** the previous two spikes left one Petals component unassessed — its
server/client/DHT layer against current hivemind. Is it a port, and can Seedmesh's signed
reputation records actually live in a hivemind DHT?

**Answer:** hivemind is the healthiest dependency in the entire stack. The breakage is
**7 import statements across 22 files**, all mechanical, with zero logic changes. And every
storage assumption the reputation design made turned out to be natively supported.

Measured 2026-07-31 in WSL2 Ubuntu 24.04. hivemind 1.1.12, torch 2.13 CPU, Python 3.12.3.

---

## Reproducing

hivemind is Linux/macOS only, so this runs in WSL, not native Windows.

```bash
# python3.12-venv is absent and sudo needs a password, so bootstrap pip inside the venv
python3 -m venv --without-pip ~/hmspike
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/hmspike/bin/python3 /tmp/get-pip.py
~/hmspike/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/hmspike/bin/pip install hivemind==1.1.12
~/hmspike/bin/pip install -e /mnt/c/Users/jrhsu/SeedMesh

~/hmspike/bin/python3 spike/hivemind_dht/probe_api.py
~/hmspike/bin/python3 spike/hivemind_dht/test_dht.py
~/hmspike/bin/python3 spike/hivemind_dht/test_validator.py
```

Remove with `rm -rf ~/hmspike`. Nothing outside that directory is touched.

## Finding 1 — hivemind is genuinely maintained

| release | date |
| --- | --- |
| 1.1.10.post2 | 2023-08-31 |
| 1.1.11 | 2025-04-20 |
| **1.1.12** | **2026-01-03** |

Petals pins a git SHA from the ~1.1.10 era. So the gap is two minor releases over ~2.5
years — against `transformers` 4.43 → 5.14, a major-version boundary. This confirms the
audit's central claim: the hard, reusable part of the stack is alive.

## Finding 2 — the breakage is import paths, and nothing else

I predicted the risky parts would be the deep reaches: `hivemind.moe.*` (which Petals
repurposes, hosting transformer blocks through the mixture-of-experts machinery),
`p2p_daemon_bindings.control` internals, and the module-level global Petals *assigns to* in
`hivemind.compression.base`.

**All of those survive, 6/6.** What broke is the opposite — hivemind narrowed its top-level
namespace, so the convenience re-exports are gone while every symbol remains at its
canonical submodule path:

```
BREAK  from hivemind import PeerID                    ->  hivemind.p2p.PeerID
BREAK  from hivemind import MSGPackSerializer         ->  hivemind.utils.serializer.MSGPackSerializer
BREAK  from hivemind import get_logger                ->  hivemind.utils.logging.get_logger
BREAK  from hivemind import TensorDescriptor          ->  hivemind.utils.tensor_descr.TensorDescriptor
BREAK  from hivemind import nested_flatten, ...       ->  hivemind.utils.nested.*
BREAK  from hivemind import serialize_torch_tensor    ->  hivemind.compression.serialization.*
BREAK  from hivemind import anext                     ->  hivemind.utils.asyncio.anext
OK     from hivemind import DHT
OK     everything under hivemind.moe / hivemind.p2p / hivemind.utils / hivemind.compression
```

7 broken statements across **22 files**. Mechanical rewriting, no logic changes, no
numerical validation needed. This is the cheapest fix of any spike so far.

*A methodology correction worth recording:* an earlier pass of `probe_api.py` reported
`from hivemind.utils import limits` as broken, using `getattr` on the imported package. That
is **not** equivalent to a `from X import Y` statement, which falls back to importing a
submodule. `limits` is a submodule and imports fine. Import-surface probes must execute the
actual statement, not approximate it.

## Finding 3 — the DHT storage model already fits the reputation design

All six checks in `test_dht.py` pass on a real 3-node DHT.

**Per-observer subkeys are native.** Petals stores block announcements as
`store_many(keys=uids, subkeys=[peer_id], ...)`, and hivemind gives each subkey its own
value *and its own expiration* under a shared key. That is exactly the shape reputation
needs — one slot per observer, newest-wins, independently TTL-expiring — so no additional
structure has to be invented:

```
key = seedmesh/reputation/<subject>
  subkey = <observer A>  ->  signed batch, expires independently
  subkey = <observer B>  ->  signed batch, expires independently
```

**TTL expiry works**, which is what makes churn handling free: a departed observer stops
influencing routing on its own, with no liveness protocol.

**512-report batches publish fine.** The design assumed `MAX_REPORTS_PER_BATCH = 512`
without checking. Measured:

| reports | bytes | roundtrip |
| --- | --- | --- |
| 1 | 412 | ok |
| 10 | 1,405 | ok |
| 100 | 11,416 | ok |
| 512 | **57,460** | ok |

A signed `ObservationBatch` survives the roundtrip and still verifies after retrieval —
signature valid, observer id intact.

## Finding 4 — the DHT gives no write protection by default

The most important result, because it converts an assumption into a measurement.

`reputation/records.py` justifies Ed25519 signing by asserting that "a DHT is an open write
surface… an unsigned record there is worth nothing." That was reasoning. It is now tested:

```
3. can bob overwrite alice's subkey?
   -> alice's slot now = {'reliability': 1.0}   (no DHT-layer protection)
```

Bob overwrote Alice's record with no error and no signal. On a default hivemind DHT,
**Seedmesh's Ed25519 signatures are carrying 100% of the attribution guarantee.**

### hivemind's own validator does block it

`RSASignatureValidator` introduces "protected records" whose key or subkey embeds
`[owner:<rsa public key>]`, and only that key's holder can write them. With it enabled:

```
2. bob tries to overwrite alice's protected subkey
   store returned False, alice's slot now = {'reliability': 0.2}   <- blocked
3. unprotected subkeys remain hijackable                            <- confirmed
```

**A test-methodology trap worth flagging**, because it produced a wrong result first: the
default `RSASignatureValidator()` uses `RSAPrivateKey.process_wide()`. That is correct in
production, where every peer is a separate process — but in a single-process test it hands
all three nodes *the same identity*, so "Bob" cryptographically **is** Alice and every
overwrite looks legitimate. The first run reported "the validator does not block it," which
was an artefact. Constructing explicit `RSAPrivateKey()` instances fixes it.

### Recommendation: use both, because they protect different things

| | protects | scope |
| --- | --- | --- |
| hivemind `RSASignatureValidator` | the DHT *slot* — only the owner may write that subkey | the DHT transport only |
| Seedmesh Ed25519 batch signature | the *content* — authorship, epoch, expiry | travels with the record, verifiable offline by anyone |

They are complementary. hivemind's validator stops a hijack at the storage layer for free;
Seedmesh's signature is what still means something once a record has been relayed, cached,
or re-served by a third party — and it is what carries the monotonic epoch that stops an old
favourable batch being replayed, which slot ownership alone does not address.

Cost note: an RSA public key in every subkey is 388 bytes, versus 32 for Ed25519. Negligible
against a 57KB batch, but it is why the content signature should stay Ed25519 rather than
being folded into hivemind's scheme.

## Verdict

| component | status |
| --- | --- |
| **hivemind DHT/p2p** | **7 import statements, 22 files — mechanical** |
| bitsandbytes | works unmodified |
| tensor_parallel | deletable for single-GPU |
| block wrappers | port required — recipe proven, 138 lines, numerically exact |
| Petals server/client logic *above* the imports | still unexercised end-to-end |

All three spikes came back cheaper than the audit predicted. The stale-dependency risk that
motivated the backend seam has largely evaporated — though the seam still earned its place,
because it is what let the trust layer be built, tested and twice *corrected* without any of
this being resolved first.

**What remains genuinely unknown** is no longer any single dependency. It is whether Petals'
server/client logic — `RemoteSequential`, the routing/sequence manager, the inference
session protocol — actually functions end-to-end once the imports are fixed. That needs a
running two-node swarm, not another compatibility probe.

And the question none of these spikes can settle, now that port cost is bounded across the
board: **whether a PyTorch/CUDA-only backend is the right choice for a volunteer network
where most contributors will not own an NVIDIA GPU.** That is a hardware-reach decision, and
it is the one still worth arguing about.
