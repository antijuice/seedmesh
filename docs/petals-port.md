# State of the Petals port

Petals has been unmaintained since 2024-09-07. Four measurement spikes established that
reviving it is bounded (`spike/*/README.md`); this is the port itself.

**Status as of 2026-07-31: a private two-server swarm generates real tokens on fully current
dependencies.** Seven patches.

```
Route found: 0:6 via …ucdzGr => 6:12 via …ZfoVCY
prompt:    'The capital of France is'
GENERATED: 'The capital of France is Paris.\nThe city of Paris is the capital of France'
2 span(s) across 2 distinct host(s)
```

Two servers each hosting half of a 12-layer model, a client discovering both through the
DHT and streaming activations across both to produce coherent text. Reproduce with
`bash tools/swarm_demo.sh`.

| dependency | Petals pinned | port runs on |
| --- | --- | --- |
| transformers | `==4.43.1` | **5.14.1** |
| hivemind | frozen git SHA (~1.1.10) | **1.1.12** |
| torch | `>=1.12` | **2.13.0** |
| numpy | `<2` | **2.5.1** |
| peft | `==0.8.2` | **0.20.0** |
| bitsandbytes | `==0.41.1` | **0.50.0** |

## A codemod, not a fork

`tools/port_petals.py` patches a Petals checkout in place; `tools/verify_petals_port.py`
checks the result works.

```bash
git clone --depth 1 https://github.com/bigscience-workshop/petals.git ~/petals
python tools/port_petals.py --petals-root ~/petals --check   # dry run
python tools/port_petals.py --petals-root ~/petals
pip install --no-deps -e ~/petals
python tools/verify_petals_port.py
```

Vendoring 67 files would make "what did we change to upstream?" unanswerable and would
silently adopt upstream's bugs as ours. A codemod keeps the delta explicit, reviewable, and
re-appliable to a newer checkout. It is idempotent — re-running reports "already applied".

Note: hivemind is Linux/macOS only, so this runs in WSL on Windows.

## The five patches

**1. hivemind import paths — 26 statements across 26 files.**
hivemind 1.1.12 narrowed its top-level namespace. Every symbol still exists at its canonical
submodule path, so this is mechanical: `from hivemind import PeerID` →
`from hivemind.p2p import PeerID`.

**2. hivemind attribute access — 9 sites across 3 files.**
A second breakage form, and the one that matters most as a lesson: `import hivemind` then
`hivemind.PeerID`. An import-statement rewrite misses it entirely. Rewritten to
`hivemind.p2p.PeerID`, needing no new imports because the submodules still resolve as
attributes of the root package.

**3. Llama block wrapper — 138 lines replacing 221.**
Petals forked transformers' attention body to splice in two CUDA-graph optimizations; that
fork is what broke. The port deletes it and delegates to the stock `LlamaDecoderLayer`,
keeping only what a hosted block genuinely needs: its own rotary embedding, its own causal
mask, cache-layout translation, and unchanged parameter names. Verified numerically exact
(`spike/transformers_port/`).

**4. Skip tensor parallelism on a single device.**
`convert_block` wrapped every server in `TensorParallel` even for one device — "for
uniformity" per upstream's own docstring — and that path reads `submodule.num_heads`, the
same attribute the block port found removed. Skipping it for one device drops the
`tensor_parallel==1.0.23` pin for the volunteer case.

**5. `get_file_from_repo` shim.**
Removed from `transformers.utils` in 5.x, used in `server/from_pretrained.py` and
`utils/peft.py`. `cached_file` is the replacement but changed two things that matter at
every call site: it raises where the old function returned `None` (Petals uses that as a
file-existence probe), and the auth kwarg was renamed `use_auth_token` → `token`. A shim at
`petals/utils/hf_compat.py` restores the original contract.

**6. `_supports_cache_class`.**
A transformers flag meaning "this model understands the Cache API". Every model does in 5.x,
so it was removed — and reading a missing attribute on an `nn.Module` raises rather than
returning `None`. Defaulted to `True`.

**7. `RemotePastKeyValues.__init__`.**
`Cache.__init__` now requires exactly one of `layers` or `layer_class_to_replicate`. Petals'
stub cache called `super().__init__()` with neither. It stores nothing locally — the real KV
cache lives on the servers — so `layers=[]` is the accurate representation.

Plus a widened — not deleted — transformers version assertion: `>=4.48` (the attention
refactor the ported block targets) instead of a single pinned patch release.

Two runtime dependencies also need installing that `--no-deps` skips: `dijkstar`,
`humanfriendly`, `async-timeout`, `sentencepiece`, `peft`, `speedtest-cli`, `requests`,
`bitsandbytes`, and `cpufeature`. The last one is worth flagging: `lm_head.py` guards its
import with `platform.machine() == "x86_64"` — a check on the *machine*, not on whether the
package is present — so on x86_64 it was mandatory, not optional.

**Since 2026-08-03 it is patched out instead of installed.** `cpufeature` has no wheels for
recent Pythons and building it needs a C compiler, which is exactly what a fresh volunteer
machine lacks — so "install it" turns one missing dependency into two. It decides a single
performance path (AVX512 present → plain bf16 beats chunked_forward by ~10x; absent → the
reverse), and never correctness. The codemod wraps the import and falls back to reading
`avx512f` from `/proc/cpuinfo`, verified to flip the decision in both directions.

A note on how this was missed: `setup.py`'s own comment named `cpufeature` as mandatory while
`RUNTIME_DEPS` omitted it. The developer machine had it from an earlier hand-built venv, so
every install here succeeded and the first genuinely fresh machine failed.

## What the verifier proves

```
PASS  import petals
PASS  config resolves the ported block class          (WrappedLlamaBlock)
PASS  block constructs via Petals' block_class        (139,520 params)
PASS  forward pass returns the expected shape         ((2, 6, 128))
PASS  incremental decode matches full sequence        (max diff 4.768e-07)
PASS  convert_block works on a single device          (no TensorParallel wrapper)
PASS  all 12 server/client/CLI entry points import    (12/12)
```

The incremental-decode check is the one that matters. It drives Petals' own `layer_past`
convention, so it exercises the cache-layout translation end to end — and a transposed
reshape there passes a single forward pass and fails from the second token onward.

## The two-server swarm

`tools/swarm_demo.sh` brings up two servers hosting complementary halves of the model and
runs `tools/swarm_client.py` against them. Everything is localhost and `--new_swarm`, so it
never touches the public network.

It runs as one shot on purpose: block announcements carry a TTL, and a client started
several minutes later finds the records already expired.

What the run establishes, beyond the block-level checks:

- **DHT announcement and discovery work across hosts.** The client resolves 12/12 blocks and
  attributes them to two distinct peer ids.
- **The sequence manager builds a real multi-host pipeline** — `Route found: 0:6 via …ucdzGr
  => 6:12 via …ZfoVCY` — rather than merely importing.
- **The ported block is numerically right in practice.** A 160M model producing "The capital
  of France is Paris." is a stronger end-to-end signal than any unit test: a subtly wrong
  attention or cache reshape yields fluent gibberish, not a correct fact.

## What it does NOT prove

- **No GPU.** CPU float32 only; the quantized `convert_block` paths need CUDA, and only
  `QuantType.NONE` has been exercised through Petals' own machinery.
- **One tiny model.** 160M parameters, 12 layers. Nothing about memory pressure, attention
  cache limits, or throughput at real model sizes.
- **Only Llama.** Falcon still carries a 361-line forked attention with the same problem.
  Bloom and Mixtral already delegate upstream and are probably fine, but untested.
- **Localhost only.** No NAT traversal, no relays, no real network latency or churn. A
  server was never killed mid-request to test rerouting.
- **One client, one request.** No concurrency, no load.

## Churn: a killed server is survived

`bash tools/churn_test.sh` runs three servers with the tail half **redundant** (blocks 0-5,
6-11, 6-11), then SIGKILLs whichever server the client preferred for the tail and generates
again. Killing the sole host of a range would only prove that inference stops.

```
2. killing the server for the tail blocks: peer ...RFesyVyM (pid 564)
   process alive after SIGKILL: False
3. generating again with the cached route now pointing at a dead peer
  [after] OK: 'The capital of France is Paris.\nThe city of Paris is the capital'

PASS: generation completed with peer ...RFesyVyM dead
      outputs identical: True
      killed peer still in candidate list (stale DHT record): True
```

`outputs identical: True` is the part that matters. Greedy decoding through the replacement
server produced the *same tokens*, so it genuinely recomputed those blocks rather than the
client silently degrading or truncating.

The dead peer remains in the candidate list until its DHT record hits TTL — expected, and
precisely why routing must tolerate stale records rather than trust them.

**A finding with teeth beyond the test:** Petals' client defaults to `max_retries=None`
(retry forever) with a 180s `request_timeout`. The first churn run simply hung. For Seedmesh
that default is actively wrong: the reputation layer learns from *outcomes*, and a call that
retries forever never returns one — so a dead peer would be recorded as neither success nor
failure and routing would keep selecting it. `PetalsBackend` therefore sets bounded retries
by default. Unbounded retry is how a trust layer goes blind.

## Retracted: the "announcements stop refreshing" bug

An earlier revision of this document reported that block announcements stopped being
refreshed — a client seeing 12/12 immediately and 6/12 minutes later. **That bug does not
exist.** Measured directly at the default `--update_period` (120s, expiration 240s):

```
t= 60s  covered 12/12  servers=['NofTF56x', 'jK6Y67v1']
t=180s  covered 12/12  servers=['NofTF56x', 'jK6Y67v1']
t=300s  covered 12/12  servers=['NofTF56x', 'jK6Y67v1']
t=420s  covered 12/12  servers=['NofTF56x', 'jK6Y67v1']
```

Seven minutes, well past the TTL, across three refresh cycles, both servers stable.

The real cause was my own tooling: several `pkill -f <pattern>` commands were issued where
the pattern also appeared in the invoking command line, so `pkill` matched and killed its
own shell — and, in one case, a server's libp2p daemon. The resulting 6/12 was self-inflicted
and misattributed to Petals.

Two lessons worth keeping, since both cost real time here:

* `pkill -f` self-matches. Put kills in a script file (whose command line is just the script
  path), or use the `[p]attern` bracket trick.
* A symptom observed in a session where you have been killing processes is not evidence
  about the software. Re-measure on a clean run before writing it down as a defect.

Cosmetic and real: hivemind's `P2P.__del__` raises `RuntimeError: There is no current event
loop` at interpreter shutdown under uvloop. Harmless, appears after results.

## What broke my own process, twice

Both worth recording because they are the kind of error that produces false confidence:

**A partial rewrite reported success.** The codemod's first version left any import
statement containing an unmapped symbol untouched — correct behaviour — but did so
*silently*. It printed four clean "OK" lines and the very next import failed. It now
collects unmapped symbols, prints them, and exits non-zero.

**An import-statement rewrite is not a namespace rewrite.** Patch 2 exists because I
assumed `from X import Y` was the only way Petals reached the hivemind root. It wasn't, and
nothing in the earlier DHT spike would have caught it — that spike tested *hivemind's* API
surface, not *Petals'* usage of it.

## The trust layer, driving real servers

`seedmesh/backends/petals_backend.py` implements the three-method backend interface.
`CLIENT=tools/backend_demo.py bash tools/churn_test.sh` runs Seedmesh's reputation and
verification against the live swarm:

```
model has 12 blocks; discovered 3 online server(s):
  ...8AhSXXiP  blocks  0- 5  profile=none/float32/eager  cluster=cluster:unknown
  ...JboMVtrf  blocks  6-11  profile=none/float32/eager  cluster=cluster:unknown
  ...C2e6sStA  blocks  6-11  profile=none/float32/eager  cluster=cluster:unknown

Seedmesh route (2 hop(s), cost 1.200):
  blocks  0- 5 via ...8AhSXXiP  score=1.000
  blocks  6-11 via ...JboMVtrf  score=1.000

running the pipeline:
  ...8AhSXXiP  ok         201.9 ms
  ...JboMVtrf  ok         144.6 ms

verifying blocks 6-11: ...JboMVtrf vs ...C2e6sStA
  relative distance: 0.000000
  verdict:           match
```

Three things this establishes that the simulator could not:

- **`run_segment` pins work to a named server.** Petals' own client picks whatever route it
  likes; verification requires sending identical input to a *chosen* peer. `run_remote_forward`
  is the primitive that allows it, and it works.
- **Petals already publishes the compute profile.** `torch_dtype` and `quant_type` are in
  its server records, which is exactly what the quantization spike concluded must be
  published for per-pair tolerances. The attention kernel is not — still a gap.
- **The distance is 0.000000 because there is no hardware diversity here** — two servers,
  identical fp32 weights, one machine. That is the null case, not a validation of the
  tolerance. Real thresholds still need the multi-GPU calibration run.

### Peer addresses, resolved

Petals' DHT records carry only a libp2p PeerID — no IP. Without one, every peer falls into
the shared "unknown" cluster and the sampler refuses to pair *any* of them, so verification
was structurally sound but inert.

`PetalsBackend._resolve_addresses` fixes this using hivemind's `P2P.list_peers()`, which
returns each connected peer's multiaddrs. Running with servers bound to distinct loopback
addresses:

```
  ...UewPQ8NA  blocks  0- 5  addr=127.0.0.1  cluster=net:127.0.0.0/16
  ...5mwYjNKU  blocks  6-11  addr=127.0.0.2  cluster=net:127.0.0.0/16
  ...AKxpJM6T  blocks  6-11  addr=127.0.0.3  cluster=net:127.0.0.0/16
  addresses resolved from the p2p layer: 3/3

  default resolver (no ASN table): independent=False
  with an ASN table (simulated GeoIP): independent=True
  sampler picks a verifier for ...5mwYjNKU: ...AKxpJM6T
  sampler-driven verification: match (distance 0.000000)
```

Three things worth reading carefully:

- **`independent=False` under the default resolver is the correct answer.** All three
  servers really are on one host, sharing a /16. The constraint is now *informed* rather
  than vacuous — it refuses because it knows they are the same network, not because it knows
  nothing.
- **The ASN table is the only simulated part**, standing in for the offline GeoIP/Team Cymru
  lookup a deployment would use. `AsnResolver` is the documented extension point for exactly
  this, and ASN is resolved locally — never taken from a peer's own claim.
- **The distance of 0.000000 remains the null case.** Identical fp32 weights on one machine.
  It shows the plumbing carries real tensors end to end; it is not a validation of the
  tolerance.

Two limitations of the mechanism itself: `list_peers` only reports peers the daemon is
*currently connected to*, so a peer that has announced blocks but not yet been dialled has
no address and correctly stays unpairable; and address selection prefers the most routable
multiaddr, since a loopback address identifies no operator.

A bug this surfaced, worth recording because the tests caught it and inspection had not:
address ranking originally used `not is_private` to mean "public". Python's `ipaddress`
reports RFC 5737 documentation space (`203.0.113.0/24`) as private, and CGNAT and reserved
blocks similarly — so the predicate silently mis-ranked several classes of address.
`is_global` is the one that actually means "routable on the public internet".

### The attention kernel is now published

Patch 8. Petals announced `torch_dtype` and `quant_type` but not the attention
implementation — two thirds of what determines a server's numerical noise floor. A client
could not tell whether two servers were comparable, so a cross-kernel pair would be judged
against a tolerance fitted for a different kernel.

`ServerInfo` gains an `attn_impl` field, populated from the block config at announcement.
The addition is safe in both directions: the record serializes as
`(state, throughput, extra_dict)` and `from_tuple` ignores unknown keys, so an old client
skips the field and a new client reads `None` from an old server.

The adapter maps an absent value to `"unknown"` rather than assuming it matches ours — which
puts such a peer in its own profile bucket, so an uncalibrated pair refuses to verify instead
of being judged against the wrong threshold. Live discovery now reports
`profile=none/float32/eager`, and because the fallback is `"unknown"`, seeing `eager` is
itself proof the field is being published and read.

## Next

1. **Multi-GPU calibration** — `tools/calibrate/`, run on Colab. This is now the only thing
   standing between "verification runs" and "verification means something".
2. **Wire a real ASN table** (GeoLite2 or Team Cymru, loaded offline) into `ClusterIndex`.
3. GPU, quantization, and a larger model — in that order.
