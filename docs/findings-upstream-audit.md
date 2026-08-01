# Upstream audit: the state of Petals and hivemind

*Verified 2026-07-31 against the live repositories and endpoints. Re-check before acting on
any of it; every claim below is a fact about a specific date.*

The MVP spec is built on one load-bearing premise — "Petals is not dead, it's just quiet…
fork it, don't rebuild." That premise was checked directly. It is **half right, and the
half that is wrong changes the plan.**

---

## 1. Petals has not been touched in 23 months

| Signal | Value |
| --- | --- |
| Last commit (`pushed_at`) | **2024-09-07** |
| `updated_at` | 2026-07-31 |
| Archived | No |
| Stars | 10,462 |
| Open issues | 113 |

The spec's read that the repo "still has commits into 2025" came from `updated_at`, which
GitHub bumps on stars, watches and forks — not on code. `pushed_at` is the commit signal,
and it points at September 2024.

"Quiet" implies someone is still there. Nobody is. This does not block a fork — MIT is MIT
— but it means every compatibility problem below is now **yours**, with no upstream to
merge fixes from. The spec's plan to vendor Petals "as a git subtree so upstream changes
can still be pulled in later" is inheriting a maintenance burden, not sharing one.

## 2. The dependency pins are the real problem

From `setup.cfg` on `main`:

```
torch>=1.12
transformers==4.43.1          # hard pin, July 2024
bitsandbytes==0.41.1          # hard pin, September 2023
numpy<2
peft==0.8.2
tensor_parallel==1.0.23
hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98...   # frozen SHA
python_requires = >=3.8
```

Two consequences.

**It will not install on this machine's default Python.** `numpy<2` has no cp313 wheels
(NumPy 1.26.4 supports up to 3.12). The Petals environment needs Python 3.10 or 3.11,
separate from whatever else is installed.

**The launch model catalog is blocked behind a port.** Qwen3 and Llama 4 need
`transformers>=4.51`. Petals pins 4.43.1.

## 3. "Modernized model catalog = mostly config" is not correct

> **Corrected 2026-07-31 by the port spike.** This section originally claimed
> `_prepare_4d_causal_attention_mask` and the other imports had been *removed*. That was
> wrong — they all still resolve on transformers 5.14.1, behind back-compat shims. The
> conclusion (this is a port, not config) survives; the mechanism was misdiagnosed. Measured
> details in [`spike/transformers_port/README.md`](../spike/transformers_port/README.md).

Spec §3.3 says adding a current model is "primarily: confirm architecture compatibility,
publish block-conversion config". The code says otherwise. `src/petals/models/llama/block.py`
subclasses `LlamaAttention` and `LlamaDecoderLayer` and carries a hand-copied fork of the
attention body so CUDA-graph optimizations can be spliced in.

The imports still work. What broke is **attributes and signatures** — a worse failure mode,
because the module loads cleanly and only fails at the first forward pass:

```
AttributeError: 'OptimizedLlamaAttention' object has no attribute 'num_heads'
```

`self.num_heads`, `self.hidden_size`, `self.num_key_value_heads` and `self.rotary_emb` are
all gone from `LlamaAttention` (rotary moved up to `LlamaModel` in 4.48); `forward` now takes
`position_embeddings` instead of `position_ids`; the cache parameter was renamed and must be
a `Cache` object; and `LlamaDecoderLayer.forward` now returns a bare tensor rather than a
tuple.

So bumping transformers means **reworking the block wrapper for every forked architecture**,
then re-validating numerical equivalence. The spike established the recipe and measured it:
138 lines replacing 221, numerically exact. Note also that the current transformers is
**5.14.1** — a major-version boundary beyond the 4.x line Petals pins, not merely the 4.48
refactor.

## 4. The public swarm is gone

| Endpoint | Status |
| --- | --- |
| `health.petals.dev` | **Connection refused** (24.144.96.147:443) |
| `chat.petals.dev` | Loads; reports *"out of capacity — attention caches of existing servers are full"* |

The only model `chat.petals.dev` still offers is **Stable Beluga 2 (70B)**, a Llama-2
fine-tune from 2023.

This kills the spec's easiest on-ramp: §0.5's "your laptop can run the client library
today against the live public Petals swarm with zero other setup" is not available, because
there is no swarm to connect to. M0 cannot begin by validating against the public network.

It also changes the pitch. "Revive the swarm" implies something is still running. Nothing
is. The honest framing is closer to *rebuild the network, reusing the protocol* — which is
a better story anyway, and it removes the risk of a launch post being corrected in public
by someone who checked.

## 5. hivemind is alive

| Signal | Value |
| --- | --- |
| Last commit (`pushed_at`) | **2026-01-11** |
| Archived | No |
| Stars | 2,507 |

This is the good news, and it is the important half. The genuinely hard, genuinely reusable
part — the Kademlia DHT, peer discovery, NAT traversal, the tensor transport — is
maintained. The spec's core insight ("don't rebuild the hard 20%") survives intact; it just
points at hivemind rather than at Petals.

---

## What this changed in the build

Petals' staleness is a risk to *the backend*. It is not a risk to the reputation and
verification layer, which is the novel work and the reason the project exists. So the
backend was put behind an interface (`seedmesh/backends/base.py`) with three methods, and
the trust layer was built against that interface and a simulator instead of against a
2-year-old fork.

Concretely, this means the porting decision stays open and reversible:

* **Petals adapter** — the block-wrapper port is now a solved, recipe-shaped problem
  (spike: 138 lines, numerically exact, one day per architecture). The remaining risk moved
  to the **dependency stack**: `bitsandbytes==0.41.1` (September 2023) and
  `tensor_parallel==1.0.23` are what actually gate a working server, and neither was touched
  by the spike.
* **llama.cpp RPC adapter** — much broader hardware reach (Apple Silicon, AMD, plain CPU),
  which matters more for a volunteer network than it does for a datacenter. Its RPC
  protocol is explicitly "insecure by default, restrict to trusted networks", so it needs a
  security layer before public use.
* **Simulator** — already implemented, and it is what the trust layer is tested against.

None of these can invalidate the reputation or verification work, because none of them
appear in its imports.

## Recommended next actions

1. **Do not lead public messaging with "reviving Petals."** The swarm is down and the repo
   is two years stale; someone will check. Lead with the mission and with the trust layer,
   which is genuinely new.
2. ~~Spike the transformers port before committing to Petals.~~ **Done** — see
   [`spike/transformers_port/`](../spike/transformers_port/). Verdict: tractable, with a
   proven recipe and a reusable equivalence harness. The port is no longer a reason to
   avoid Petals.
3. **Spike quantization next.** `bitsandbytes==0.41.1` is where the remaining unknown risk
   sits, and it gates whether a server can hold useful block counts at all.
4. **Evaluate llama.cpp RPC seriously**, per spec §9. Now that port cost is bounded, this
   choice should turn on *hardware reach* — most volunteers will not have an NVIDIA GPU.
4. **Vendor hivemind, not Petals, as the thing you depend on.** It is maintained.

## Sources

- `https://api.github.com/repos/bigscience-workshop/petals` — `pushed_at`, `archived`, counts
- `https://raw.githubusercontent.com/bigscience-workshop/petals/main/setup.cfg` — pins
- `https://raw.githubusercontent.com/bigscience-workshop/petals/main/src/petals/models/llama/block.py` — internals
- `https://api.github.com/repos/learning-at-home/hivemind` — `pushed_at`
- `https://health.petals.dev/` — connection refused
- `https://chat.petals.dev/` — capacity notice, model list
