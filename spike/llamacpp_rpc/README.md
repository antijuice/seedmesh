# Spike: llama.cpp RPC vs Petals as a Seedmesh backend

**Question:** the three Petals spikes all came back cheap, so port cost no longer decides
the backend. The remaining argument was hardware reach. Does llama.cpp's RPC backend win it?

**Answer:** llama.cpp wins hardware reach decisively and loses suitability decisively, and
the two are not close enough to trade off. Its RPC protocol is a **remote-GPU** protocol for
trusted LANs, not a peer-to-peer inference protocol — measured below, an anonymous stranger
with a raw socket enumerated the host's hardware and allocated 256 MiB on it.

That is not a criticism of llama.cpp. Its README says exactly this. It is a statement about
fit for a *public permissionless* swarm, which is a thing it never set out to be.

Measured 2026-07-31. llama.cpp commit `876a432` (dated 2026-07-31), RPC protocol v5.0.0,
built CPU-only in WSL2 Ubuntu 24.04.

---

## Reproducing

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llamacpp
cd ~/llamacpp
cmake -B build -DGGML_RPC=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-rpc-server -j $(nproc)   # note: NOT "rpc-server"

./build/bin/ggml-rpc-server -p 50052 &
python3 spike/llamacpp_rpc/probe_rpc.py 127.0.0.1 50052
```

`probe_rpc.py` links no llama.cpp code. It is a raw TCP socket speaking the wire format
read out of `ggml/src/ggml-rpc/ggml-rpc.cpp`.

## 1. Hardware reach — llama.cpp wins, and I had overstated *why*

**A correction first.** Earlier notes in this project asserted "most volunteers will not
have an NVIDIA GPU." That is **not supported** for the population it implied. The Steam
Hardware Survey (June 2026) puts NVIDIA at ~72% of surveyed GPUs, AMD ~19%, Intel ~9%. Among
gaming PCs with discrete GPUs, most volunteers *would* have NVIDIA.

The real reach argument is not share-within-a-pool. It is that llama.cpp's backends open
**pools Petals cannot reach at all**:

| backend | hardware | Petals | llama.cpp |
| --- | --- | --- | --- |
| CUDA | NVIDIA discrete | ✅ | ✅ |
| Metal | **Apple Silicon** | ❌ | ✅ |
| ROCm / HIP | AMD RDNA / CDNA | ❌ | ✅ |
| SYCL | Intel Arc, Data Center, iGPU | ❌ | ✅ |
| Vulkan | cross-vendor, incl. iGPUs | ❌ | ✅ |
| OpenCL | Adreno (mobile) | ❌ | ✅ |
| CANN / MUSA | Ascend NPU, Moore Threads | ❌ | ✅ |
| CPU (+BLAS, ZenDNN, KleidiAI) | anything | ❌ (impractical) | ✅ |

Two of those matter disproportionately for a volunteer network:

- **Apple Silicon.** Unified memory makes an ordinary 16–64GB Mac a genuinely good inference
  host — often better than a 4GB gaming laptop — and Petals cannot use one at all. The
  self-hosting community Seedmesh needs first has a large Mac contingent.
- **Plain CPU.** It makes "donate a slice" possible on any machine, which changes who can
  participate rather than how fast they are.

So the honest framing: within gaming PCs, llama.cpp is roughly 72% → ~100%; beyond them, it
is the difference between a pool and no pool.

## 2. Security — measured, and it is the decision

The README warns: *"Never run the RPC server on an open network or in a sensitive
environment!"*, describing the implementation as "fragile and insecure" and a
proof-of-concept. That phrasing is easy to skim past in a comparison table, so it was
measured.

An anonymous client, raw socket, no credential of any kind:

```
1. HELLO                -> protocol v5.0.0
2. DEVICE_COUNT         -> 1 device(s) enumerated
3. GET_DEVICE_MEMORY[0] -> free 15.62 GiB / total 15.62 GiB
4. GET_MAX_SIZE         -> 17179869184.00 GiB max single buffer
5. ALLOC_BUFFER(256MiB) -> granted, remote ptr 0x609bd8e6a2e0
```

The complete unauthenticated command set (`enum rpc_cmd`, 18 commands) includes:

| command | what a stranger can do |
| --- | --- |
| `ALLOC_BUFFER` / `FREE_BUFFER` | allocate and free memory on the host |
| `SET_TENSOR` / `MEMSET_TENSOR` | **write** into host memory |
| `GET_TENSOR` / `COPY_TENSOR` | **read** host memory back |
| `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` | **execute a client-supplied compute graph** |
| `GET_DEVICE_MEMORY` / `DEVICE_COUNT` | fingerprint the host's hardware |

There is no authentication, no authorization, and no per-client capability restriction. Note
also `GET_MAX_SIZE` returning effectively `SIZE_MAX` on the CPU backend: allocation is
uncapped, so memory exhaustion is a one-line denial of service against a volunteer's
machine.

### This breaks Seedmesh's core promise to volunteers

`docs/security-privacy.md` tells volunteers: *hosting a server block does not let other
peers run arbitrary code on your machine — they can only send tensors through the model
layers you are hosting.* That is true for Petals, and it is **true because of an
architectural property**: the server owns the weights and the computation graph; the client
supplies only an input tensor.

llama.cpp RPC inverts that. The **client** owns the model and submits the graph; the server
is a general executor of whatever operations it is sent. A volunteer running `rpc-server` on
a public network has not donated a model shard — they have donated a programmable remote
accelerator to anonymous strangers.

Making it public-swarm-safe would require authentication, plus validating that each
submitted graph corresponds to the layers the volunteer agreed to host. That second part is
essentially re-deriving Petals' fixed-block model on top of a general RPC. It is not "add
TLS."

## 3. Architectural fit — three more mismatches

**No discovery layer.** Peers are specified explicitly: `--rpc 192.168.88.10:50052,...`.
There is no DHT, no peer announcement, no churn handling. hivemind supplies all of that
free, and the DHT spike confirmed its storage model already fits Seedmesh's reputation
records.

**Designed for LAN.** The documented and tested topology is a local network. Petals was
built for the internet, with activation compression sized for home upload bandwidth. WAN
behaviour of the RPC protocol is not something this spike measured, and should not be
assumed.

**Sharding model doesn't map.** llama.cpp distributes weights *proportionally to available
memory*, client-driven, via `--tensor-split`. Petals has servers **announce which block
ranges they host**, which is what Seedmesh's `BlockRange`, routing and per-range reputation
are built on. Reputation-per-block-range has no natural equivalent when the client decides
the split each run.

## 4. Where llama.cpp would actually be *better*

Fairness requires this section.

- **Verification gets easier.** Because the client orchestrates the whole graph, it already
  holds every intermediate tensor — sketching needs no protocol extension, and redundant
  execution is just sending the same subgraph to two servers.
- **Maintenance.** llama.cpp commit dated *today*, versus Petals dead since 2024-09-07. On
  liveness it is not a contest.
- **No quantization archaeology.** GGUF quantization is native and current, versus
  `bitsandbytes==0.41.1` (which, per the quantization spike, does still work — but GGUF is
  where the ecosystem's low-bit work actually happens).
- **Onboarding.** A single static binary against a `pip install` of a PyTorch stack is a
  large difference for non-technical volunteers.

## 5. Maintenance scoreboard

| project | last activity |
| --- | --- |
| **llama.cpp** | commits **2026-07-31** (today) |
| hivemind | release 1.1.12, **2026-01-03** |
| Petals | last commit **2024-09-07** — 23 months |

## Verdict

**Neither backend is right for both products, and that is the actual finding.**

| | Petals + hivemind | llama.cpp RPC |
| --- | --- | --- |
| public permissionless swarm | ✅ designed for it | ❌ explicitly warns against it |
| peer discovery / churn | ✅ DHT, TTLs | ❌ none |
| volunteer safety guarantee | ✅ structural | ❌ inverted |
| block-range sharding | ✅ matches Seedmesh | ❌ memory-proportional |
| hardware reach | ❌ NVIDIA only | ✅ everything |
| maintenance | ❌ dead 23 months | ✅ today |
| onboarding | ❌ PyTorch stack | ✅ single binary |

The split is clean and it maps onto a product question the original research doc already
flagged as unresolved: *is people's real want a shared public swarm, or exo's "mesh my own
devices" experience?*

- **Public swarm → Petals + hivemind.** It is the only option that is architecturally
  public-safe, and all three spikes showed the revival cost is bounded and mostly mechanical.
- **Private/LAN mesh → llama.cpp RPC.** It is nearly ideal there: trusted network, any
  hardware, one binary, no discovery needed.

Recommended: **Petals + hivemind for v1**, with llama.cpp RPC as a second backend behind the
same seam for private-swarm and LAN use — which is also the lowest-risk way to reach Mac and
CPU volunteers later, since a private-mesh user is not exposed to the security model that
disqualifies it publicly.

The backend seam in `seedmesh/backends/base.py` is what makes "both, for different modes" a
real option rather than a rewrite. That is the third time it has paid for itself.

**What this spike did not measure:** WAN latency and bandwidth behaviour of the RPC protocol
(needs two hosts and a model), and whether a constrained graph-validating proxy in front of
`rpc-server` could make it public-safe at acceptable cost. The second is the interesting
follow-up if hardware reach ever becomes the binding constraint.

## Sources

- [llama.cpp RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) — security warnings, usage, tensor-split
- [ggml-rpc.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-rpc/ggml-rpc.cpp) — `enum rpc_cmd`, wire format, `hello()` handler
- [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) — backend matrix
- [Steam Hardware & Software Survey](https://store.steampowered.com/hwsurvey/videocard/) — GPU vendor share
- [Steam Hardware Survey June 2026 coverage](https://www.techtimes.com/articles/319581/20260703/steam-hardware-survey-june-2026-windows-11-tops-70-amd-closes-intel.htm)
