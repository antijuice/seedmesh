"""Can a genuinely remote client reach this swarm's servers DIRECTLY, without any router?

Paste into Google Colab. This is the one test no local machine can run: a client on the same
NAT as a server always finds it over the LAN, and a client dialling its own network's public
address needs router hairpinning, which many routers refuse. Either way a local result says
nothing about a stranger's experience -- and a stranger's experience is the product.

WHY THIS MATTERS BEYOND SPEED
-----------------------------
Three things ride on a server being directly reachable rather than relayed:

  1. A relayed connection is reset after ~128 KiB per direction (go-libp2p circuit-relay-v2
     `DefaultResources()`). That is under one request of any real model. The budget CANNOT be
     raised -- a shim was built, deployed and measured three times, and p2pd parses the flags
     without applying them. Do not retry that.
  2. Verification needs to place both peers on a network. A circuit multiaddr carries the
     RELAY's address, not the peer's, so a relay-only server cannot be shown to be an
     independent operator and is refused as a verification partner.
  3. Without verification there are no verdicts, so the routing gate has nothing to act on.

WHAT IS ALREADY KNOWN (2026-08-01, measured from Colab)
-------------------------------------------------------
With four bootstrap peers, the home server learned and advertised its own public address and
Colab dialled it DIRECTLY -- 12/12 at 117 KB/request, with zero `p2p-circuit` and zero
`reservation` mentions in the server log. DCUtR was attempted and FAILED
(`protocols not supported: [/libp2p/dcutr]`); the direct connection came from the NAT being
endpoint-independent, not from hole punching.

So this run is asking a narrower question: does that still hold, and does it hold for a
server started with `--public-ip`, which forces the advertisement rather than waiting for
Petals' pessimistic startup check to allow it?

HOW TO READ THE RESULT
----------------------
  direct + no stalls past 128 KiB   -> router-independent hosting works here
  relayed + stalls at ~128 KiB      -> this NAT needs a real forward, or client-only
  relayed + no stalls               -> re-check the byte arithmetic before believing it

SETUP (Colab cell 1)
--------------------
    !pip install -q git+https://github.com/<your-fork>/seedmesh
    !seedmesh setup --skip-install || true
    !pip install -q hivemind==1.1.12

Then paste this file as cell 2.
"""

import os

# Must precede every hivemind import: p2pd inherits this at launch, and setting it later
# silently produces a quiet daemon and an empty diagnosis.
os.environ["GOLOG_LOG_LEVEL"] = "error,relay=info,swarm2=info,basichost=info"

import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from hivemind.dht import DHT  # noqa: E402
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker  # noqa: E402

from petals import AutoDistributedModelForCausalLM  # noqa: E402
from seedmesh.cli.swarm import resolve_peers  # noqa: E402

MODEL = "JackFram/llama-160m"
HIDDEN, DTYPE_BYTES = 768, 4  # fp32 client
RELAY_BUDGET = 128 * 1024
TRIALS = 12

peers, note = resolve_peers(None, None)
print(note or f"{len(peers)} bootstrap peer(s)")

# DHT before the model: hivemind allocates a shared-memory buffer on first DHT construction,
# and share_memory_() raises inside transformers' low-cpu-mem init context.
dht = DHT(initial_peers=peers, client_mode=True, start=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoDistributedModelForCausalLM.from_pretrained(
    MODEL, initial_peers=peers, torch_dtype=torch.float32, request_timeout=20, max_retries=1
)

from seedmesh.verification.petals_bridge import sequence_manager_of  # noqa: E402

manager = sequence_manager_of(model)
prompt = tokenizer("The history of distributed systems", return_tensors="pt")["input_ids"]
n_prompt = prompt.shape[1]


def connections():
    """Every peer this client is connected to, and how."""
    try:
        listed = RemoteExpertWorker.run_coroutine(manager.state.p2p.list_peers())
    except Exception as exc:
        print(f"  could not list peers: {type(exc).__name__}: {exc}")
        return {}
    found = {}
    for peer in listed:
        addrs = [str(a) for a in (getattr(peer, "addrs", []) or [])]
        found[str(peer.peer_id)] = addrs
    return found


def report(label):
    print(f"\n=== connections {label} ===")
    for peer_id, addrs in connections().items():
        relayed = all("p2p-circuit" in a for a in addrs) if addrs else True
        kind = "RELAYED" if relayed else "DIRECT"
        ip = "?"
        for addr in addrs:
            if "p2p-circuit" not in addr:
                match = re.match(r"^/ip4/(\d{1,3}(?:\.\d{1,3}){3})/", addr)
                if match:
                    ip = match.group(1)
                    break
        print(f"  ...{peer_id[-8:]}  {kind:8s} {ip}")
        for addr in addrs[:3]:
            print(f"      {addr}")
    print("  NOTE: this label comes from `list_peers`, which reports ONE address per peer")
    print("  and is NOT reliably the path carrying data -- measured 2026-08-02, a peer shown")
    print("  here as RELAYED was simultaneously moving 402 KB/request, 3.1x the relay")
    print("  budget, so provably direct. THE BYTE SWEEP BELOW IS THE AUTHORITY, not this.")


# One warm-up request so a route exists and connections are established.
with torch.inference_mode():
    model.generate(prompt, max_new_tokens=2, do_sample=False)
report("after warm-up")


def once(max_new):
    began = time.perf_counter()
    try:
        with torch.inference_mode():
            model.generate(prompt, max_new_tokens=max_new, do_sample=False)
        return ".", time.perf_counter() - began
    except Exception as exc:
        # A routing failure and a byte-budget stall look identical in a bare pass/fail
        # tally, and confusing them once turned 45 instant routing errors into a reported
        # "45/45 stalled". Kept apart here.
        return ("R" if "routing" in str(exc).lower() else "X"), time.perf_counter() - began


print(f"\n=== byte sweep (relay budget is {RELAY_BUDGET / 1024:.0f} KiB) ===")
print(f"{'tokens':>7} {'KB/req':>8} {'xbudget':>8}  result")
for max_new in (1, 8, 32, 64, 128):
    kb = (n_prompt + max_new) * HIDDEN * DTYPE_BYTES / 1024
    marks, oks = [], []
    for _ in range(TRIALS):
        mark, secs = once(max_new)
        marks.append(mark)
        if mark == ".":
            oks.append(secs)
    text = "".join(marks)
    median = f"{statistics.median(oks):.2f}s" if oks else "n/a"
    print(f"{max_new:>7} {kb:>8.0f} {kb * 1024 / RELAY_BUDGET:>7.1f}x  {text}  "
          f"ok={text.count('.')}/{TRIALS} stalled={text.count('X')} "
          f"routing={text.count('R')} median={median}")

report("after the sweep")

print("\n=== verdict ===")
final = connections()
direct = [p for p, a in final.items() if a and any("p2p-circuit" not in x for x in a)]
relayed = [p for p, a in final.items() if not a or all("p2p-circuit" in x for x in a)]
print(f"  {len(direct)} peer(s) reachable directly, {len(relayed)} relay-only")
if direct:
    print("  -> router-independent hosting WORKS for the direct ones. Those servers can")
    print("     host large models and can act as verification partners.")
if relayed:
    print("  -> the relay-only ones are capped at 128 KiB per request and cannot be")
    print("     verified. They need --public-ip (if their NAT is endpoint-independent) or")
    print("     a real port forward.")
print("\n  Anything above ~1.0x budget succeeding is proof the path is not a relay.")
dht.shutdown()
