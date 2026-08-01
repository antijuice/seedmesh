"""Does DCUtR hole punching work between two genuinely remote NAT'd peers?

Run this on Colab, against a laptop server at home. That is the pair that matters and the
one no local test can produce: client and server on the same machine always find a direct
LAN path, so hole punching is never exercised.

What is already established (2026-08-01, measured):
  * A relayed connection is reset after ~128 KiB per direction -- go-libp2p's
    circuit-relay-v2 DefaultResources(). The stall period tracks bytes/request exactly.
  * DCUtR is compiled in and enabled: p2pd/main.go:345 calls libp2p.EnableHolePunching()
    whenever `-autonat=1`, which hivemind always passes.
  * But hivemind's fork also sets network.WithUseTransient(..., "relayEnabled"), overriding
    go-libp2p's refusal to run heavy streams over a relayed connection.

So the open question is which of these is true:
  (a) DCUtR succeeds, and the direct connection is simply not preferred for inference
      -> fix is a stream-policy patch in the daemon fork, which we can already build
  (b) DCUtR is attempted and fails (symmetric NAT / CGNAT)
      -> hole punching cannot be relied on; hosts need reachable addresses
  (c) DCUtR is never attempted
      -> a wiring bug, and the most fixable outcome

The daemon logs hole-punch activity under the `p2p-holepunch` subsystem. Env must be set
before any DHT is constructed, because p2pd inherits it at launch.
"""

import os

# MUST come before hivemind imports anything that starts a daemon.
os.environ["GOLOG_LOG_LEVEL"] = (
    "error,p2p-holepunch=debug,autonat=debug,relay=debug,swarm2=info,basichost=debug"
)
os.environ["HIVEMIND_LOGLEVEL"] = "DEBUG"

import logging  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from hivemind.dht import DHT  # noqa: E402
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from petals import AutoDistributedModelForCausalLM  # noqa: E402
from petals.utils.dht import get_remote_module_infos  # noqa: E402

# ALL FOUR, deliberately. go-libp2p accepts an observed public address only after four
# DISTINCT peers report it (identify/obsaddr.go: ActivationThresh = 4), and the hole-punch
# service stays dormant until then. Connecting to one bootstrap leaves DCUtR asleep on THIS
# end even though the server side is now awake -- both peers must have it running.
BOOTSTRAPS = [
    "/ip4/159.89.52.179/tcp/31337/p2p/QmTG981oPjsPFX5WWNegdXmkiNUgWqjUvK9spieg4hSi1h",
    "/ip4/137.184.60.251/tcp/31337/p2p/QmYECCgHhRRVTdeA9Mf3mhjiQWubgnqvbkpvL2GD6PUqdq",
    "/ip4/143.244.144.71/tcp/31337/p2p/QmWniG4aH2XHifkRDfjbEomWWTeMHja3DtSWrXURDM6ed4",
    "/ip4/104.248.51.133/tcp/31337/p2p/QmbeoRPB8ERB8D5FCtwxBBBRfTRhChqNewCvZ8PGMXYiF9",
]
MODEL = "JackFram/llama-160m"
PREFIX = os.environ.get("SEEDMESH_PREFIX", "llama-160m-hf")  # server uses the default
HIDDEN_BYTES = 768 * 4  # fp32 on the wire

# Capture the daemon's own log lines; hivemind forwards them to this logger.
punch_events = []


class PunchCollector(logging.Handler):
    def emit(self, record):
        text = record.getMessage()
        if any(k in text.lower() for k in ("holepunch", "hole punch", "dcutr", "direct connect")):
            punch_events.append(text[:200])


logging.getLogger("hivemind").addHandler(PunchCollector())

print(f"connecting to {len(BOOTSTRAPS)} bootstrap peers\n")
dht = DHT(initial_peers=BOOTSTRAPS, client_mode=True, start=True)

print("=== servers announced in the DHT ===")
infos = get_remote_module_infos(dht, [f"{PREFIX}.{i}" for i in range(12)], latest=True)
servers = {}
for info in infos:
    if info and info.servers:
        for peer_id, server in info.servers.items():
            servers[str(peer_id)] = server
for peer_id, server in servers.items():
    print(f"  {peer_id}")
    print(f"    state={server.state.name}  using_relay={getattr(server, 'using_relay', '?')}"
          f"  name={server.public_name}")
if not servers:
    print("  none -- start the laptop server first")
    dht.shutdown()
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoDistributedModelForCausalLM.from_pretrained(
    MODEL, initial_peers=BOOTSTRAPS, dht_prefix=PREFIX,
    torch_dtype=torch.float32, request_timeout=10, max_retries=1,
)
ids = tokenizer("What is the capital of France", return_tensors="pt")["input_ids"]
n_prompt = ids.shape[1]


def once(max_new):
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            model.generate(ids, max_new_tokens=max_new, do_sample=False)
        return ".", time.perf_counter() - start
    except Exception as exc:
        # R = DHT routing (instant), X = timeout (the byte budget). Never merge them.
        return ("R" if "routing: not found" in str(exc) else "X"), time.perf_counter() - start


print("\n=== warming up (peer routing records take minutes to propagate) ===")
for attempt in range(40):
    mark, secs = once(1)
    if mark == ".":
        print(f"routable after ~{attempt * 15}s ({secs:.2f}s)\n")
        break
    print(f"  not ready: {mark} ({secs:.2f}s)", flush=True)
    time.sleep(15)
else:
    print("never became routable")
    dht.shutdown()
    sys.exit(1)


async def connections():
    p2p = await dht.replicate_p2p()
    return await p2p.list_peers()


def report_connections(label):
    print(f"=== live connections ({label}) ===")
    try:
        peers = RemoteExpertWorker.run_coroutine(connections())
    except Exception as exc:
        print(f"  could not list: {exc}")
        return
    for peer in peers:
        addrs = [str(a) for a in peer.addrs]
        relayed = any("p2p-circuit" in a for a in addrs)
        print(f"  {'RELAYED' if relayed else 'DIRECT '} {str(peer.peer_id)[:22]}...")
        for a in addrs:
            print(f"      {a}")


report_connections("after warm-up")

# The byte sweep. From a genuinely remote peer, a relayed connection MUST show the
# budget: ~117 KB/request should fail nearly every time. If it does not, the connection
# was upgraded to direct.
print("\n=== byte sweep from a remote peer ===")
for max_new in (1, 8, 32):
    kb = (n_prompt + max_new) * HIDDEN_BYTES / 1024
    pattern, oks = [], []
    for _ in range(12):
        mark, secs = once(max_new)
        pattern.append(mark)
        if mark == ".":
            oks.append(secs)
    text = "".join(pattern)
    median = f"{statistics.median(oks):.2f}s" if oks else "n/a"
    print(f"  max_new={max_new:>2} (~{kb:>3.0f} KB/req): {text}  "
          f"ok={text.count('.')}/12 timeouts={text.count('X')} routing={text.count('R')} "
          f"median={median}")

report_connections("after the sweep")

print("\n=== did DCUtR start on THIS end? ===")
started = [e for e in punch_events if "Starting holepunch protocol" in e]
print(f"  {'YES' if started else 'NO -- still waiting for a public address'}")
print("  (the server side started it within 15s once it had four observers)")

print("\n=== hole-punch / DCUtR events seen ===")
if punch_events:
    for line in punch_events[-25:]:
        print(f"  {line}")
else:
    print("  NONE -- DCUtR never logged anything, which points at (c): never attempted")

print("\n=== how to read this ===")
print("  budget stalls present + RELAYED   -> hole punching did not produce a usable path")
print("  no stalls + DIRECT                -> DCUtR worked; the fix is stream policy")
print("  no stalls + RELAYED               -> unexpected; re-check the byte arithmetic")
dht.shutdown()
