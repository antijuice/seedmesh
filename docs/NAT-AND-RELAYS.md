# Hosting from behind NAT (a laptop on wifi)

Most volunteers are on a home connection with no port forwarding. This is the normal case,
not an edge case, and it works — but it is worth understanding what happens, because the
failure modes look confusing from the outside.

## What happens automatically

On startup a server tests whether other peers can dial it back
(`check_direct_reachability`). If they cannot, it:

1. joins the DHT in **client mode** (it queries but is not dialled), and
2. becomes reachable through **libp2p circuit relays**, found automatically via the DHT.

Both are on by default (`use_relay=True`, `use_auto_relay=True`). The log line to look for:

```
This server is accessible via relays
```

versus `accessible directly` if you do have an open port. Either is fine; the first is what
a laptop on wifi should print.

## What this needs from the swarm

**At least one publicly reachable peer**, which is what the bootstrap VPS is for
([BOOTSTRAP.md](BOOTSTRAP.md)). Relays are peers too — a swarm of exclusively NAT'd nodes has
nothing to relay *through* and cannot form. One reachable box is enough to unblock any number
of NAT'd volunteers.

## Costs, honestly

Relayed traffic takes an extra hop through a third machine, so a relayed server is slower
than a directly-reachable one and consumes the relay's bandwidth. For a small friends-and-
family swarm this is fine. At scale it is a reason to want several reachable peers rather
than one.

Seedmesh's reputation layer sees relayed servers as simply slower — latency feeds the score,
and `latency_factor` is deliberately bounded so a slow honest server still outranks a fast
dishonest one. A relayed volunteer is not penalised as though they were faulty.

## If you *can* forward a port

Better, but not required:

```bash
seedmesh serve --model <m> --initial-peers <addr> \
  -- --public_ip YOUR_PUBLIC_IP --port 31337
```

Forward 31337/tcp to your machine in your router. You will then see `accessible directly`.

## Troubleshooting

**"Server has not become reachable from the Internet"** — this error, which points at
`health.petals.dev` and a Discord link, only fires when joining the *public* Petals swarm.
That swarm is offline and the health endpoint is down. If you see it, you are missing
`--initial-peers` and defaulted to the public bootstrap list. Pass your swarm's address.

**Server starts but the client never routes to it** — check the client can actually see your
blocks. Discovery reports what is announced; if your peer appears with no blocks, it joined
but failed to load them (look further up its log).

**Everything is slow** — expected if several peers are relayed through one bootstrap. Adding
a second reachable peer helps more than anything else.

**Colab specifically** — works as a client and, via relays, as a server. But sessions time
out and its terms discourage long-running non-interactive workloads, so treat it as a
test node rather than a contributor. Note also that Colab egress is Google's AS15169, so two
Colab nodes count as the *same* network cluster and cannot verify each other.

## Measured: relayed hosting is unreliable (2026-08-01)

Relays work, but not well. Measured against a real NAT'd laptop server hosting all 12 blocks
of `JackFram/llama-160m`, reached through a VPS bootstrap, `using_relay=True`:

| | |
| --- | --- |
| healthy request latency | median **0.88s**, max **3.93s** (n=14) |
| stall rate | **~1 request in 3** (6/20 at a 60s timeout, 14/30 at 8s) |
| do stalls recover? | **never** — they sit until the timeout, whatever it is set to |
| does an immediate fresh attempt work? | **yes, 14/14**, in ~1.2s |

Two consequences:

**A long timeout is actively harmful.** Since a stalled request never completes, the timeout
only decides how long a doomed request blocks. Petals' 180s default turns a one-second retry
into a three-minute hang. `seedmesh chat` defaults to 10s for this reason, and retries
with a fresh session rather than waiting.

**A stall that lands mid-generation still kills the request.** Petals' recovery path rebuilds
the server session at position 0 while the client is mid-sequence, so it raises
`AssertionError('0 and N')`. Setting the position instead trips the session's pre-allocated
length (`prefix 16 + current 17 exceeds pre-allocated maximum 23`), because regeneration
resends the whole prefix. Fixing it properly means changing how the outer step loop re-feeds
inputs after a failure — not just the session bookkeeping. Untouched for now, and the reason
roughly one prompt in three still fails on a relayed swarm.

A direct connection avoids all of it — see *The real constraint* below for how to get one,
and why "just forward a port" is not the whole answer on Windows.

## Located: the stall is on the relay return path, every 3rd request

Investigated 2026-08-01 with logs on both ends (own server behind the same NAT, own client,
separate `--dht_prefix` so the live swarm was untouched).

**The request arrives and the server answers it. The response is lost.** Matching client
stalls against the server's own `rpc_inference` log:

```
client trial 2   STALL 8.62s
server           15:40:01.981 open  ->  15:40:02.563 close   (0.58s, clean)
```

`opens=19 closes=19` across the run — every request, including the stalled ones, reached the
server and was completed and closed normally in about half a second. The client simply never
received the answer.

### What it is invariant to

Each of these was measured, and none of them changes the pattern `..X..X..X..X`:

| varied | result |
| --- | --- |
| prompt content and length (2–16 tokens) | no effect |
| timeout (8s vs 60s) | no effect — stalls hit whatever wall is set, never recover |
| spacing between requests (0s vs 6s) | **identical pattern** — not a cleanup or timing effect |
| reusing one session vs a fresh one per prompt | still every 3rd request |
| **direct connection instead of a relay** | **0 stalls in 30** |

So: relay-only, strictly every 3rd request, on the return path. Not a configured limit —
p2pd's relevant defaults are `connHi 512`, `relayMaxCircuits 16`, `relayDataLimit 4 GB`,
`relayTimeLimit 30m`, none of which is 3.

### Located: go-libp2p's 128 KiB circuit-relay budget

With `GOLOG_LOG_LEVEL` raised on both daemons, the relayed connection turns out to be
**silently discarded and re-dialled**, three requests apart:

```
16:04:02.865 rpc_inference.close   (trial 3, OK)
16:04:03.653 rpc_inference.close   (trial 4, OK)
16:04:04.262 rpc_inference.close   (trial 5, STALLED -- server still completed it)
16:04:12.292 swarm dialing ... /p2p-circuit    <- brand-new circuit, after the timeout
```

No close, no reset, no resource-manager message. go-libp2p's circuit-relay-v2
`DefaultResources()` caps a relayed connection at **128 KiB per direction** and resets it
when the budget runs out; nothing above the transport is told, so the in-flight request just
never gets an answer.

That predicts the stall period should track *bytes*, not requests. For llama-160m (hidden
768, fp32 = 3072 bytes/token), bytes/request = (prompt + new tokens) × 3072:

| `max_new_tokens` | KB/request | predicted period | **measured** |
| --- | --- | --- | --- |
| 1 | 24 | 5.3 | **6.0** |
| 8 | 45 | 2.8 | **3.0** |
| 32 | 117 | 1.1 | **1.0 (15/15 stalled)** |

Back-solving from all three gives ~130–145 KB, i.e. 128 KiB plus framing. Confirmed.

### Why this is worse than it looks

Bytes/request scales with hidden size and dtype. **Llama-3.1-8B is hidden 4096 at bf16 =
8192 bytes/token**, so one 30-token request moves ~245 KB — past the budget *within a single
request*, every request, where retrying cannot help. Relayed hosting of a real model is
impossible on this budget, not merely flaky.

A 128 KiB budget is sized for bootstrapping a direct connection via hole punching, and
**p2pd exposes no hole-punching (DCUtR) flag** — so relayed connections stay relayed forever
and never escape it.

### The budget cannot be raised. Do not try again.

p2pd accepts `-relayDataLimit` and `-relayTimeLimit`, and hivemind can be made to pass them
by wrapping `P2P._make_process_args`. **This was built, deployed to a real relay, and
measured three times. It changes nothing** — byte-for-byte identical stall patterns with and
without the flags.

The tell was visible beforehand and was missed: p2pd's own help advertises
`-relayDataLimit ... (default 4294967296)`, i.e. 4 GiB. If that default were reaching the
relay service the observed budget would never have been 128 KiB. p2pd parses these flags but
never applies them to go-libp2p's relay resources, so `DefaultResources()` stands. The shim
was removed rather than left in place claiming a budget it never got.

## The real constraint: how many NATs is the host behind?

Since the relay budget is fixed and small, a volunteer hosting anything larger than a toy
model needs a **direct** address. p2pd already tries for one — it is launched with
`-natPortMap=1` (UPnP / NAT-PMP). Why that fails is platform-specific, and the answer
matters more than anything else in this document.

**Hosting from WSL2 is behind two NATs.** Measured on Windows 11 with WSL 2.5.7:

```
Running a server on ['/ip4/127.0.0.1/tcp/44945/p2p/...',
                     '/ip4/172.25.114.109/tcp/44945/p2p/...']
Direct reachability: 0/1
This server is accessible via relays
```

`172.25.114.109` is WSL2's *virtual switch*, not the machine's LAN address. So UPnP
negotiates with the Hyper-V virtual gateway rather than the router, and a router
port-forward would land on the Windows host without ever entering WSL. Port forwarding alone
does not fix a WSL2 host; it also needs a `netsh interface portproxy` rule on the Windows
side.

**Fix for Windows hosts: mirrored networking.** WSL ≥ 2.0.0 on Windows 11 22H2+ can share the
host's interfaces directly, removing the WSL NAT layer entirely. In `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

then `wsl --shutdown` and reopen. WSL then holds the machine's real LAN address, so UPnP can
reach the actual router and a port-forward arrives where it should. Caveats: mirrored mode
interacts badly with some VPN clients, and Windows Firewall may need an inbound allow rule.

**The ladder, best first:**

| | who | effort |
| --- | --- | --- |
| 1. Native Linux or macOS host | one NAT; UPnP often just works | none |
| 2. WSL2 + `networkingMode=mirrored` | Windows volunteers | one config line + restart |
| 3. Router port forward (+ `netsh portproxy` if still on WSL NAT) | anyone whose router lacks UPnP | real setup |
| 4. Relay | everyone else | works, but capped at 128 KiB/connection — toy models only |

### Measured after mirrored networking + pinned port + firewall rule (2026-08-01)

Same laptop, server pinned to `0.0.0.0:31338`, Windows Firewall inbound allowed:

| max_new_tokens | KB/request | before | after |
| --- | --- | --- | --- |
| 1 | 24 | 13/15 | **15/15** |
| 8 | 45 | 10/15 | **15/15** |
| 32 | 117 | **0/15** | **15/15** |

The byte-budget stalls are gone — but read the server's own record before concluding
anything:

```
using_relay : True
```

`using_relay` is `reachable_via_relay`, set once at startup by `check_direct_reachability`
(can the bootstrap dial me back?). `True` means **the server is still not publicly
reachable**. The sweep passed because the client was on the *same LAN* and dialled
`10.0.0.155:31338` directly, bypassing the relay entirely.

So what this proves: mirrored networking + a pinned port + a firewall rule make the direct
path work **for same-LAN clients**. What it does not prove: anything about a remote peer,
who still gets the relay and still hits the 128 KiB budget.

**A firewall rule is not enough on its own.** Inbound has to clear three things: the Windows
firewall (done), the WSL NAT layer (removed by mirrored mode), and **the router**, which must
map the public IP's port 31338 to the machine. UPnP is enabled in p2pd (`-natPortMap=1`) and
is evidently not getting that mapping, so it needs either UPnP switched on in the router or a
manual forward.

The check that matters: restart the server and look for `accessible directly` instead of
`accessible via relays`. Until that flips, remote volunteers are relayed.

Note also the warm-up window varies — measured at ~180-195s repeatedly, but ~45s in this
run. Treat it as "up to a few minutes", not a constant.


### Why the mitigation is the right shape anyway

Because the server completes the work and only the reply is lost, a retry is cheap and
correct rather than a papering-over: `seedmesh chat` uses a short timeout and retries with a
fresh session (`generate_with_retry`). Measured 12/12 prompts answered.

**Do not "fix" this by holding one session open across prompts.** It looks like the obvious
optimisation and it is actively worse: once a stall lands inside a shared session, the
session is permanently poisoned with `AssertionError: Broken input cache` and every later
prompt in it fails. Measured 10/12 failures versus 4/12.

## Root cause: a swarm needs at least 4 publicly reachable peers

Everything above — the 128 KiB budget, the stalls, the "just forward a port" advice —
descends from one deployment fact. Measured 2026-08-01 with hole-punch logging on both a
home-NAT laptop server and a Colab client:

```
[holepunch/svc.go:98] waiting until we have at least one public address
```

on **both** ends, forever. go-libp2p v0.32.1 explains it exactly:

- `holepunch/svc.go` — `watchForPublicAddr()` blocks until the host observes a public address
  for itself; the hole-punch handler waits on that channel. No public address, no DCUtR.
- `identify/obsaddr.go` — `var ActivationThresh = 4`, and an address is accepted only when
  `len(oa.seenBy) >= ActivationThresh`: **four distinct peers must independently report the
  same observed address.**

This swarm has one public peer. Four are needed. So no NAT'd host ever learns its own public
address, DCUtR never starts, every connection stays relayed, and relayed connections are cut
off at 128 KiB.

**The fix is deployment, not code.** Run four or more publicly reachable peers and volunteers
behind NAT should upgrade to direct connections on their own — no router configuration by
anyone. This is a source-derived prediction, not yet measured; DCUtR must still traverse the
real NATs, and symmetric NAT or CGNAT can defeat it regardless. Note also
`maxObservedAddrsPerIPAndTransport = 2`, so four peers sharing one IP may not count as four
observers.
