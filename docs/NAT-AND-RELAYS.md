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
into a three-minute hang. `seedmesh chat` defaults to 30s for this reason.

**A stall that lands mid-generation still kills the request.** Petals' recovery path rebuilds
the server session at position 0 while the client is mid-sequence, so it raises
`AssertionError('0 and N')`. Setting the position instead trips the session's pre-allocated
length (`prefix 16 + current 17 exceeds pre-allocated maximum 23`), because regeneration
resends the whole prefix. Fixing it properly means changing how the outer step loop re-feeds
inputs after a failure — not just the session bookkeeping. Untouched for now, and the reason
roughly one prompt in three still fails on a relayed swarm.

### So: forward the port if you can

If you are hosting and can forward **TCP 31337** to your machine on your router, do it.
Petals' `check_direct_reachability` will then succeed, the server runs as a full peer instead
of `client_mode`, and none of the above applies — no relay, no stalls. Relaying is the
fallback for volunteers who cannot change their router, and it should be described to them as
"works, intermittently" rather than "works".
