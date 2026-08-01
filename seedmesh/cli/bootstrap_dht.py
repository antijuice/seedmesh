"""Run Petals' DHT bootstrap peer with the relay's data budget raised.

Why this shim exists
--------------------
go-libp2p's circuit-relay-v2 gives each relayed connection a small budget and silently
resets it when the budget runs out. Neither Petals nor hivemind notices, so the in-flight
request simply never gets an answer and the client waits out its timeout.

Measured against a real two-host swarm on 2026-08-01 (llama-160m, hidden 768, fp32 =
3072 bytes/token, so bytes/request = (prompt + new tokens) x 3072):

    max_new_tokens=1    ~24 KB/request   ->  1 stall every  6.0 requests
    max_new_tokens=8    ~45 KB/request   ->  1 stall every  3.0 requests
    max_new_tokens=32  ~117 KB/request   ->  every request stalled (15/15)

Back-solving the budget from all three gives ~130-145 KB, i.e. go-libp2p's
`DefaultResources()` limit of 128 KiB per direction. The period is set by bytes relayed,
not by request count, elapsed time, prompt content, or session reuse -- each of those was
varied and none changed it. On a direct (non-relayed) connection the stall never occurs:
0 in 30.

Why it matters more than the toy numbers suggest: bytes/request scales with hidden size and
dtype. Llama-3.1-8B is hidden 4096 at bf16 = 8192 bytes/token, so a single 30-token request
moves ~245 KB -- past the budget *within one request*, every request, where retrying cannot
help. A 128 KiB budget is sized for bootstrapping a direct connection via hole punching, and
p2pd exposes no hole-punching (DCUtR) flag, so relayed connections stay relayed forever.

p2pd does accept `-relayDataLimit` and `-relayTimeLimit`, but hivemind assembles its p2pd
argv from a fixed keyword set with no passthrough, so they are unreachable from the public
API. `P2P._make_process_args` turns each keyword into `-key=value`, so wrapping it is enough.

This only affects the bootstrap process Seedmesh launches. Nothing else is patched, and the
limits are the relay's to grant: the relay advertises them in the reservation, so raising
them here fixes every volunteer behind a NAT without any of them touching their router.
"""

from __future__ import annotations

import sys

# 4 GiB per direction and a 24h ceiling: p2pd documents these as its own flag defaults
# ("default 4294967296", "default 30m0s"), so this is asking for the limit p2pd already
# claims rather than inventing a new one. The relay only forwards activations for peers it
# is already relaying for, and the DHT bootstrap does no other work.
RELAY_DATA_LIMIT = 4 * 1024**3
RELAY_TIME_LIMIT = "24h"


def raise_relay_budget(data_limit: int = RELAY_DATA_LIMIT, time_limit: str = RELAY_TIME_LIMIT) -> None:
    """Make every p2pd this process launches serve relayed connections a real budget."""
    from hivemind.p2p.p2p_daemon import P2P

    if getattr(P2P._make_process_args, "_seedmesh_patched", False):
        return

    original = P2P._make_process_args

    def with_relay_budget(*args, **kwargs):
        # setdefault, not assignment: an explicit caller value wins.
        kwargs.setdefault("relayService", 1)
        kwargs.setdefault("relayDataLimit", data_limit)
        kwargs.setdefault("relayTimeLimit", time_limit)
        return original(*args, **kwargs)

    with_relay_budget._seedmesh_patched = True
    P2P._make_process_args = staticmethod(with_relay_budget)


def main() -> int:
    raise_relay_budget()
    print(
        f"relay budget raised: {RELAY_DATA_LIMIT // 1024**3} GiB per direction, "
        f"{RELAY_TIME_LIMIT} per connection",
        flush=True,
    )
    print(
        "  (go-libp2p defaults to 128 KiB, which silently resets a relayed connection "
        "mid-request)",
        flush=True,
    )

    from petals.cli.run_dht import main as run_dht_main

    run_dht_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
