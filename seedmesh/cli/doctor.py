"""`seedmesh doctor` -- can this machine host, and if not, exactly why?

The failure this exists for is silent. A volunteer behind a symmetric NAT or CGNAT starts a
server, sees no errors, and hosts blocks nobody can use. Nothing in the logs says why, and
the honest remedy (a port forward) is never suggested because nothing detected the problem.

The mechanism, measured and traced through go-libp2p v0.32.1:

  * A host learns its own public address from `identify`: peers report the address they see
    you at. `identify/obsaddr.go` accepts one only after **four distinct peers** report the
    SAME address (`ActivationThresh = 4`).
  * On an endpoint-independent ("full cone") NAT every peer sees the same external port, so
    four observers agree and the address is accepted.
  * On a **symmetric** NAT the external port differs per destination, so four observers
    report four different addresses and none ever reaches the threshold.
  * Without a public address, `holepunch/svc.go` never registers `/libp2p/dcutr` (svc.go:111)
    and `holepuncher.go:215` refuses to initiate -- "aborting hole punch initiation as we
    have no public address". So hole punching cannot rescue this peer in either role.
  * The remaining path is a circuit relay, which go-libp2p severs after 128 KiB -- fine for a
    toy model, useless for a real one.

So the diagnosis is binary and worth stating plainly: either this host learns a public
address, or it needs a port forward to be useful for anything larger than a demo.
"""

from __future__ import annotations

import ipaddress
import re
import time
from dataclasses import dataclass
from typing import List

# Below this many bootstrap peers, a public address cannot be learned even on a friendly NAT
# -- there are not enough observers to reach ActivationThresh.
from seedmesh.cli.swarm import MIN_BOOTSTRAP_PEERS

ADDR_RE = re.compile(r"/ip4/(\d{1,3}(?:\.\d{1,3}){3})/tcp/(\d+)")


@dataclass(frozen=True)
class Address:
    ip: str
    port: int

    @property
    def kind(self) -> str:
        try:
            parsed = ipaddress.ip_address(self.ip)
        except ValueError:
            return "invalid"
        if parsed.is_loopback:
            return "loopback"
        if parsed.is_global:
            return "public"
        return "private"


def parse_addresses(maddrs) -> List[Address]:
    """Pull (ip, port) out of whatever multiaddr representation we are handed."""
    found = []
    for maddr in maddrs:
        match = ADDR_RE.search(str(maddr))
        if match:
            found.append(Address(match.group(1), int(match.group(2))))
    return found


@dataclass(frozen=True)
class Diagnosis:
    verdict: str
    detail: str
    can_host_large_models: bool


def diagnose(
    addresses: List[Address], n_peers: int, waited: float, timeout: float
) -> Diagnosis:
    """Turn observed addresses into an actionable verdict.

    Deliberately does not guess at causes it cannot distinguish. "No public address after
    the full timeout, with enough peers" has exactly one common explanation worth acting on,
    and saying so is more useful than listing four possibilities.
    """
    public = [a for a in addresses if a.kind == "public"]

    if public:
        ports = sorted({a.port for a in public})
        return Diagnosis(
            "reachable",
            "This host knows its own public address: "
            + ", ".join(f"{a.ip}:{a.port}" for a in public)
            + f"\n  Other peers can dial you on port {ports[0]}. You can host any model size.",
            True,
        )

    if n_peers < MIN_BOOTSTRAP_PEERS:
        return Diagnosis(
            "too-few-peers",
            f"Only {n_peers} bootstrap peer(s) reachable, and {MIN_BOOTSTRAP_PEERS} are\n"
            f"  needed before a host can learn its own public address (go-libp2p accepts an\n"
            f"  observed address only after four distinct peers report it).\n"
            f"  This is a swarm configuration problem, not a problem with your network.",
            False,
        )

    if waited < timeout:
        return Diagnosis(
            "still-waiting",
            "No public address yet, but the wait was cut short. Re-run and let it finish.",
            False,
        )

    return Diagnosis(
        "symmetric-nat",
        f"{n_peers} peers connected, but after {waited:.0f}s none agreed on a public address\n"
        "  for this host. That is the signature of a symmetric NAT or carrier-grade NAT:\n"
        "  your router assigns a different external port per destination, so no two peers\n"
        "  see the same address and the threshold is never met.\n\n"
        "  Consequences, stated plainly:\n"
        "    - Hole punching cannot help. go-libp2p refuses to initiate without a public\n"
        "      address, and never registers the responder handler either.\n"
        "    - Your only remaining path is a circuit relay, which is severed after 128 KiB.\n"
        "      That is under one request for any real model.\n\n"
        "  What actually works: forward a TCP port to this machine and host with\n"
        "    seedmesh serve --host-maddrs /ip4/0.0.0.0/tcp/31338\n"
        "  You can still USE the swarm as a client with no changes at all.",
        False,
    )


def cmd_doctor(args) -> int:
    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(args.initial_peers, args.swarm_file)
    if note:
        print(note)
    if not peers:
        print("no bootstrap peers to test against; pass --initial-peers or --swarm-file")
        return 2

    print("\nchecking whether this machine can be dialled by other peers")
    print(f"  listening on 0.0.0.0:{args.port}, connecting to {len(peers)} bootstrap peer(s)...")
    print("  (this listens like a real server -- a client-mode check cannot answer this)")

    try:
        from hivemind.dht import DHT
    except ImportError as exc:
        print(f"  backend not installed ({exc}); run `seedmesh setup`")
        return 2

    # MUST listen, exactly as a server would. The first version used client_mode=True, which
    # makes hivemind pass -noListenAddrs=1 to the daemon: the host has nothing to be dialled
    # ON, so no peer can ever observe it at a public address and the check reports
    # symmetric-NAT for every machine on earth. Caught by running it against a host already
    # known to be directly reachable, which it confidently misdiagnosed.
    dht = DHT(
        initial_peers=peers,
        client_mode=False,
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.port}"],
        start=True,
    )
    try:
        deadline = time.monotonic() + args.timeout
        addresses: List[Address] = []
        last_report = 0.0
        while time.monotonic() < deadline:
            addresses = parse_addresses(dht.get_visible_maddrs())
            if any(a.kind == "public" for a in addresses):
                break
            waited = args.timeout - (deadline - time.monotonic())
            if waited - last_report >= 15:
                last_report = waited
                print(f"  {waited:.0f}s: no public address yet "
                      f"({len(addresses)} address(es) visible)")
            time.sleep(2)
        waited = args.timeout - max(0.0, deadline - time.monotonic())
    finally:
        dht.shutdown()

    print("\naddresses this host advertises:")
    for address in addresses or []:
        print(f"  {address.ip}:{address.port}  ({address.kind})")
    if not addresses:
        print("  (none)")

    result = diagnose(addresses, len(peers), waited, args.timeout)
    print(f"\n=== {result.verdict} ===")
    print(f"  {result.detail}")
    return 0 if result.can_host_large_models else 1
