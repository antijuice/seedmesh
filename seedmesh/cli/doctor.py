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
  * The remaining path is a circuit relay.

CORRECTED 2026-08-03. This module used to say a relay is severed at 128 KiB and that a
relay-only host is 'useless for a real model'. Both were measured on a ONE-bootstrap
swarm and neither holds here: a relay-only server has since hosted all 32 blocks of an
8B model on this swarm and answered real requests. What a relay-only host actually
loses is being a VERIFICATION partner -- it cannot be placed on a network, so nobody
can show it is independent of the peer it checks.

It also used to count bootstrap peers from swarm.json rather than probing them, which
meant one dead droplet made every machine on earth look symmetric-NAT'd.
"""

from __future__ import annotations

import ipaddress
import re
import time
from dataclasses import dataclass
from typing import List, Optional

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


def reachable_bootstrap_peers(peers, timeout: float = 6.0):
    """Which bootstrap peers actually accept a TCP connection right now. Returns (up, down).

    CONFIGURED IS NOT REACHABLE, and conflating the two produced the worst output this tool
    can produce: a confident wrong diagnosis. One of four droplets was down, so only three
    observers existed, so `ActivationThresh = 4` could never be met -- and `doctor`, which
    counted entries in swarm.json, reported symmetric-nat. It blamed a volunteer's router
    for a dead server while that volunteer's machine was serving an 8B model perfectly well.
    """
    import socket

    up, down = [], []
    for peer in peers:
        match = ADDR_RE.search(str(peer))
        if not match:
            continue
        host, port = match.group(1), int(match.group(2))
        try:
            with socket.create_connection((host, port), timeout=timeout):
                up.append(host)
        except Exception:
            down.append(host)
    return up, down


def diagnose(
    addresses: List[Address],
    n_peers: int,
    waited: float,
    timeout: float,
    unreachable: Optional[List[str]] = None,
) -> Diagnosis:
    """Turn observed addresses into an actionable verdict.

    Deliberately does not guess at causes it cannot distinguish. "No public address after
    the full timeout, with enough peers" has exactly one common explanation worth acting on,
    and saying so is more useful than listing four possibilities.
    """
    unreachable = list(unreachable or [])
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
        offline = f" ({', '.join(unreachable)} did not answer)" if unreachable else ""
        return Diagnosis(
            "too-few-peers",
            f"Only {n_peers} bootstrap peer(s) are REACHABLE{offline}, and\n"
            f"  {MIN_BOOTSTRAP_PEERS} are needed before a host can learn its own public\n"
            f"  address -- go-libp2p accepts an observed address only after four DISTINCT\n"
            f"  peers report it, so three can never be enough however good your network is.\n\n"
            f"  This is the swarm's problem, not yours. Nothing you change on this machine\n"
            f"  can produce a public address until a fourth bootstrap peer is back.\n\n"
            f"  Serving still works meanwhile: measured on this swarm, a relay-only server\n"
            f"  hosted all 32 blocks of an 8B model and answered real requests. What you\n"
            f"  lose is being usable as a verification partner, because a relayed peer\n"
            f"  cannot be placed on a network.",
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
        "    - You cannot act as a verification partner: a relayed peer cannot be placed\n"
        "      on a network, so nobody can show you are independent of who you check.\n\n"
        "  What this does NOT mean: that you cannot serve. Measured on this swarm, a\n"
        "  relay-only server hosted all 32 blocks of an 8B model and answered real\n"
        "  requests. An earlier version of this message said a relay is severed at\n"
        "  128 KiB, which was measured on a ONE-bootstrap swarm and does not hold here.\n\n"
        "  If you want direct reachability anyway, forward a TCP port and host with\n"
        "    seedmesh serve --host-maddrs /ip4/0.0.0.0/tcp/31338",
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
    # Probe before listening: "four peers configured" and "four peers answering" are
    # different facts, and only the second one can satisfy ActivationThresh.
    up, down = reachable_bootstrap_peers(peers)
    print(f"  {len(up)} of {len(peers)} bootstrap peer(s) answering"
          + (f"; unreachable: {', '.join(down)}" if down else ""))
    print(f"  listening on 0.0.0.0:{args.port}...")
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

    result = diagnose(addresses, len(up), waited, args.timeout, unreachable=down)
    print(f"\n=== {result.verdict} ===")
    print(f"  {result.detail}")
    return 0 if result.can_host_large_models else 1
