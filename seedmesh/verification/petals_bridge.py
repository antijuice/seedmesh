"""Wiring inline verification to a live Petals client.

`InlineVerifier` takes its two Petals-shaped operations as callables so the sampling policy
and the comparison stay testable without a backend. This is where those callables are built
from a running `seedmesh chat`.

Two things are needed:

* **discover** — who is hosting what, as Seedmesh `ServerInfo` (with compute profile and
  address, since the sampler needs both to pick an *independent, comparable* verifier).
* **run_on** — run a named block range on a *named* peer. Petals' own client picks whichever
  route it likes; redundant verification is meaningless unless the second execution goes
  where we say.

The DHT read is borrowed from `seedmesh.cli.monitor`, which already decodes announcements
including the compute profile, rather than reimplemented a third time.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from seedmesh.core.types import BlockRange, ComputeProfile, PeerId, ServerInfo


def _profile_of(announcement: dict) -> ComputeProfile:
    # "unknown" rather than a guessed default: an unpatched server does not report attn_impl,
    # and assuming it matches ours would judge the pair against a tolerance calibrated for a
    # combination neither of them is running. An uncalibrated pair must refuse to verify.
    return ComputeProfile(
        quant=str(announcement.get("quant_type") or "none").lower(),
        dtype=str(announcement.get("torch_dtype") or "float32").replace("torch.", ""),
        attn=str(announcement.get("attn_impl") or "unknown"),
    )


def make_discover(
    dht, block_uids: List[str], model_id: str, *, addresses: Optional[Callable] = None
) -> Callable[[], List[ServerInfo]]:
    """A callable returning who is online, as Seedmesh server records."""
    from seedmesh.cli.monitor import _read_blocks

    first_seen: Dict[PeerId, float] = {}

    def discover() -> List[ServerInfo]:
        per_block = _read_blocks(dht, block_uids)
        now = time.time()
        known = addresses() if addresses is not None else {}

        spans: Dict[PeerId, dict] = {}
        for index, servers in enumerate(per_block):
            for peer_id, announcement in (servers or {}).items():
                if announcement.get("state") != "ONLINE":
                    continue
                entry = spans.setdefault(
                    peer_id,
                    {"start": index, "end": index + 1, "announcement": announcement},
                )
                entry["start"] = min(entry["start"], index)
                entry["end"] = max(entry["end"], index + 1)

        results = []
        for peer_id, entry in spans.items():
            first_seen.setdefault(peer_id, now)
            results.append(
                ServerInfo(
                    peer_id=peer_id,
                    block_range=BlockRange(model_id, entry["start"], entry["end"]),
                    first_seen=first_seen[peer_id],
                    address=known.get(peer_id),
                    advertised_throughput=float(
                        entry["announcement"].get("throughput") or 0.0
                    ),
                    profile=_profile_of(entry["announcement"]),
                )
            )
        return results

    return discover


def make_address_resolver(sequence_manager) -> Callable[[], Dict[PeerId, str]]:
    """Peer id -> IP, from the p2p daemon's view of who it is connected to.

    Petals' DHT records carry no address, and without one every peer lands in the shared
    "unknown" cluster -- where the sampler correctly refuses to pair anyone, because it
    cannot show two peers are independent operators. So this is what makes verification
    possible at all on a real swarm, not a nicety.

    A peer that has announced blocks but has not been dialled yet has no address here, and
    stays unpairable until it does.

    **Relayed multiaddrs are discarded, and this is a security property, not tidiness.**
    A circuit address looks like::

        /ip4/143.244.144.71/tcp/31337/p2p/QmRelay.../p2p-circuit

    The IP in it belongs to the *relay*, not to the peer. Measured on the live swarm: both
    servers resolved to bootstrap-droplet IPs this way, and neither was the machine actually
    hosting the blocks. Taking that at face value breaks the diversity rule in the direction
    that matters -- one operator running two NAT'd servers relayed through two different
    droplets would look like two independent networks and could verify its own work, which
    is exactly the collusion the pairing rule exists to prevent. (The other direction, two
    servers behind one relay looking related, merely refuses a legitimate pair.)

    So a peer reachable only through a relay stays unlocated, lands in the unknown cluster,
    and is refused as a verification partner. Verification becomes possible for that peer
    when it becomes directly reachable -- the same requirement as hosting a real model.
    """
    import re

    ADDR = re.compile(r"^/ip4/(\d{1,3}(?:\.\d{1,3}){3})/")

    def rank(address: str) -> tuple:
        first = address.split(".")[0]
        return (0 if first not in ("10", "127", "172", "192") else 1, address)

    # Accumulated across calls, never recomputed from scratch. `list_peers` reports ONE
    # address per peer and it is not reliably the path carrying data: measured 2026-08-02,
    # a Colab client moved 402 KB/request to a server -- 3.1x the relay budget, so provably
    # direct -- while `list_peers` showed only that peer's circuit address. A single
    # observation therefore under-reports, and under-reporting means refusing to verify a
    # peer that is in fact placeable. Remembering the best address ever seen fixes the
    # common case (a connection that starts relayed and upgrades) without ever inventing one.
    best: Dict[PeerId, str] = {}

    def resolve() -> Dict[PeerId, str]:
        found: Dict[PeerId, str] = {}
        try:
            from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker

            peers = RemoteExpertWorker.run_coroutine(
                sequence_manager.state.p2p.list_peers()
            )
        except Exception:
            return found
        for peer in peers:
            candidates = []
            for maddr in getattr(peer, "addrs", []) or []:
                text = str(maddr)
                if "p2p-circuit" in text:
                    continue  # tells us where the relay is, not where the peer is
                match = ADDR.match(text)
                if match:
                    candidates.append(match.group(1))
            if candidates:
                found[str(peer.peer_id)] = sorted(candidates, key=rank)[0]
        best.update(found)
        return dict(best)

    return resolve


def make_runner(sequence_manager, block_uids: List[str]) -> Callable:
    """A callable that runs a block range on exactly one named peer.

    Mirrors `PetalsBackend.run_segment`. Stateless `rpc_forward`, which is why only prefill
    steps are replayable -- see `seedmesh/verification/inline.py`.
    """

    def run_on(
        verifier: ServerInfo, block_range: BlockRange, hidden: np.ndarray, nonce: bytes
    ) -> Optional[np.ndarray]:
        import torch
        from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
        from hivemind.p2p import PeerID
        from hivemind.utils.serializer import MSGPackSerializer

        from petals.client.remote_forward_backward import run_remote_forward
        from petals.data_structures import CHAIN_DELIMITER
        from petals.server.handler import TransformerConnectionHandler
        from petals.utils.misc import DUMMY
        from petals.utils.packaging import pack_args_kwargs

        inputs = torch.as_tensor(np.asarray(hidden), dtype=torch.float32)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)

        span_uids = CHAIN_DELIMITER.join(block_uids[block_range.start : block_range.end])
        peer = PeerID.from_base58(verifier.peer_id)
        stub = TransformerConnectionHandler.get_stub(sequence_manager.state.p2p, peer)
        flat_tensors, args_structure = pack_args_kwargs(inputs, DUMMY)
        metadata = sequence_manager.get_request_metadata(
            "rpc_forward", args_structure, span_uids, *flat_tensors
        )

        (outputs,) = RemoteExpertWorker.run_coroutine(
            run_remote_forward(
                span_uids,
                stub,
                sequence_manager.rpc_info,
                *flat_tensors,
                config=sequence_manager.config,
                metadata=MSGPackSerializer.dumps(metadata),
            )
        )
        return outputs.detach().cpu().float().numpy()

    return run_on


def sequence_manager_of(model):
    """Dig the RemoteSequenceManager out of a distributed model, whatever it is called."""
    for path in ("model.layers", "transformer.h", "model.h"):
        target = model
        for part in path.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        manager = getattr(target, "sequence_manager", None)
        if manager is not None:
            return manager
    return None
