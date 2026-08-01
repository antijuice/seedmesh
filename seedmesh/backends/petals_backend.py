"""Petals adapter: the seam between Seedmesh's trust layer and real distributed inference.

This is the first thing that puts reputation and verification in front of actual servers
rather than a simulator. It implements the three methods of
:class:`~seedmesh.backends.base.InferenceBackend`:

``discover``     hivemind DHT lookup of who is hosting which blocks
``n_blocks``     the model's layer count
``run_segment``  one forward pass of one block range on **one named server**

That last one is what makes verification possible at all. Petals' own client picks a route
and streams through it; Seedmesh needs to send *the same input to a specific server* so two
servers' outputs can be compared. `run_remote_forward` is the primitive that allows it.

Requires a ported Petals checkout (``tools/port_petals.py``) and therefore Linux/WSL. It is
deliberately **not** imported by ``seedmesh.backends.__init__`` -- the core must stay
installable, and testable, without torch or petals.
"""

from __future__ import annotations

import ipaddress
import time
from typing import Optional, Sequence

import numpy as np

from seedmesh.backends.base import SegmentResult
from seedmesh.core.types import BlockRange, ComputeProfile, Outcome, PeerId, ServerInfo
from seedmesh.verification.sketch import Sketch, SketchSpec, derive_seed, sketch_hidden_state


class PetalsBackend:
    """Drives a Petals swarm through the Seedmesh backend interface."""

    def __init__(
        self,
        model_name: str,
        initial_peers: Sequence[str],
        *,
        request_timeout: float = 30.0,
        max_retries: int = 2,
        sketch_spec: Optional[SketchSpec] = None,
    ) -> None:
        import torch
        from hivemind.dht import DHT

        from petals.client.routing import RemoteSequenceManager
        from petals.utils.auto_config import AutoDistributedConfig

        self._torch = torch
        self.model_name = model_name
        self.sketch_spec = sketch_spec or SketchSpec()

        # The DHT must exist before anything triggers transformers' low-cpu-mem init
        # context: hivemind allocates a shared-memory buffer on first DHT construction, and
        # `share_memory_()` fails inside that context. Building it first also warms the
        # buffer for any model the caller creates later. See docs/petals-port.md.
        self.dht = DHT(initial_peers=list(initial_peers), client_mode=True, start=True)

        self.config = AutoDistributedConfig.from_pretrained(model_name)
        self.config.allowed_servers = None

        # Petals defaults to `max_retries=None`, i.e. retry forever, with a 180s request
        # timeout. That is reasonable for a client whose only goal is to get an answer; it
        # is wrong here. Seedmesh's reputation layer learns from outcomes, and a call that
        # retries forever never *returns* an outcome -- a dead peer would be recorded as
        # neither success nor failure, so routing would keep choosing it. Bounded retries
        # are what turn churn into a TIMEOUT observation the scorer can act on.
        self.config.request_timeout = request_timeout
        self.config.max_retries = max_retries
        self.config.min_backoff = 1
        self.config.max_backoff = 5
        self._n_blocks = self.config.num_hidden_layers
        self.dht_prefix = self.config.dht_prefix
        self.block_uids = [f"{self.dht_prefix}.{i}" for i in range(self._n_blocks)]

        self.sequence_manager = RemoteSequenceManager(self.config, self.block_uids, dht=self.dht)

        # Petals reports peers as libp2p PeerIDs with no first-seen time, so arrival is
        # tracked locally. The diversity rules need *some* notion of "appeared together",
        # and a client's own first sighting is the honest available approximation.
        self._first_seen: dict[PeerId, float] = {}
        self._addresses: dict[PeerId, str] = {}

    # ---- peer addresses --------------------------------------------------------

    @staticmethod
    def _address_rank(address: str) -> tuple[int, str]:
        """Order candidate addresses: public first, then private, loopback last.

        A peer usually advertises several multiaddrs. The one that identifies its
        *operator's network* -- which is what the anti-sybil clustering prices -- is the
        most routable one; a loopback address says nothing about who runs it.
        """
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return (3, address)
        if parsed.is_loopback:
            return (2, address)
        # `is_global` rather than `not is_private`: the two disagree on ranges that matter
        # here. RFC 5737 documentation space (203.0.113.0/24), CGNAT and reserved blocks are
        # all reported as private-ish by `is_private` but are equally useless for
        # identifying an operator's network. `is_global` is the predicate that actually
        # means "routable on the public internet".
        return (0, address) if parsed.is_global else (1, address)

    def _resolve_addresses(self) -> dict[PeerId, str]:
        """Map peer id -> IP, via the p2p daemon's view of connected peers.

        Petals' DHT records carry no address -- peers are libp2p PeerIDs. Without this,
        every peer falls into the shared "unknown" network cluster and the verification
        sampler refuses to pair *any* of them, because it cannot show two peers are
        independent operators. Resolving addresses is what makes verification usable on a
        real swarm.

        Limitation worth knowing: `list_peers` reports peers the daemon is currently
        connected to. A peer that has announced blocks but has not been dialled yet will
        have no address, and will correctly remain unpairable until it does.
        """
        from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker

        try:
            p2p = RemoteExpertWorker.run_coroutine(self.dht.replicate_p2p())
            peers = RemoteExpertWorker.run_coroutine(p2p.list_peers())
        except Exception:
            return dict(self._addresses)

        for info in peers:
            candidates = []
            for maddr in info.addrs:
                for protocol in ("ip4", "ip6"):
                    try:
                        candidates.append(maddr.value_for_protocol(protocol))
                    except Exception:
                        continue
            if candidates:
                candidates.sort(key=self._address_rank)
                self._addresses[str(info.peer_id)] = candidates[0]

        return dict(self._addresses)

    # ---- InferenceBackend ------------------------------------------------------

    def n_blocks(self, model_id: str) -> int:
        if model_id != self.model_name:
            raise KeyError(f"this backend serves {self.model_name!r}, not {model_id!r}")
        return self._n_blocks

    def discover(self, model_id: str) -> list[ServerInfo]:
        """Who is hosting which blocks, as Seedmesh server records."""
        from petals.data_structures import ServerState
        from petals.utils.dht import get_remote_module_infos

        if model_id != self.model_name:
            raise KeyError(f"this backend serves {self.model_name!r}, not {model_id!r}")

        infos = get_remote_module_infos(self.dht, self.block_uids, latest=True)
        addresses = self._resolve_addresses()
        now = time.time()

        # Petals announces one record per block; collapse them into the contiguous range
        # each peer actually serves, which is what routing and reputation reason about.
        spans: dict[PeerId, dict] = {}
        for block_index, info in enumerate(infos):
            for peer_id, server in (info.servers or {}).items():
                if server.state != ServerState.ONLINE:
                    continue
                key = str(peer_id)
                entry = spans.setdefault(
                    key, {"start": block_index, "end": block_index + 1, "server": server}
                )
                entry["start"] = min(entry["start"], block_index)
                entry["end"] = max(entry["end"], block_index + 1)

        results = []
        for peer_id, entry in spans.items():
            self._first_seen.setdefault(peer_id, now)
            server = entry["server"]
            results.append(
                ServerInfo(
                    peer_id=peer_id,
                    block_range=BlockRange(model_id, entry["start"], entry["end"]),
                    first_seen=self._first_seen[peer_id],
                    # Resolved from the p2p daemon, since Petals' DHT records carry only a
                    # PeerID. Still None for a peer that has announced but not yet been
                    # dialled -- such a peer stays in the "unknown" cluster and correctly
                    # remains unpairable for verification.
                    address=addresses.get(peer_id),
                    # ASN is deliberately left for the ClusterIndex's resolver: it is a
                    # local lookup (an offline GeoIP/Team Cymru table), never something a
                    # peer gets to assert about itself.
                    asn=None,
                    advertised_throughput=float(getattr(server, "throughput", 0.0) or 0.0),
                    # All three fields come from the server's own announcement. A server
                    # that lies about its profile is not a threat here -- the declared
                    # profile selects the tolerance for its *own* comparisons, so
                    # misdeclaring is self-punishing (see docs/threat-model.md, T8).
                    #
                    # `attn_impl` is absent on an unpatched server; "unknown" keeps such a
                    # peer in its own profile bucket rather than silently assuming it
                    # matches ours, so an uncalibrated pair refuses to verify instead of
                    # being judged against the wrong threshold.
                    profile=ComputeProfile(
                        quant=str(getattr(server, "quant_type", None) or "none").lower(),
                        dtype=str(getattr(server, "torch_dtype", None) or "float32").replace(
                            "torch.", ""
                        ),
                        attn=str(getattr(server, "attn_impl", None) or "unknown"),
                    ),
                )
            )
        return results

    def run_segment(
        self,
        server: ServerInfo,
        block_range: BlockRange,
        hidden: np.ndarray,
        *,
        nonce: bytes,
        step: int = 0,
    ) -> SegmentResult:
        """Run ``block_range`` on exactly ``server`` and fingerprint the result.

        Pinned to one server on purpose. Petals' own client would pick whichever route it
        liked; redundant verification requires sending identical input to a *named* peer.
        """
        torch = self._torch
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
            inputs = inputs.unsqueeze(0)  # (tokens, dim) -> (1, tokens, dim)

        span_uids = CHAIN_DELIMITER.join(self.block_uids[block_range.start : block_range.end])
        started = time.perf_counter()

        try:
            peer = PeerID.from_base58(server.peer_id)
            stub = TransformerConnectionHandler.get_stub(self.sequence_manager.state.p2p, peer)
            flat_tensors, args_structure = pack_args_kwargs(inputs, DUMMY)
            metadata = self.sequence_manager.get_request_metadata(
                "rpc_forward", args_structure, span_uids, *flat_tensors
            )

            (outputs,) = RemoteExpertWorker.run_coroutine(
                run_remote_forward(
                    span_uids,
                    stub,
                    self.sequence_manager.rpc_info,
                    *flat_tensors,
                    config=self.sequence_manager.config,
                    metadata=MSGPackSerializer.dumps(metadata),
                )
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            # A timeout is churn, not fraud -- the scorer treats the two very differently,
            # so the distinction has to survive the adapter.
            outcome = (
                Outcome.TIMEOUT
                if isinstance(exc, (TimeoutError, __import__("asyncio").TimeoutError))
                else Outcome.ERROR
            )
            return SegmentResult(outcome, None, None, latency_ms, f"{type(exc).__name__}: {exc}")

        latency_ms = (time.perf_counter() - started) * 1000.0
        array = outputs.detach().cpu().float().numpy()
        sketch = self._sketch(array, block_range, nonce, step)
        return SegmentResult(Outcome.OK, array, sketch, latency_ms)

    # ---- helpers ---------------------------------------------------------------

    def _sketch(self, array: np.ndarray, block_range: BlockRange, nonce: bytes, step: int) -> Sketch:
        flat = array[0] if array.ndim == 3 else array
        seed = derive_seed(nonce, block_range.key(), step)
        return sketch_hidden_state(flat, seed=seed, spec=self.sketch_spec)

    def close(self) -> None:
        try:
            self.sequence_manager.shutdown()
        except Exception:
            pass
        self.dht.shutdown()

    def __enter__(self) -> "PetalsBackend":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
