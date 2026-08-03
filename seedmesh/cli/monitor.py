"""`seedmesh monitor` -- who is donating what, and is the model actually covered?

A volunteer who has just typed `seedmesh serve` has no way to see that it worked, and nobody
running a swarm can answer "is anyone hosting blocks 20-24?" without reading logs. This is
the read-only view of both.

**Coverage is the number that matters, and it is not "how many servers".** A model is only
usable if *every* block has at least one online host: twelve servers all hosting blocks 0-3
serve nothing. So the report leads with the least-covered block, not with a peer count.

**Two independent sources, deliberately kept apart.**

*What the DHT says* -- peers, block ranges, state, throughput -- is self-reported. A server
announces its own throughput and its own name; nothing here has been checked. It is exactly
as trustworthy as the peer that wrote it, which is to say not at all.

*What observers report* comes from signed reputation records published by clients that
actually sent requests (`seedmesh/reputation/gossip.py`). Every record is signature-verified
before it counts, and the aggregator caps how much weight any one network cluster can carry.

Presenting these as one blended number would quietly launder a self-claim into a measurement.
They are separate columns, and the report says which is which. A server with a high
self-reported throughput and no observations is a server nobody has tested.

**The monitor publishes nothing.** It is a reader: it takes an ephemeral identity, never
calls `publish()`, and adds no entry to the observer directory. Running a dashboard should
not make you an author of reputation you did not measure.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

UNKNOWN = "-"

DTYPE_ABBREVIATIONS = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


@dataclass
class ServerRow:
    """One peer's contribution, as announced. Nothing here is verified."""

    peer_id: str
    start: int
    end: int
    state: str = "ONLINE"
    throughput: Optional[float] = None
    public_name: Optional[str] = None
    version: Optional[str] = None

    # The compute profile. Shown because differing quantisation, dtype or attention kernel
    # is the single most common reason two honest servers produce slightly different
    # numbers -- and mistaking that for fraud is the defect this project has had to undo
    # more than once. A monitor that hides it invites the same mistake.
    quant: Optional[str] = None
    dtype: Optional[str] = None
    attn: Optional[str] = None
    relayed: Optional[bool] = None

    # Filled in from signed reputation records, when any exist for this peer.
    reliability: Optional[float] = None
    observers: int = 0
    clusters: int = 0

    @property
    def blocks(self) -> int:
        return self.end - self.start

    @property
    def short_id(self) -> str:
        return f"...{self.peer_id[-8:]}" if len(self.peer_id) > 8 else self.peer_id

    @property
    def label(self) -> str:
        return self.public_name or self.short_id

    @property
    def profile(self) -> str:
        """Compact, because it shares a terminal line with four other columns.

        dtype names are abbreviated rather than truncated: cutting `nf4/bfloat16/eager
        (relayed)` to fit produced `nf4/bfloat16/eager (`, which reads as a formatting bug
        and silently drops the one part an operator most needs to see.
        """
        dtype = DTYPE_ABBREVIATIONS.get(self.dtype or "", self.dtype)
        parts = [p for p in (self.quant, dtype, self.attn) if p]
        text = "/".join(parts) if parts else UNKNOWN
        return f"{text}+relay" if self.relayed else text


@dataclass
class SwarmReport:
    model: str
    n_blocks: int
    servers: List[ServerRow] = field(default_factory=list)
    replicas: List[int] = field(default_factory=list)
    """Online hosts per block, index 0..n_blocks-1."""

    observer_count: int = 0
    records_accepted: int = 0
    generated_at: float = 0.0

    @property
    def covered(self) -> bool:
        return bool(self.replicas) and min(self.replicas) > 0

    @property
    def min_replicas(self) -> int:
        return min(self.replicas) if self.replicas else 0

    @property
    def missing_blocks(self) -> List[int]:
        return [i for i, count in enumerate(self.replicas) if count == 0]

    @property
    def total_blocks_donated(self) -> int:
        return sum(row.blocks for row in self.servers)


def collapse_spans(per_block_servers: List[Dict[str, Any]]) -> List[ServerRow]:
    """Turn Petals' per-block announcements into one row per peer.

    Petals writes a DHT record per block, so a peer hosting blocks 0-11 appears twelve
    times. Routing and reputation both reason about the contiguous range, so collapse first.

    ``per_block_servers[i]`` maps ``peer_id -> {state, throughput, public_name, version}``
    for block ``i``. Taking a plain structure rather than Petals' `RemoteModuleInfo` keeps
    this testable on a machine with no backend installed.
    """
    spans: Dict[str, ServerRow] = {}
    for block_index, servers in enumerate(per_block_servers):
        for peer_id, info in (servers or {}).items():
            peer_id = str(peer_id)
            row = spans.get(peer_id)
            if row is None:
                spans[peer_id] = ServerRow(
                    peer_id=peer_id,
                    start=block_index,
                    end=block_index + 1,
                    state=str(info.get("state", "ONLINE")),
                    throughput=info.get("throughput"),
                    public_name=info.get("public_name"),
                    version=info.get("version"),
                    quant=info.get("quant_type"),
                    dtype=info.get("torch_dtype"),
                    attn=info.get("attn_impl"),
                    relayed=info.get("using_relay"),
                )
            else:
                row.start = min(row.start, block_index)
                row.end = max(row.end, block_index + 1)
    return sorted(spans.values(), key=lambda r: (r.start, r.peer_id))


def count_replicas(rows: List[ServerRow], n_blocks: int) -> List[int]:
    """Online hosts per block. Only ONLINE peers count -- a joining server serves nothing."""
    replicas = [0] * n_blocks
    for row in rows:
        if row.state != "ONLINE":
            continue
        for block in range(max(0, row.start), min(n_blocks, row.end)):
            replicas[block] += 1
    return replicas


def build_report(
    model: str,
    n_blocks: int,
    per_block_servers: List[Dict[str, Any]],
    *,
    aggregates: Optional[Dict[str, Any]] = None,
    observer_count: int = 0,
    records_accepted: int = 0,
    now: Optional[float] = None,
) -> SwarmReport:
    rows = collapse_spans(per_block_servers)
    for row in rows:
        score = (aggregates or {}).get(row.peer_id)
        if score is not None:
            row.reliability = score.reliability
            row.observers = score.observer_count
            row.clusters = score.distinct_clusters
    return SwarmReport(
        model=model,
        n_blocks=n_blocks,
        servers=rows,
        replicas=count_replicas(rows, n_blocks),
        observer_count=observer_count,
        records_accepted=records_accepted,
        generated_at=now if now is not None else time.time(),
    )


# ---- rendering ---------------------------------------------------------------


def fit(text: str, width: int) -> str:
    """Pad or truncate to exactly `width`, marking any truncation.

    A silently cut field is worse than a narrower one: it reads as real content.
    """
    if len(text) <= width:
        return text.ljust(width)
    return text[: max(0, width - 1)] + "~"


def coverage_bar(replicas: List[int], width: int = 48) -> str:
    """One character per block where it fits, otherwise the worst block per bucket.

    Bucketing by minimum rather than average on purpose: a gap next to a well-covered block
    must not average away into looking fine.
    """
    if not replicas:
        return ""
    if len(replicas) <= width:
        buckets = [[count] for count in replicas]
    else:
        size = (len(replicas) + width - 1) // width
        buckets = [replicas[i:i + size] for i in range(0, len(replicas), size)]
    symbols = {0: "_", 1: "1", 2: "2", 3: "3"}
    return "".join(symbols.get(min(b), "*") for b in buckets)


def render_text(report: SwarmReport) -> List[str]:
    lines = [
        f"swarm: {report.model}",
        f"  {len(report.servers)} server(s) donating {report.total_blocks_donated} "
        f"block-slot(s) across {report.n_blocks} blocks",
        "",
    ]

    if not report.servers:
        lines.append("  nobody is hosting this model right now.")
        lines.append("  If you just started a server, its block records take a minute to")
        lines.append("  reach the DHT. See docs/QUICKSTART.md.")
        return lines

    lines.append(f"  coverage  [{coverage_bar(report.replicas)}]")
    if report.covered:
        lines.append(
            f"            every block has at least {report.min_replicas} host(s) -- "
            "the model is usable"
        )
    else:
        missing = report.missing_blocks
        shown = ", ".join(str(b) for b in missing[:12])
        more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        lines.append(f"            NOT USABLE: {len(missing)} block(s) have no host: {shown}{more}")

        # Name the reason when servers exist but none of them counts yet. Without this the
        # report contradicts itself -- "1 server donating 23 block-slots" directly above
        # "32 blocks have no host" -- and the natural reading is that the tool is broken
        # rather than that the server is still loading. Hit for real while bringing up 8B.
        joining = [row for row in report.servers if row.state == "JOINING"]
        if joining and not any(row.state == "ONLINE" for row in report.servers):
            covered_when_ready = sum(row.end - row.start for row in joining)
            lines.append(
                f"            {len(joining)} server(s) are still JOINING -- loading blocks, "
                "not serving yet."
            )
            lines.append(
                f"            Nothing is wrong; wait for them. They will cover "
                f"{covered_when_ready} block-slot(s)."
            )
            if covered_when_ready < report.n_blocks:
                lines.append(
                    f"            That is still short of {report.n_blocks}; "
                    "you need more volunteers."
                )
        else:
            lines.append("            A model needs every block covered; more servers on the")
            lines.append("            same blocks does not help.")
    lines.append("")

    lines.append("  server                blocks  profile                self-rep.  observed")
    lines.append("  " + "-" * 79)
    for row in report.servers:
        throughput = f"{row.throughput:.0f} tok/s" if row.throughput else UNKNOWN
        if row.observers:
            observed = f"{row.reliability:.3f} ({row.observers} obs, {row.clusters} clust)"
        else:
            observed = "not yet measured"
        state = "" if row.state == "ONLINE" else f" [{row.state}]"
        lines.append(
            f"  {fit(row.label, 19)}  {row.start:>3}-{row.end:<3}  {fit(row.profile, 21)}  "
            f"{throughput:>10s}  {observed}{state}"
        )

    lines.append("")
    lines.append(
        f"  reputation: {report.observer_count} observer(s) advertised, "
        f"{report.records_accepted} signed record(s) accepted"
    )
    # Said every time, because the two columns look equally authoritative and are not.
    lines.append(
        "  'self-reported' is whatever the server claims and is unverified; 'observed by"
    )
    lines.append(
        "  others' comes from signature-checked records published by clients that used it."
    )
    return lines


def render_html(report: SwarmReport) -> str:
    """A self-contained page -- no external CSS, fonts or scripts, so it can be served
    from anywhere or opened straight off disk."""
    import html as html_mod

    when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(report.generated_at))
    if report.covered:
        banner_class, banner = "ok", (
            f"Model is usable &mdash; every block has at least {report.min_replicas} host(s)"
        )
    elif report.servers:
        banner_class, banner = "bad", (
            f"Not usable &mdash; {len(report.missing_blocks)} of {report.n_blocks} "
            "blocks have no host"
        )
    else:
        banner_class, banner = "bad", "Nobody is hosting this model right now"

    rows = []
    for row in report.servers:
        throughput = f"{row.throughput:.1f} tok/s" if row.throughput else UNKNOWN
        observed = (
            f"{row.reliability:.3f} <span class='dim'>from {row.observers} observer(s), "
            f"{row.clusters} cluster(s)</span>"
            if row.observers
            else "<span class='dim'>not yet measured</span>"
        )
        rows.append(
            "<tr>"
            f"<td>{html_mod.escape(row.label)}</td>"
            f"<td class='mono dim'>{html_mod.escape(row.short_id)}</td>"
            f"<td>{row.start}&ndash;{row.end} <span class='dim'>({row.blocks})</span></td>"
            f"<td class='mono dim'>{html_mod.escape(row.profile)}</td>"
            f"<td>{html_mod.escape(throughput)}</td>"
            f"<td>{observed}</td>"
            "</tr>"
        )

    cells = "".join(
        f"<i class='b{min(count, 3)}' title='block {index}: {count} host(s)'></i>"
        for index, count in enumerate(report.replicas)
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seedmesh &middot; {html_mod.escape(report.model)}</title>
<style>
 :root {{ color-scheme: light dark; --fg:#111; --dim:#666; --line:#ddd; --bg:#fff; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e8e8e8; --dim:#999; --line:#333; --bg:#141414; }} }}
 body {{ font:15px/1.5 system-ui,sans-serif; color:var(--fg); background:var(--bg);
        margin:0; padding:2rem 1rem; }}
 main {{ max-width:60rem; margin:0 auto; }}
 h1 {{ font-size:1.3rem; margin:0 0 .25rem; }}
 .sub {{ color:var(--dim); margin:0 0 1.5rem; }}
 .banner {{ padding:.75rem 1rem; border-radius:6px; font-weight:600; margin-bottom:1.5rem; }}
 .ok {{ background:#0a05; }} .bad {{ background:#f005; }}
 .cov {{ display:flex; gap:2px; flex-wrap:wrap; margin-bottom:.5rem; }}
 .cov i {{ width:14px; height:26px; border-radius:2px; background:#f00; }}
 .b1 {{ background:#7a3 !important; }} .b2 {{ background:#4a2 !important; }}
 .b3 {{ background:#182 !important; }}
 table {{ border-collapse:collapse; width:100%; margin-top:1rem; }}
 th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line); }}
 th {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:var(--dim); }}
 .mono {{ font-family:ui-monospace,monospace; font-size:.85em; }}
 .dim {{ color:var(--dim); }}
 .note {{ color:var(--dim); font-size:.85rem; margin-top:1.5rem;
          border-top:1px solid var(--line); padding-top:1rem; }}
 .overflow {{ overflow-x:auto; }}
</style></head><body><main>
<h1>{html_mod.escape(report.model)}</h1>
<p class="sub">{report.n_blocks} blocks &middot; {len(report.servers)} volunteer(s) &middot;
 generated {when}</p>
<div class="banner {banner_class}">{banner}</div>
<h2 style="font-size:1rem">Block coverage</h2>
<div class="overflow"><div class="cov">{cells}</div></div>
<p class="dim" style="font-size:.85rem">One cell per block. Red means no host, and a single
 red cell makes the whole model unusable &mdash; extra servers on blocks that are already
 covered do not help.</p>
<div class="overflow"><table>
<thead><tr><th>Volunteer</th><th>Peer</th><th>Blocks</th><th>Compute profile</th>
<th>Self-reported</th><th>Observed by others</th></tr></thead>
<tbody>{"".join(rows) or "<tr><td colspan='6' class='dim'>No servers online.</td></tr>"}</tbody>
</table></div>
<p class="note"><strong>Self-reported</strong> is what the server says about itself. Nothing
 has checked it. <strong>Observed by others</strong> comes from signed reputation records
 published by clients that actually sent it requests; each is signature-verified, and no
 single network cluster can dominate the result. A volunteer with a high self-reported
 throughput and no observations is one nobody has tested yet.
 {report.observer_count} observer(s) advertised, {report.records_accepted} record(s)
 accepted.</p>
<p class="note"><strong>Compute profile</strong> is quantisation / dtype / attention kernel.
 Volunteers differ here, and that is normal &mdash; a mismatch makes two honest servers
 produce slightly different numbers, which is why verification compares within a profile
 rather than treating the difference as a fault.</p>
</main></body></html>
"""


def render_json(report: SwarmReport) -> str:
    payload = asdict(report)
    payload["covered"] = report.covered
    payload["min_replicas"] = report.min_replicas
    payload["missing_blocks"] = report.missing_blocks
    return json.dumps(payload, indent=2)


# ---- live collection ---------------------------------------------------------


def dht_prefix_for(model: str, model_type: str) -> str:
    """Petals' DHT namespace for a model, derived without importing Petals.

    Reading the swarm needs exactly two facts -- the block count and this prefix -- and both
    live in `config.json`, which `hardware.fetch_config` pulls over plain HTTP.

    Measured peak RSS for one `collect()` against the live swarm: **390 MiB with Petals
    imported, 279 MiB without**. Less of a saving than it looks like it should be, because
    hivemind depends on torch, so torch is resident either way -- what Petals adds on top is
    transformers and its model registry. 111 MiB still matters on the $5/1 GiB VPS that is
    the natural home for a public dashboard and is already running a bootstrap peer.

    The bigger reason is reach, not memory: this path needs no `seedmesh setup` at all, so a
    box provisioned only as a bootstrap peer can serve the dashboard, and `seedmesh monitor`
    works on Windows where the backend refuses to run.

    Reimplemented rather than imported, and it is genuinely per-architecture -- llama and
    deepseek_v3 append `-hf` and keep only the repo name, falcon keeps the repo name without
    the suffix, bloom appends `-petals` to the FULL path, and mixtral keeps the full path
    with no suffix. Getting one wrong reads an empty namespace and reports a healthy swarm
    as empty, so `test_monitor.py` checks every rule against the installed Petals when it is
    present, and `collect` prefers Petals' own answer whenever it can import it.
    """
    if model_type == "bloom":
        return f"{model}-petals".replace(".", "-")
    if model_type == "mixtral":
        return model.replace(".", "-")

    prefix = model.split("/")[-1].replace(".", "-")
    if model_type in ("llama", "deepseek_v3", "kimi_k2") and not prefix.endswith("-hf"):
        prefix += "-hf"
    return prefix


def swarm_shape(model: str) -> tuple[int, str]:
    """(block count, DHT prefix), preferring Petals' own answer when it is installed.

    Petals is authoritative: if it is importable, its derivation is the one the servers
    actually used. The config-only path exists for the machine that has no backend at all.
    """
    try:
        from petals.utils.auto_config import AutoDistributedConfig

        config = AutoDistributedConfig.from_pretrained(model)
        return config.num_hidden_layers, config.dht_prefix
    except Exception:
        pass

    from seedmesh.cli.hardware import fetch_config

    config = fetch_config(model)
    return config["num_hidden_layers"], dht_prefix_for(model, config.get("model_type", ""))


# Petals stores one DHT record per block: key = "<prefix>.<index>", subkeys = peer ids,
# each value the tuple `ServerInfo.to_tuple()` produces -- (state value, throughput, extras).
SERVER_STATES = {0: "OFFLINE", 1: "JOINING", 2: "ONLINE"}


def announcement_from_tuple(raw: Any) -> Dict[str, Any]:
    """Decode one server's announcement without importing Petals.

    Mirrors `ServerInfo.from_tuple`: everything past the first two positions is a plain dict
    of extras, so new fields upstream arrive here automatically rather than breaking the read.
    """
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        return {"state": "OFFLINE"}
    state, throughput = raw[0], raw[1]
    extras = raw[2] if len(raw) > 2 and isinstance(raw[2], dict) else {}
    return {
        "state": SERVER_STATES.get(state, str(state)),
        "throughput": throughput,
        **{key: value for key, value in extras.items() if key not in ("state", "throughput")},
    }


def _read_blocks(dht, block_uids: List[str]) -> List[Dict[str, Any]]:
    """Per-block ``{peer id: announcement}``.

    Prefers Petals' own reader when it is installed, because it is authoritative. Falls back
    to plain DHT reads so the monitor runs anywhere hivemind does -- including a box
    provisioned only as a bootstrap peer, and Windows, where the backend refuses to run.

    Verified against the live swarm by blocking the `petals` import and comparing the two
    reports field by field: identical (`spike/` A/B, 2026-08-02). A wrong rule here would not
    raise -- it would read an empty namespace and report a healthy swarm as dead -- so the
    check is an equality test against the real thing, not a shape assertion.
    """
    try:
        from petals.utils.dht import get_remote_module_infos

        infos = get_remote_module_infos(dht, block_uids, latest=True)
        return [
            {str(peer): _server_dict(server) for peer, server in (info.servers or {}).items()}
            for info in infos
        ]
    except ImportError:
        pass

    from seedmesh.reputation.dht import dht_fetch

    found = dht_fetch(dht, list(block_uids))
    per_block = []
    for uid in block_uids:
        entry = found.get(uid)
        servers: Dict[str, Any] = {}
        if isinstance(entry, dict):
            for peer, raw in entry.items():
                servers[str(peer)] = announcement_from_tuple(raw)
        per_block.append(servers)
    return per_block


def _server_dict(server: Any) -> Dict[str, Any]:
    """Flatten Petals' ServerInfo into the plain mapping the pure functions take."""
    state = getattr(server, "state", None)
    return {
        "state": getattr(state, "name", str(state)),
        "throughput": getattr(server, "throughput", None),
        "public_name": getattr(server, "public_name", None),
        "version": getattr(server, "version", None),
        "quant_type": getattr(server, "quant_type", None),
        "torch_dtype": getattr(server, "torch_dtype", None),
        "attn_impl": getattr(server, "attn_impl", None),
        "using_relay": getattr(server, "using_relay", None),
    }


def collect(dht, model: str, prefix: str = "seedmesh") -> SwarmReport:
    """Read the swarm's announcements and everything observers have signed about it."""
    n_blocks, prefix_for_blocks = swarm_shape(model)
    block_uids = [f"{prefix_for_blocks}.{i}" for i in range(n_blocks)]
    per_block = _read_blocks(dht, block_uids)

    aggregates, observers, accepted = _read_reputation(dht, prefix)
    return build_report(
        model,
        n_blocks,
        per_block,
        aggregates=aggregates,
        observer_count=observers,
        records_accepted=accepted,
    )


def _read_reputation(dht, prefix: str = "seedmesh"):
    """Fetch and verify what observers have published. Never publishes anything.

    Built with a throwaway identity precisely because it must not participate: a monitor
    that announced itself in the observer directory would send every reader a lookup for
    records it never writes.
    """
    from seedmesh.core.identity import Identity
    from seedmesh.reputation.aggregate import AggregationConfig, ReputationAggregator
    from seedmesh.reputation.diversity import UNKNOWN_CLUSTER
    from seedmesh.reputation.dht import transport_for
    from seedmesh.reputation.gossip import ReputationGossip
    from seedmesh.reputation.scorer import ReputationScorer

    publish, fetch = transport_for(dht)
    aggregator = ReputationAggregator(lambda _peer: UNKNOWN_CLUSTER, AggregationConfig())
    gossip = ReputationGossip(
        Identity.generate(),
        ReputationScorer(),
        aggregator,
        publish_fn=publish,
        fetch_fn=fetch,
        transport_id=str(dht.peer_id),
        prefix=prefix,
    )
    try:
        observers = gossip.known_observers()
        accepted = gossip.fetch(observers)
        return aggregator.aggregate_all(), len(observers), accepted
    except Exception:
        # A swarm with no reputation records yet is the common case, not an error.
        return {}, 0, 0


def cmd_monitor(args) -> int:
    from seedmesh.cli.swarm import load_swarm, resolve_peers

    # Deliberately NOT gated on the Petals backend. Monitoring reads DHT keys; it needs
    # hivemind and nothing else. Requiring the backend would mean a public dashboard could
    # only run on a machine provisioned to serve, which is the opposite of the point.
    try:
        from hivemind.dht import DHT
    except ImportError:
        print("monitoring needs hivemind: pip install hivemind==1.1.12")
        return 2

    peers, note = resolve_peers(args.initial_peers, args.swarm_file)
    if note and not args.json:
        print(note)
    if not peers:
        print("no bootstrap peers; pass --initial-peers or --swarm-file")
        return 2

    prefix = "seedmesh"
    try:
        swarm = load_swarm(args.swarm_file)
        if swarm is not None and swarm.name:
            # Must match what `seedmesh chat` publishes under, or the monitor reads an empty
            # namespace and reports "0 observers" on a swarm that has plenty.
            prefix = f"seedmesh/{swarm.name}"
    except Exception:
        pass

    dht = DHT(initial_peers=peers, client_mode=True, start=True)
    try:
        while True:
            report = collect(dht, args.model, prefix=prefix)
            if args.html:
                from pathlib import Path

                Path(args.html).write_text(render_html(report), encoding="utf-8")
                print(f"wrote {args.html}")
            elif args.json:
                print(render_json(report))
            else:
                print()
                for line in render_text(report):
                    print(line)
            if not args.watch:
                return 0 if report.covered else 1
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    finally:
        dht.shutdown()

