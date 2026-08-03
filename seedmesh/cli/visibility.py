"""Telling a volunteer whether the swarm can actually see them.

`seedmesh serve` printed `Started` and nothing else. That line means the local process came
up; it says nothing about whether any record reached the DHT, and those are different
questions. A volunteer whose announcement never lands looks — from their own terminal —
exactly like one who is serving perfectly.

That gap cost three round-trips through a third person: one machine loaded all 32 blocks,
logged `Started`, and never appeared to anyone else. Nobody could tell from either end
whether the server had stopped, was still loading, or was announcing into a void.

**How this identifies "you" without knowing your peer id.** Petals generates a fresh peer id
per run unless `--identity_path` is set, and the id is only discoverable by scraping the
server's stdout — which would mean piping it, which would kill the download progress bars
that are the one piece of feedback a first run does have. So instead: snapshot the peers
already in the swarm *before* launching, then watch for one that was not there before. A new
peer covering the range we asked for is us.

The failure mode of that inference is a *different* volunteer joining in the same window,
which would produce a false "you are visible". That is the safe direction — it can only
report success early, never invent a failure — and it is stated in the output rather than
hidden, because a swarm this small makes the coincidence unlikely but not impossible.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Set


def snapshot_peers(dht: Any, model: str, prefix: str = "seedmesh") -> Optional[Set[str]]:
    """Peer ids currently announcing blocks for this model, or None if it cannot be read.

    None and empty-set mean different things: "the lookup failed" versus "nobody is here".
    Conflating them would let a broken read masquerade as an empty swarm.
    """
    try:
        from seedmesh.cli.monitor import collect

        report = collect(dht, model, prefix=prefix)
    except Exception:
        return None
    try:
        return {str(row.peer_id) for row in report.servers}
    except Exception:
        return None


def new_arrivals(before: Optional[Set[str]], after: Optional[Set[str]]) -> Set[str]:
    """Peers present now that were not present at the baseline."""
    if before is None or after is None:
        return set()
    return after - before


def describe(found: Set[str], waited: float, baseline_ok: bool) -> List[str]:
    """The verdict, as lines to print."""
    if not baseline_ok:
        return [
            "",
            "  visibility check skipped -- could not read the swarm before starting,",
            "  so there is no baseline to compare against.",
        ]
    if found:
        peers = ", ".join(f"...{peer[-8:]}" for peer in sorted(found))
        lines = [
            "",
            f"  VISIBLE: the swarm can see a new server ({peers}).",
            "  Your blocks reached the DHT, so other peers can find you.",
        ]
        if len(found) > 1:
            lines.append(
                "  (more than one peer joined while waiting, so one of these may be someone"
                " else)"
            )
        return lines
    return [
        "",
        f"  NOT VISIBLE after {waited:.0f}s: no new server appeared in the swarm.",
        "  Your server may still be loading -- large models take many minutes, and this",
        "  check gives up before they finish. But if it stays this way, nothing you host",
        "  is reachable by anyone.",
        "",
        "  Worth checking, in order:",
        "    seedmesh monitor          -- do you appear in your own swarm view?",
        "    seedmesh doctor           -- can other peers dial this machine at all?",
        "  A server that never announces is usually a DHT write that is not landing,",
        "  not a problem with the blocks themselves.",
    ]


def watch(
    make_dht: Callable[[], Any],
    model: str,
    *,
    baseline: Optional[Set[str]],
    prefix: str = "seedmesh",
    checks: int = 6,
    interval: float = 30.0,
    sleep: Optional[Callable[[float], None]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> Set[str]:
    """Poll until a new peer appears, or the checks run out. Returns what was found.

    Polls rather than waiting once: Petals announces blocks as JOINING *before* it downloads
    them, so the record usually lands within a minute even when loading takes fifteen. A
    single late check would answer the question long after the volunteer stopped wondering.
    """
    import time as time_module

    sleeper = sleep or time_module.sleep
    write = emit or print

    if baseline is None:
        write("\n".join(describe(set(), 0.0, baseline_ok=False)))
        return set()

    dht = None
    try:
        for attempt in range(checks):
            sleeper(interval)
            if dht is None:
                try:
                    dht = make_dht()
                except Exception:
                    continue
            found = new_arrivals(baseline, snapshot_peers(dht, model, prefix=prefix))
            if found:
                write("\n".join(describe(found, (attempt + 1) * interval, baseline_ok=True)))
                return found
        write("\n".join(describe(set(), checks * interval, baseline_ok=True)))
        return set()
    finally:
        if dht is not None:
            try:
                dht.shutdown()
            except Exception:
                pass
