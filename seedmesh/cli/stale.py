"""Leftover server processes from a previous run.

Stopping `seedmesh serve` does not always stop the server. Petals runs a small tree of
processes, and if the parent dies without cleaning up -- a closed terminal, a `kill -9`, a
crashed shell -- the children survive holding GPU memory and their DHT identity. Starting
again then fails or, worse, half-works, and the remedy people learn is
`pkill -f run_server`, which is a blunt instrument to hand a volunteer.

**What counts as stale is not "another server exists".** Running two servers on purpose is
legitimate and this project does it -- one machine hosted blocks 9-32 while another hosted
0-9. Killing those would be worse than the problem. So the test is *orphanhood*: a Petals
server whose parent is gone (reparented to PID 1) was left behind by a session that is no
longer running. A server with a live parent belongs to somebody and is left alone.

Reads `/proc` directly rather than shelling out to `pgrep`, so the logic is testable without
spawning anything, and so a missing `pgrep` cannot turn cleanup into a crash.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional


@dataclass(frozen=True)
class ServerProcess:
    pid: int
    ppid: int
    cmdline: str

    @property
    def orphaned(self) -> bool:
        """Parent gone. On Linux an orphan is reparented to init (PID 1)."""
        return self.ppid <= 1

    def mentions(self, model: str) -> bool:
        return bool(model) and model in self.cmdline


def _read_proc(pid: int) -> Optional[ServerProcess]:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        ).strip()
    except Exception:
        return None
    if not cmdline:
        return None
    try:
        # /proc/<pid>/stat: field 4 is ppid, but comm (field 2) may contain spaces inside
        # parentheses, so split after the closing paren rather than on whitespace.
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        after = stat[stat.rindex(")") + 1 :].split()
        ppid = int(after[1])
    except Exception:
        return None
    return ServerProcess(pid=pid, ppid=ppid, cmdline=cmdline)


def list_petals_servers(
    reader: Optional[Callable[[int], Optional[ServerProcess]]] = None,
    pids: Optional[Iterable[int]] = None,
) -> List[ServerProcess]:
    """Every running `petals.cli.run_server`, excluding this process and its own children."""
    read = reader or _read_proc
    if pids is None:
        try:
            pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
        except Exception:
            return []
    mine = os.getpid()
    found = []
    for pid in pids:
        if pid == mine:
            continue
        process = read(pid)
        if process is None:
            continue
        if "petals.cli.run_server" in process.cmdline:
            found.append(process)
    return found


def stale_for(model: str, processes: Iterable[ServerProcess]) -> List[ServerProcess]:
    """Orphaned servers for this model -- the ones safe to clear automatically.

    Both conditions matter. Same model, because a server for a different model is a
    different swarm and none of our business. Orphaned, because a live parent means somebody
    is running it deliberately.
    """
    return [p for p in processes if p.orphaned and p.mentions(model)]


def others_for(model: str, processes: Iterable[ServerProcess]) -> List[ServerProcess]:
    """Non-orphaned servers for this model: reported, never killed."""
    return [p for p in processes if not p.orphaned and p.mentions(model)]


def terminate(
    processes: Iterable[ServerProcess],
    *,
    kill: Optional[Callable[[int, int], None]] = None,
    wait: Optional[Callable[[float], None]] = None,
    still_alive: Optional[Callable[[int], bool]] = None,
) -> List[int]:
    """SIGTERM, give them a moment, then SIGKILL what is left. Returns pids signalled.

    Graceful first: a Petals server that gets SIGTERM deregisters from the DHT, so the swarm
    stops advertising blocks it is no longer serving. Going straight to SIGKILL leaves stale
    announcements that make the model look covered when it is not.
    """
    import time as time_module

    send = kill or os.kill
    pause = wait or time_module.sleep
    alive = still_alive if still_alive is not None else _is_alive

    signalled = []
    for process in processes:
        try:
            send(process.pid, signal.SIGTERM)
            signalled.append(process.pid)
        except Exception:
            pass
    if not signalled:
        return []

    pause(3.0)
    # SIGKILL is absent on Windows. serve refuses there anyway, but a module that raises
    # AttributeError on import-and-call is a trap for anyone reusing it.
    hard = getattr(signal, "SIGKILL", signal.SIGTERM)
    for pid in signalled:
        if alive(pid):
            try:
                send(pid, hard)
            except Exception:
                pass
    return signalled


def _is_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def describe(stale: List[ServerProcess], others: List[ServerProcess]) -> List[str]:
    lines = []
    if stale:
        lines.append(
            f"  clearing {len(stale)} leftover server process(es) from a previous run "
            f"({', '.join(str(p.pid) for p in stale)})"
        )
        lines.append("  they were orphaned -- their parent is gone -- and hold GPU memory")
    if others:
        lines.append(
            f"  note: {len(others)} other server(s) for this model are running with a live "
            "parent"
        )
        lines.append("  leaving them alone; stop them yourself if they are not wanted")
    return lines
