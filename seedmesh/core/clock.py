"""Time as an injected dependency.

Every decay, TTL and expiry check in Seedmesh reads the clock through this interface.
That is not ceremony: reputation is a function of elapsed time, so tests that cannot
control time can only assert on trivial properties. ``ManualClock`` lets the test suite
advance a half-life and assert the exact decay, and lets the simulator run a simulated
week in milliseconds.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Source of wall-clock time, in seconds since the Unix epoch."""

    def now(self) -> float:  # pragma: no cover - protocol definition
        ...


class SystemClock:
    """Real time. The default outside of tests."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()

    def __repr__(self) -> str:
        return "SystemClock()"


class ManualClock:
    """Explicitly advanced time, for deterministic tests and simulation."""

    __slots__ = ("_now",)

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        """Move time forward. Time never moves backwards; that would corrupt decay state."""
        if seconds < 0:
            raise ValueError(f"cannot advance clock by a negative amount: {seconds}")
        self._now += float(seconds)
        return self._now

    def set(self, when: float) -> float:
        if when < self._now:
            raise ValueError(
                f"refusing to move clock backwards ({self._now} -> {when}); "
                "decay state assumes monotonic time"
            )
        self._now = float(when)
        return self._now

    def __repr__(self) -> str:
        return f"ManualClock(now={self._now})"
