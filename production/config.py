"""Per-agent configuration and run control (stack-independent)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set


@dataclass
class AgentConfig:
    """Tunable budgets for one agent. All timeouts in seconds; None = unlimited."""

    max_iterations: int = 5
    tool_timeout: Optional[float] = None
    overall_timeout: Optional[float] = None
    turn_timeout: Optional[float] = None
    per_tool_timeouts: Dict[str, float] = field(default_factory=dict)
    max_output_chars: Optional[int] = None
    tool_allowlist: Optional[Set[str]] = None
    tool_denylist: Optional[Set[str]] = None
    cost_ceiling: Optional[float] = None


@dataclass
class RunState:
    """Checkpointable run state between turns."""

    run_id: str
    goal: str
    status: str  # running | done | cancelled | timed_out | max_iterations
    iterations: int = 0
    answer: Optional[str] = None
    trace: list = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class CancellationToken:
    """Cooperative cancellation; check between turns/tool calls."""

    def __init__(self) -> None:
        self._cancelled = False
        self.reason: Optional[str] = None

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancelled = True
        self.reason = reason

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError(self.reason or "cancelled")


class CancelledError(Exception):
    pass


class Deadline:
    """Wall-clock deadline using an injectable clock."""

    def __init__(self, seconds: Optional[float], clock: Callable[[], float]):
        self._clock = clock
        self._deadline = None if seconds is None or seconds <= 0 else clock() + seconds

    def remaining(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return self._deadline - self._clock()

    def expired(self) -> bool:
        r = self.remaining()
        return r is not None and r <= 0

    def raise_if_expired(self) -> None:
        if self.expired():
            raise TimeoutError("overall deadline exceeded")
