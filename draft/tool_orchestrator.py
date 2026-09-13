"""Register tools, resolve capabilities, enforce scopes, and execute batches concurrently.

Re-registering a name replaces it; highest priority wins and alphabetical name
breaks ties; unknown capabilities raise KeyError; missing scopes raise
PermissionDenied before execution; parallel tasks use real thread concurrency,
preserve input order, and isolate failures as {"ok": False, "error": ...}.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set


class PermissionDenied(Exception):
    """Raised when a required scope is missing."""

    def __init__(self, tool_name: str, missing_scopes: Set[str]):
        self.tool_name = tool_name
        self.missing_scopes = missing_scopes
        super().__init__(
            f"Permission denied for tool '{tool_name}': missing scopes {sorted(missing_scopes)}"
        )


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    capabilities: Set[str] = field(default_factory=set)
    scopes: Set[str] = field(default_factory=set)
    priority: int = 0


class ToolOrchestrator:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        capabilities: Optional[Sequence[str]] = None,
        scopes: Optional[Sequence[str]] = None,
        priority: int = 0,
    ) -> None:
        """Register (or replace) a tool under ``name``."""
        self._tools[name] = ToolSpec(
            name=name,
            fn=fn,
            capabilities=set(capabilities or []),
            scopes=set(scopes or []),
            priority=priority,
        )

    def resolve(self, capability: str) -> ToolSpec:
        """Return the highest-priority tool that advertises ``capability``.

        Ties are broken alphabetically by name.  Raises KeyError if none match.
        """
        candidates = [
            t for t in self._tools.values() if capability in t.capabilities
        ]
        if not candidates:
            raise KeyError(f"No tool registered for capability: {capability}")
        # Highest priority first, then alphabetical name
        candidates.sort(key=lambda t: (-t.priority, t.name))
        return candidates[0]

    def execute(
        self,
        name: str,
        args: Optional[dict] = None,
        *,
        granted_scopes: Optional[Sequence[str]] = None,
    ) -> Any:
        """Execute a single tool after scope check."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        tool = self._tools[name]
        granted = set(granted_scopes or [])
        missing = tool.scopes - granted
        if missing:
            raise PermissionDenied(name, missing)
        return tool.fn(**(args or {}))

    def execute_batch(
        self,
        tasks: Sequence[dict],
        *,
        granted_scopes: Optional[Sequence[str]] = None,
        max_workers: int = 4,
    ) -> List[dict]:
        """Execute multiple tool calls concurrently.

        Each task is a dict with at least ``name`` and optional ``args``.
        Results preserve input order.  Failures are isolated as
        ``{"ok": False, "error": ...}``.
        """
        results: List[Optional[dict]] = [None] * len(tasks)

        def _run(idx: int, task: dict) -> None:
            name = task.get("name") or task.get("tool")
            args = task.get("args") or task.get("arguments") or {}
            try:
                value = self.execute(name, args, granted_scopes=granted_scopes)
                results[idx] = {"ok": True, "result": value}
            except Exception as exc:
                results[idx] = {"ok": False, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run, i, t): i for i, t in enumerate(tasks)
            }
            for fut in as_completed(futures):
                fut.result()  # surface any unexpected exception from the worker itself

        return [r if r is not None else {"ok": False, "error": "missing"} for r in results]
