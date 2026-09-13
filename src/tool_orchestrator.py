"""Project 3 — Multi-Tool Orchestrator.

Dynamic tool registry, capability-based routing with priority conflict
resolution, permission scoping, and parallel execution.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class Tool:
    name: str
    fn: Callable
    capabilities: set[str]
    required_scope: str | None = None  # None = public
    priority: int = 0  # higher wins when capabilities conflict


class PermissionDenied(Exception):
    pass


class Orchestrator:
    def __init__(self):
        """Set up an empty registry."""
        self._registry: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool. Re-registering the same name replaces it."""
        self._registry[tool.name] = tool

    def resolve(self, capability: str) -> Tool:
        """Return the tool for a capability.

        Requirements:
            - If several tools share the capability, the highest `priority`
              wins; ties break alphabetically by name (deterministic).
            - Unknown capability -> KeyError.
        """
        candidates = [
            t for t in self._registry.values() if capability in t.capabilities
        ]
        if not candidates:
            raise KeyError(f"No tool for capability: {capability}")
        candidates.sort(key=lambda t: (-t.priority, t.name))
        return candidates[0]

    def execute(self, capability: str, scopes: set[str], **kwargs):
        """Resolve and run one tool.

        Requirements:
            - If the tool has a required_scope not present in `scopes`,
              raise PermissionDenied WITHOUT executing the tool.
        """
        tool = self.resolve(capability)
        if tool.required_scope is not None and tool.required_scope not in scopes:
            raise PermissionDenied(
                f"Tool '{tool.name}' requires scope '{tool.required_scope}'"
            )
        return tool.fn(**kwargs)

    def execute_parallel(self, tasks: list[dict], scopes: set[str]) -> list[dict]:
        """Run many tasks concurrently (threads); each task is
        {"capability": str, "kwargs": dict}.

        Requirements:
            - MUST use real concurrency (concurrent.futures) — grading asserts
              wall-clock time of parallel sleeps.
            - Results return IN INPUT ORDER as
              {"ok": True, "result": ...} or {"ok": False, "error": str}.
            - One failing/forbidden task must not affect the others.
        """
        results: List[Optional[dict]] = [None] * len(tasks)

        def _run(idx: int, task: dict) -> None:
            capability = task.get("capability", "")
            kwargs = task.get("kwargs") or {}
            try:
                value = self.execute(capability, scopes, **kwargs)
                results[idx] = {"ok": True, "result": value}
            except Exception as exc:
                results[idx] = {"ok": False, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
            futures = [pool.submit(_run, i, t) for i, t in enumerate(tasks)]
            for f in as_completed(futures):
                f.result()

        return [r if r is not None else {"ok": False, "error": "missing"} for r in results]
