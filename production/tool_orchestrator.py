"""Project 3 — Tool Orchestrator (production, stack-independent)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from errors import ToolTimeoutError, ToolValidationError, classify_tool_exception


@dataclass
class Tool:
    name: str
    fn: Callable
    capabilities: set[str]
    required_scope: str | None = None
    priority: int = 0
    arg_schema: Optional[dict] = None
    timeout: Optional[float] = None
    rate_limit_per_sec: Optional[float] = None


class PermissionDenied(Exception):
    pass


class ToolRuntimeRateLimit(Exception):
    code = "tool_rate_limit"


class _RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self._tokens = rate_per_sec
        self._last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class Orchestrator:
    def __init__(self, default_timeout: Optional[float] = None):
        self._registry: Dict[str, Tool] = {}
        self.default_timeout = default_timeout
        self._limiters: Dict[str, _RateLimiter] = {}

    def register(self, tool: Tool) -> None:
        self._registry[tool.name] = tool
        if tool.rate_limit_per_sec and tool.rate_limit_per_sec > 0:
            self._limiters[tool.name] = _RateLimiter(tool.rate_limit_per_sec)

    def resolve(self, capability: str) -> Tool:
        candidates = [t for t in self._registry.values() if capability in t.capabilities]
        if not candidates:
            raise KeyError(f"No tool for capability: {capability}")
        candidates.sort(key=lambda t: (-t.priority, t.name))
        return candidates[0]

    def _validate_args(self, tool: Tool, kwargs: dict) -> None:
        schema = tool.arg_schema
        if not schema:
            return
        for key in schema.get("required") or []:
            if key not in kwargs:
                raise ToolValidationError(
                    f"missing required arg '{key}' for tool '{tool.name}'"
                )
        props = schema.get("properties") or {}
        type_map = {
            "string": str, "number": (int, float), "integer": int,
            "boolean": bool, "object": dict, "array": list,
        }
        for key, prop in props.items():
            if key not in kwargs:
                continue
            expected = prop.get("type")
            if expected and expected in type_map:
                if not isinstance(kwargs[key], type_map[expected]):
                    raise ToolValidationError(
                        f"arg '{key}' for tool '{tool.name}' must be {expected}"
                    )

    def execute(
        self,
        capability: str,
        scopes: set[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        tool = self.resolve(capability)
        if tool.required_scope is not None and tool.required_scope not in scopes:
            raise PermissionDenied(
                f"Tool '{tool.name}' requires scope '{tool.required_scope}'"
            )
        limiter = self._limiters.get(tool.name)
        if limiter and not limiter.allow():
            raise ToolRuntimeRateLimit(f"rate limit exceeded for tool '{tool.name}'")
        self._validate_args(tool, kwargs)
        limit = (
            timeout if timeout is not None
            else (tool.timeout if tool.timeout is not None else self.default_timeout)
        )
        if limit is None or limit <= 0:
            try:
                return tool.fn(**kwargs)
            except PermissionDenied:
                raise
            except Exception as exc:
                raise classify_tool_exception(exc) from exc
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(tool.fn, **kwargs)
            try:
                return fut.result(timeout=limit)
            except FuturesTimeout as exc:
                fut.cancel()
                raise ToolTimeoutError(
                    f"tool '{tool.name}' timed out after {limit}s"
                ) from exc
            except Exception as exc:
                if isinstance(exc, (PermissionDenied, ToolTimeoutError, ToolValidationError)):
                    raise
                raise classify_tool_exception(exc) from exc

    def execute_parallel(
        self,
        tasks: list[dict],
        scopes: set[str],
        timeout: Optional[float] = None,
    ) -> list[dict]:
        results: List[Optional[dict]] = [None] * len(tasks)
        limit = timeout if timeout is not None else self.default_timeout

        def _run(idx: int, task: dict) -> None:
            capability = task.get("capability", "")
            kwargs = task.get("kwargs") or {}
            try:
                value = self.execute(capability, scopes, timeout=limit, **kwargs)
                results[idx] = {"ok": True, "result": value}
            except Exception as exc:
                results[idx] = {
                    "ok": False,
                    "error": str(exc),
                    "code": getattr(exc, "code", "error"),
                }

        with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
            futures = [pool.submit(_run, i, t) for i, t in enumerate(tasks)]
            for f in as_completed(futures):
                f.result()
        return [r if r is not None else {"ok": False, "error": "missing"} for r in results]
