"""Project 10 — Production Agent Observability.

Instrument an agent with tracing spans, per-call cost/latency accounting,
and a repeated-prompt loop alarm — the testable core of what LangSmith/Arize
give you in production.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional


class Tracer:
    def __init__(self, clock=None):
        """clock: injectable time function (tests pass a fake; default
        time.monotonic). Must set up self.spans: list of finished spans,
        {"name": str, "duration": float, "parent": str | None}, appended in
        FINISH order."""
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self.spans: List[dict] = []
        self._stack: List[str] = []

    @contextmanager
    def span(self, name: str):
        """Context manager measuring a named span.

        Requirements:
            - duration = clock() at exit - clock() at entry.
            - Nested spans record their enclosing span's name as parent
              (top level -> None).
            - The span is recorded even if the body raises (exception must
              propagate).
        """
        parent = self._stack[-1] if self._stack else None
        start = self._clock()
        self._stack.append(name)
        try:
            yield
        finally:
            end = self._clock()
            self._stack.pop()
            self.spans.append(
                {
                    "name": name,
                    "duration": end - start,
                    "parent": parent,
                }
            )


class InstrumentedLLM:
    """Wrap an LLM client with cost accounting and a loop alarm."""

    def __init__(
        self,
        llm,
        tracer: Tracer,
        cost_per_call: float = 0.01,
        loop_threshold: int = 3,
    ):
        """Must set up:
            - self.total_cost, self.call_count
            - self.alerts: list of {"type": "loop", "prompt": str,
              "count": int}
        """
        self.llm = llm
        self.tracer = tracer
        self.cost_per_call = cost_per_call
        self.loop_threshold = loop_threshold
        self.total_cost = 0.0
        self.call_count = 0
        self.alerts: List[dict] = []
        self._prompt_counts: Dict[str, int] = {}

    def complete(self, prompt: str) -> str:
        """Delegate to the wrapped llm inside a tracer span named "llm.complete".

        Requirements:
            - Add cost_per_call to total_cost per call (also on failures).
            - Loop alarm: when the SAME prompt string is seen for the
              loop_threshold-th time, append one alert (once per threshold
              multiple: at 3, 6, 9... for threshold 3).
        """
        self.call_count += 1
        self.total_cost += self.cost_per_call

        count = self._prompt_counts.get(prompt, 0) + 1
        self._prompt_counts[prompt] = count
        if count >= self.loop_threshold and count % self.loop_threshold == 0:
            self.alerts.append(
                {"type": "loop", "prompt": prompt, "count": count}
            )

        with self.tracer.span("llm.complete"):
            return self.llm.complete(prompt)

    def report(self) -> dict:
        """{"calls": int, "total_cost": float, "avg_latency": float
        (mean duration of llm.complete spans, 0.0 when none),
        "alerts": <the alerts list>}."""
        llm_spans = [
            s for s in self.tracer.spans if s["name"] == "llm.complete"
        ]
        avg = (
            sum(s["duration"] for s in llm_spans) / len(llm_spans)
            if llm_spans
            else 0.0
        )
        return {
            "calls": self.call_count,
            "total_cost": self.total_cost,
            "avg_latency": avg,
            "alerts": list(self.alerts),
        }
