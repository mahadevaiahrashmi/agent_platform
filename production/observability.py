"""Project 10 — Observability (production, stack-independent)."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

from interfaces import NullSpanExporter, Redactor, SpanExporter, identity_redactor


class Tracer:
    def __init__(
        self,
        clock=None,
        *,
        exporter: Optional[SpanExporter] = None,
        correlation_id: Optional[str] = None,
    ):
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self.spans: List[dict] = []
        self._stack: List[str] = []
        self.exporter = exporter or NullSpanExporter()
        self.correlation_id = correlation_id or str(uuid.uuid4())

    @contextmanager
    def span(self, name: str):
        parent = self._stack[-1] if self._stack else None
        start = self._clock()
        self._stack.append(name)
        try:
            yield
        finally:
            end = self._clock()
            self._stack.pop()
            record = {
                "name": name,
                "duration": end - start,
                "parent": parent,
                "correlation_id": self.correlation_id,
            }
            self.spans.append(record)
            self.exporter.export([record])


class InstrumentedLLM:
    def __init__(
        self,
        llm,
        tracer: Tracer,
        cost_per_call: float = 0.01,
        loop_threshold: int = 3,
        *,
        redactor: Optional[Redactor] = None,
    ):
        self.llm = llm
        self.tracer = tracer
        self.cost_per_call = cost_per_call
        self.loop_threshold = loop_threshold
        self.total_cost = 0.0
        self.call_count = 0
        self.alerts: List[dict] = []
        self._prompt_counts: Dict[str, int] = {}
        self.redactor = redactor or identity_redactor

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.total_cost += self.cost_per_call
        count = self._prompt_counts.get(prompt, 0) + 1
        self._prompt_counts[prompt] = count
        if count >= self.loop_threshold and count % self.loop_threshold == 0:
            self.alerts.append({
                "type": "loop", "prompt": self.redactor(prompt), "count": count,
            })
        with self.tracer.span("llm.complete"):
            return self.llm.complete(prompt)

    def report(self) -> dict:
        llm_spans = [s for s in self.tracer.spans if s["name"] == "llm.complete"]
        avg = sum(s["duration"] for s in llm_spans) / len(llm_spans) if llm_spans else 0.0
        return {
            "calls": self.call_count,
            "total_cost": self.total_cost,
            "avg_latency": avg,
            "alerts": list(self.alerts),
            "correlation_id": self.tracer.correlation_id,
        }
