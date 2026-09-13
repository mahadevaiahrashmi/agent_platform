"""Trace spans and wrap an LLM with per-call accounting and loop detection.

Spans use the injected clock, capture parent names, and are appended in finish
order even when exceptions propagate; each wrapped call runs inside a span
named llm.complete, increments count and cost even on failure, and alarms at
every repeated-prompt threshold multiple; reports contain calls, total cost,
mean completed LLM-span latency, and alerts.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Span:
    name: str
    start: float
    end: Optional[float] = None
    parent: Optional[str] = None
    error: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.end is None:
            return None
        return self.end - self.start


@dataclass
class ObservabilityReport:
    calls: int
    total_cost: float
    mean_llm_latency: float
    alerts: List[str]
    spans: List[Span]


class Tracer:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self.spans: List[Span] = []
        self._stack: List[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any):
        parent_name = self._stack[-1].name if self._stack else None
        sp = Span(
            name=name,
            start=self._clock(),
            parent=parent_name,
            attributes=dict(attributes),
        )
        self._stack.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.error = str(exc)
            raise
        finally:
            sp.end = self._clock()
            self._stack.pop()
            # Append in finish order even when exceptions propagate
            self.spans.append(sp)


class InstrumentedLLM:
    """Wrap an LLM, recording every complete() call inside an llm.complete span."""

    def __init__(
        self,
        llm: Any,
        tracer: Tracer,
        cost_per_call: float = 0.01,
        loop_threshold: int = 3,
    ):
        self.llm = llm
        self.tracer = tracer
        self.cost_per_call = cost_per_call
        self.loop_threshold = loop_threshold
        self.call_count = 0
        self.total_cost = 0.0
        self._prompt_counts: Dict[str, int] = {}
        self.alerts: List[str] = []

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        self.total_cost += self.cost_per_call

        # Loop detection
        key = prompt.strip()
        self._prompt_counts[key] = self._prompt_counts.get(key, 0) + 1
        count = self._prompt_counts[key]
        if count >= self.loop_threshold and count % self.loop_threshold == 0:
            alert = f"repeated_prompt_threshold:{count}"
            self.alerts.append(alert)

        with self.tracer.span("llm.complete", prompt_len=len(prompt)):
            try:
                return self.llm.complete(prompt, **kwargs)
            except Exception:
                # cost and count already incremented
                raise

    def report(self) -> ObservabilityReport:
        llm_spans = [
            s for s in self.tracer.spans
            if s.name == "llm.complete" and s.duration is not None
        ]
        mean_lat = (
            sum(s.duration for s in llm_spans) / len(llm_spans)
            if llm_spans
            else 0.0
        )
        return ObservabilityReport(
            calls=self.call_count,
            total_cost=self.total_cost,
            mean_llm_latency=mean_lat,
            alerts=list(self.alerts),
            spans=list(self.tracer.spans),
        )
