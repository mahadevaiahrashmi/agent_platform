"""Project 10 — Production Agent Observability.
Instrument an agent with tracing spans, per-call cost/latency accounting,
and a repeated-prompt loop alarm — the testable core of what LangSmith/Arize
give you in production.
"""


class Tracer:
    def __init__(self, clock=None):
        """clock: injectable time function (tests pass a fake; default
        time.monotonic). Must set up self.spans: list of finished spans,
        {"name": str, "duration": float, "parent": str | None}, appended in
        FINISH order."""
        raise NotImplementedError

    def span(self, name: str):
        """Context manager measuring a named span.
        Requirements:
            - duration = clock() at exit - clock() at entry.
            - Nested spans record their enclosing span's name as parent
              (top level -> None).
            - The span is recorded even if the body raises (exception must
              propagate).
        """
        raise NotImplementedError


class InstrumentedLLM:
    """Wrap an LLM client with cost accounting and a loop alarm."""

    def __init__(self, llm, tracer: Tracer, cost_per_call: float = 0.01,
                 loop_threshold: int = 3):
        """Must set up:
            - self.total_cost, self.call_count
            - self.alerts: list of {"type": "loop", "prompt": str,
              "count": int}
        """
        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        """Delegate to the wrapped llm inside a tracer span named "llm.complete".
        Requirements:
            - Add cost_per_call to total_cost per call (also on failures).
            - Loop alarm: when the SAME prompt string is seen for the
              loop_threshold-th time, append one alert (once per threshold
              multiple: at 3, 6, 9... for threshold 3).
        """
        raise NotImplementedError

    def report(self) -> dict:
        """{"calls": int, "total_cost": float, "avg_latency": float
        (mean duration of llm.complete spans, 0.0 when none),
        "alerts": <the alerts list>}."""
        raise NotImplementedError
