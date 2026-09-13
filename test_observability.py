import pytest
from fake_llm import FakeLLM
from observability import InstrumentedLLM, Tracer


class FakeClock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        self.t += 1.0
        return self.t


def test_span_duration_and_order():
    tracer = Tracer(clock=FakeClock())
    with tracer.span("outer"):
        pass
    assert len(tracer.spans) == 1
    s = tracer.spans[0]
    assert s["name"] == "outer" and s["parent"] is None
    assert s["duration"] == 1.0


def test_nested_spans_have_parents():
    tracer = Tracer(clock=FakeClock())
    with tracer.span("outer"):
        with tracer.span("inner"):
            pass
    names = {s["name"]: s for s in tracer.spans}
    assert names["inner"]["parent"] == "outer"
    assert names["outer"]["parent"] is None
    assert tracer.spans[0]["name"] == "inner"


def test_span_recorded_on_exception():
    tracer = Tracer(clock=FakeClock())
    with pytest.raises(ValueError):
        with tracer.span("boom"):
            raise ValueError("x")
    assert tracer.spans and tracer.spans[0]["name"] == "boom"


def test_cost_accounting():
    llm = InstrumentedLLM(FakeLLM(["a", "b"]), Tracer(clock=FakeClock()), cost_per_call=0.5)
    llm.complete("p1")
    llm.complete("p2")
    assert llm.call_count == 2
    assert abs(llm.total_cost - 1.0) < 1e-9


def test_cost_counted_on_failure_too():
    llm = InstrumentedLLM(FakeLLM([RuntimeError("api down")]), Tracer(clock=FakeClock()),
                          cost_per_call=0.5)
    with pytest.raises(RuntimeError):
        llm.complete("p")
    assert llm.total_cost == 0.5


def test_loop_alarm_fires_at_threshold_multiples():
    llm = InstrumentedLLM(FakeLLM(["x"] * 7), Tracer(clock=FakeClock()), loop_threshold=3)
    for _ in range(6):
        llm.complete("same prompt")
    llm.complete("different prompt")
    assert len(llm.alerts) == 2
    assert llm.alerts[0] == {"type": "loop", "prompt": "same prompt", "count": 3}
    assert llm.alerts[1]["count"] == 6


def test_report():
    llm = InstrumentedLLM(FakeLLM(["a", "b"]), Tracer(clock=FakeClock()), cost_per_call=1.0)
    llm.complete("p1")
    llm.complete("p2")
    r = llm.report()
    assert r["calls"] == 2 and r["total_cost"] == 2.0
    assert r["avg_latency"] == 1.0
    assert r["alerts"] == []
