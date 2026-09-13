import json
import pytest
from fake_llm import FakeLLM
from cost_router import BudgetExceeded, CostRouter, estimate_complexity

SURE = json.dumps({"answer": "42", "confidence": 0.9})
UNSURE = json.dumps({"answer": "hmm", "confidence": 0.3})


def models(small_responses, large_responses):
    return {
        "small": {"client": FakeLLM(small_responses), "cost_per_call": 1.0, "max_complexity": 20},
        "large": {"client": FakeLLM(large_responses), "cost_per_call": 10.0, "max_complexity": 1000},
    }


def test_complexity_heuristic():
    assert estimate_complexity("what is two plus two") == 5
    assert estimate_complexity("analyze this trace") == 13
    assert estimate_complexity("Compare and analyze the architecture") == 35


def test_routes_simple_to_cheap_and_complex_to_capable():
    r = CostRouter(models([], []), budget=100)
    assert r.route("short question") == "small"
    long_task = "analyze compare architecture " + "word " * 30
    assert r.route(long_task) == "large"


def test_early_exit_skips_escalation():
    m = models([SURE], [])
    r = CostRouter(m, budget=100)
    result = r.run_task("easy question")
    assert result["model"] == "small" and result["cost"] == 1.0
    assert m["large"]["client"].call_count == 0
    assert r.analytics()["escalation_rate"] == 0.0


def test_low_confidence_escalates_once():
    m = models([UNSURE], [SURE])
    r = CostRouter(m, budget=100)
    result = r.run_task("tricky short question")
    assert result["model"] == "large"
    assert result["cost"] == 11.0
    assert r.analytics()["escalation_rate"] == 1.0


def test_budget_enforced_before_call():
    m = models([SURE, SURE], [])
    r = CostRouter(m, budget=1.5)
    r.run_task("first")
    with pytest.raises(BudgetExceeded):
        r.run_task("second")
    assert m["small"]["client"].call_count == 1


def test_escalation_respects_budget():
    m = models([UNSURE], [SURE])
    r = CostRouter(m, budget=5)
    result = r.run_task("short")
    assert result["model"] == "small"
    assert result["cost"] == 1.0


def test_analytics_totals():
    m = models([SURE, UNSURE], [SURE])
    r = CostRouter(m, budget=100)
    r.run_task("one")
    r.run_task("two")
    a = r.analytics()
    assert a["total_cost"] == 12.0
    assert a["calls"] == 3
    assert a["by_model"] == {"small": 2.0, "large": 10.0}
    assert a["escalation_rate"] == 0.5
