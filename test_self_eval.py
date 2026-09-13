import json
from fake_llm import FakeLLM
from self_eval import SelfEvalAgent


def judge_json(score, critique="needs work"):
    return json.dumps({"score": score, "critique": critique})


def test_passes_first_try():
    worker = FakeLLM(["draft one"])
    judge = FakeLLM([judge_json(0.9, "great")])
    agent = SelfEvalAgent(worker, judge)
    result = agent.run("write a haiku", "must be 3 lines")
    assert result == {"output": "draft one", "score": 0.9, "passed": True,
                      "attempts": 1, "improvement": 0.0}
    assert "write a haiku" in worker.calls[0]["prompt"]
    assert "must be 3 lines" in worker.calls[0]["prompt"]


def test_judge_sees_criteria_and_output():
    worker = FakeLLM(["draft one"])
    judge = FakeLLM([judge_json(0.9)])
    SelfEvalAgent(worker, judge).run("task", "criteria-marker")
    assert "criteria-marker" in judge.calls[0]["prompt"]
    assert "draft one" in judge.calls[0]["prompt"]


def test_regenerates_with_critique_constraint():
    worker = FakeLLM(["bad draft", "good draft"])
    judge = FakeLLM([judge_json(0.3, "too vague, add numbers"), judge_json(0.85, "ok")])
    agent = SelfEvalAgent(worker, judge)
    result = agent.run("task", "criteria")
    assert result["passed"] is True and result["attempts"] == 2
    retry_prompt = worker.calls[1]["prompt"]
    assert "too vague, add numbers" in retry_prompt
    assert "bad draft" in retry_prompt
    assert abs(result["improvement"] - 0.55) < 1e-9


def test_stops_at_max_attempts_returns_best():
    worker = FakeLLM(["d1", "d2", "d3", "d4"])
    judge = FakeLLM([judge_json(0.2), judge_json(0.6), judge_json(0.4)])
    agent = SelfEvalAgent(worker, judge, max_attempts=3)
    result = agent.run("task", "criteria")
    assert result["attempts"] == 3
    assert worker.call_count == 3
    assert result["passed"] is False
    assert result["output"] == "d2" and result["score"] == 0.6
    assert abs(result["improvement"] - 0.2) < 1e-9


def test_history_logged():
    worker = FakeLLM(["d1", "d2"])
    judge = FakeLLM([judge_json(0.1, "c1"), judge_json(0.9, "c2")])
    agent = SelfEvalAgent(worker, judge)
    agent.run("task", "criteria")
    assert [h["attempt"] for h in agent.history] == [1, 2]
    assert agent.history[0] == {"attempt": 1, "output": "d1", "score": 0.1, "critique": "c1"}
