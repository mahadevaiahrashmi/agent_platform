import json
from fake_llm import FakeLLM
from react_loop import ReActAgent

TOOLS = {
    "search": lambda query: f"results for {query}",
    "add": lambda a, b: str(a + b),
    "boom": lambda: (_ for _ in ()).throw(RuntimeError("tool exploded")),
}

ACT_SEARCH = json.dumps({"thought": "need data", "action": "search", "args": {"query": "rag"}})
ACT_ADD = json.dumps({"thought": "sum it", "action": "add", "args": {"a": 2, "b": 3}})
ACT_MISSING = json.dumps({"thought": "try", "action": "no_such_tool", "args": {}})
ACT_BOOM = json.dumps({"thought": "risky", "action": "boom", "args": {}})
FINAL = json.dumps({"thought": "done", "final": "the answer is 5"})


def test_happy_path_two_steps():
    llm = FakeLLM([ACT_SEARCH, ACT_ADD, FINAL])
    agent = ReActAgent(llm, TOOLS)
    result = agent.run("compute something")
    assert result["status"] == "done"
    assert result["answer"] == "the answer is 5"
    assert result["iterations"] == 3
    assert len(agent.trace) == 3
    assert agent.trace[0]["observation"] == "results for rag"
    assert agent.trace[1]["observation"] == "5"


def test_observations_fed_back_to_llm():
    llm = FakeLLM([ACT_SEARCH, FINAL])
    agent = ReActAgent(llm, TOOLS)
    agent.run("goal")
    assert "results for rag" in llm.calls[1]["prompt"]


def test_max_iterations_degrades_gracefully():
    llm = FakeLLM([ACT_SEARCH] * 10)
    agent = ReActAgent(llm, TOOLS, max_iterations=4)
    result = agent.run("goal")
    assert result["status"] == "max_iterations"
    assert result["iterations"] == 4
    assert llm.call_count == 4
    assert isinstance(result["answer"], str) and result["answer"].strip()


def test_unknown_tool_recovers():
    llm = FakeLLM([ACT_MISSING, FINAL])
    agent = ReActAgent(llm, TOOLS)
    result = agent.run("goal")
    assert result["status"] == "done"
    assert "ERROR" in agent.trace[0]["observation"]


def test_tool_exception_recovers():
    llm = FakeLLM([ACT_BOOM, FINAL])
    agent = ReActAgent(llm, TOOLS)
    result = agent.run("goal")
    assert result["status"] == "done"
    assert "ERROR" in agent.trace[0]["observation"]


def test_invalid_llm_output_counts_as_iteration():
    llm = FakeLLM(["I think we should probably...", FINAL])
    agent = ReActAgent(llm, TOOLS)
    result = agent.run("goal")
    assert result["status"] == "done"
    assert result["iterations"] == 2
