import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
from tool_orchestrator import Orchestrator, PermissionDenied, Tool


def make_orch():
    o = Orchestrator()
    o.register(Tool("web_search", lambda q: f"web:{q}", {"search"}, priority=1))
    o.register(Tool("kb_search", lambda q: f"kb:{q}", {"search"}, priority=5))
    o.register(Tool("emailer", lambda to: f"sent:{to}", {"notify"}, required_scope="comms"))
    o.register(Tool("sleeper", lambda s: (time.sleep(s), "slept")[1], {"wait"}))
    return o


def test_capability_routing_conflict_resolution():
    o = make_orch()
    assert o.resolve("search").name == "kb_search"


def test_tie_breaks_deterministically():
    o = Orchestrator()
    o.register(Tool("b_tool", lambda: "b", {"x"}, priority=1))
    o.register(Tool("a_tool", lambda: "a", {"x"}, priority=1))
    assert o.resolve("x").name == "a_tool"


def test_unknown_capability_raises():
    with pytest.raises(KeyError):
        make_orch().resolve("teleport")


def test_permission_scoping():
    o = make_orch()
    assert o.execute("notify", scopes={"comms"}, to="a@b.c") == "sent:a@b.c"
    with pytest.raises(PermissionDenied):
        o.execute("notify", scopes=set(), to="a@b.c")


def test_reregistration_replaces():
    o = make_orch()
    o.register(Tool("kb_search", lambda q: f"kb2:{q}", {"search"}, priority=5))
    assert o.execute("search", scopes=set(), q="x") == "kb2:x"


def test_parallel_execution_is_concurrent_and_ordered():
    o = make_orch()
    tasks = [{"capability": "wait", "kwargs": {"s": 0.3}} for _ in range(4)]
    start = time.monotonic()
    results = o.execute_parallel(tasks, scopes=set())
    elapsed = time.monotonic() - start
    assert elapsed < 0.9, "4 x 0.3s sleeps should overlap, not serialize"
    assert all(r["ok"] and r["result"] == "slept" for r in results)


def test_parallel_isolates_failures():
    o = make_orch()
    tasks = [
        {"capability": "search", "kwargs": {"q": "a"}},
        {"capability": "notify", "kwargs": {"to": "x"}},
        {"capability": "missing", "kwargs": {}},
        {"capability": "search", "kwargs": {"q": "b"}},
    ]
    results = o.execute_parallel(tasks, scopes=set())
    assert results[0] == {"ok": True, "result": "kb:a"}
    assert results[1]["ok"] is False
    assert results[2]["ok"] is False
    assert results[3] == {"ok": True, "result": "kb:b"}
