from fake_llm import FakeLLM
from memory_agent import Memory


def test_short_term_window_and_eviction():
    m = Memory(FakeLLM([]), short_window=2)
    for t in ["turn one", "turn two", "turn three"]:
        m.add_turn(t)
    assert m.short_term == ["turn two", "turn three"]
    assert [e["text"] for e in m.long_term] == ["turn one"]
    assert m.long_term[0]["source"] == "turn"


def test_relevance_scoring():
    m = Memory(FakeLLM([]))
    assert m.relevance("python bug", "we fixed a python bug today") == 1.0
    assert m.relevance("python bug", "the weather is nice") == 0.0
    assert abs(m.relevance("python bug fix", "python only") - 1 / 3) < 1e-9
    assert m.relevance("", "anything") == 0.0


def test_recall_ranks_and_filters():
    m = Memory(FakeLLM([]), short_window=1)
    for t in ["user likes python and rust",
              "user ordered pizza",
              "python question about generators",
              "meeting at noon"]:
        m.add_turn(t)
    hits = m.recall("python generators", k=2)
    assert hits[0] == "python question about generators"
    assert hits[1] == "user likes python and rust"
    assert "meeting at noon" not in m.recall("python generators", k=10)


def test_compress_uses_llm_and_replaces_turns():
    llm = FakeLLM(["SUMMARY: user is a python dev who likes pizza"])
    m = Memory(llm, short_window=1)
    for t in ["user codes python", "user likes pizza", "current turn"]:
        m.add_turn(t)
    m.compress()
    assert llm.call_count == 1
    assert "user codes python" in llm.calls[0]["prompt"]
    summaries = [e for e in m.long_term if e["source"] == "summary"]
    turns = [e for e in m.long_term if e["source"] == "turn"]
    assert len(summaries) == 1 and turns == []
    assert summaries[0]["text"].startswith("SUMMARY:")


def test_compress_noop_when_nothing_to_compress():
    llm = FakeLLM([])
    m = Memory(llm, short_window=5)
    m.add_turn("only turn")
    m.compress()
    assert llm.call_count == 0


def test_cross_session_save_load(tmp_path):
    llm = FakeLLM([])
    m = Memory(llm, short_window=2)
    for t in ["alpha beta", "gamma delta", "epsilon zeta"]:
        m.add_turn(t)
    p = str(tmp_path / "mem.json")
    m.save(p)
    restored = Memory.load(FakeLLM([]), p)
    assert restored.short_term == m.short_term
    assert restored.long_term == m.long_term
    assert restored.recall("alpha", k=1) == ["alpha beta"]
