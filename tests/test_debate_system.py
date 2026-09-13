import json
from fake_llm import FakeLLM
from debate_system import Debate


def critic_json(*scores):
    return json.dumps({"scores": [
        {"index": i, "score": s, "critique": f"critique-{i}"} for i, s in enumerate(scores)
    ]})


def test_full_debate_flow():
    proposers = [FakeLLM(["use FAISS"]), FakeLLM(["use pgvector"]), FakeLLM(["use pinecone"])]
    critic = FakeLLM([critic_json(0.5, 0.9, 0.6)])
    aggregator = FakeLLM(["Final: pgvector, because managed SQL."])
    debate = Debate(proposers, critic, aggregator)
    result = debate.run("which vector store?")
    assert result["winner_index"] == 1
    assert result["proposals"] == ["use FAISS", "use pgvector", "use pinecone"]
    assert result["scores"] == [0.5, 0.9, 0.6]
    assert result["answer"] == "Final: pgvector, because managed SQL."
    assert abs(result["confidence"] - (0.9 - 0.55)) < 1e-9


def test_proposers_are_independent():
    proposers = [FakeLLM(["A"]), FakeLLM(["B"])]
    critic = FakeLLM([critic_json(0.9, 0.1)])
    debate = Debate(proposers, critic, FakeLLM(["final"]))
    debate.run("q")
    assert "B" not in proposers[0].calls[0]["prompt"]
    assert "A" not in proposers[1].calls[0]["prompt"]


def test_critic_sees_all_proposals():
    proposers = [FakeLLM(["alpha-idea"]), FakeLLM(["beta-idea"])]
    critic = FakeLLM([critic_json(0.5, 0.6)])
    debate = Debate(proposers, critic, FakeLLM(["final"]))
    debate.run("q")
    assert "alpha-idea" in critic.calls[0]["prompt"]
    assert "beta-idea" in critic.calls[0]["prompt"]


def test_tie_breaks_on_lower_index():
    proposers = [FakeLLM(["A"]), FakeLLM(["B"])]
    critic = FakeLLM([critic_json(0.7, 0.7)])
    result = Debate(proposers, critic, FakeLLM(["final"])).run("q")
    assert result["winner_index"] == 0


def test_aggregator_gets_winner_and_critique():
    proposers = [FakeLLM(["loser"]), FakeLLM(["winner-proposal"])]
    critic = FakeLLM([critic_json(0.1, 0.9)])
    aggregator = FakeLLM(["final"])
    Debate(proposers, critic, aggregator).run("the question")
    prompt = aggregator.calls[0]["prompt"]
    assert "winner-proposal" in prompt and "critique-1" in prompt and "the question" in prompt


def test_single_proposer_confidence():
    result = Debate([FakeLLM(["only"])], FakeLLM([critic_json(0.8)]), FakeLLM(["f"])).run("q")
    assert abs(result["confidence"] - 0.8) < 1e-9


def test_confidence_clamped():
    proposers = [FakeLLM(["A"]), FakeLLM(["B"])]
    critic = FakeLLM([critic_json(1.0, -0.5)])
    result = Debate(proposers, critic, FakeLLM(["f"])).run("q")
    assert 0.0 <= result["confidence"] <= 1.0
