"""Project 8 — Multi-Agent Debate System.
N proposer agents answer independently, a critic scores each proposal,
consensus picks a winner, and an aggregator synthesizes the final answer
with a confidence value.

Proposers reply with plain text. The critic replies with JSON:
{"scores": [{"index": 0, "score": 0.0-1.0, "critique": "..."}, ...]}.
"""


class Debate:
    def __init__(self, proposers: list, critic, aggregator):
        """proposers: LLM clients; critic and aggregator: LLM clients."""
        raise NotImplementedError

    def run(self, question: str) -> dict:
        """Run one debate round.
        Requirements:
            - Every proposer is asked the question independently (its prompt
              must NOT contain other proposals).
            - The critic is called once; its prompt must contain ALL proposals;
              parse its scores.
            - Winner = highest score; ties break on LOWER index (deterministic).
            - confidence = winner_score - mean(other scores), clamped to
              [0.0, 1.0]. Single proposer -> confidence = winner_score.
            - The aggregator is called once with the question, the winning
              proposal, and the critic's critique of it; its text response is
              the final answer.
            - Return {"answer": <aggregator text>, "winner_index": int,
              "confidence": float, "proposals": [str, ...],
              "scores": [float, ...]}.
        """
        raise NotImplementedError
