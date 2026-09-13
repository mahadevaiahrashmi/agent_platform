"""Project 8 — Multi-Agent Debate System.

N proposer agents answer independently, a critic scores each proposal,
consensus picks a winner, and an aggregator synthesizes the final answer
with a confidence value.

Proposers reply with plain text. The critic replies with JSON:
{"scores": [{"index": 0, "score": 0.0-1.0, "critique": "..."}, ...]}.
"""

from __future__ import annotations

import json
from typing import Any, List


class Debate:
    def __init__(self, proposers: list, critic, aggregator):
        """proposers: LLM clients; critic and aggregator: LLM clients."""
        self.proposers = proposers
        self.critic = critic
        self.aggregator = aggregator

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
        # Independent proposals
        proposals: List[str] = []
        for p in self.proposers:
            text = p.complete(question)
            proposals.append(text)

        # Critic sees all
        critic_prompt_parts = [f"Question: {question}", "", "Proposals:"]
        for i, prop in enumerate(proposals):
            critic_prompt_parts.append(f"[{i}] {prop}")
        critic_prompt_parts.append(
            "\nRespond with JSON: "
            '{"scores": [{"index": 0, "score": 0.0-1.0, "critique": "..."}, ...]}'
        )
        critic_raw = self.critic.complete("\n".join(critic_prompt_parts))
        critic_data = json.loads(critic_raw)
        score_entries = critic_data["scores"]

        # Build score + critique lists aligned by index
        scores = [0.0] * len(proposals)
        critiques = [""] * len(proposals)
        for entry in score_entries:
            idx = int(entry["index"])
            if 0 <= idx < len(proposals):
                scores[idx] = float(entry["score"])
                critiques[idx] = str(entry.get("critique", ""))

        # Winner: highest score, lower index on tie
        winner_index = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[winner_index] or (
                scores[i] == scores[winner_index] and i < winner_index
            ):
                winner_index = i
        # Actually lower index already preferred by scanning left-to-right with >
        # but to be explicit:
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        winner_index = best

        winner_score = scores[winner_index]
        if len(scores) == 1:
            confidence = winner_score
        else:
            others = [s for i, s in enumerate(scores) if i != winner_index]
            mean_others = sum(others) / len(others) if others else 0.0
            confidence = max(0.0, min(1.0, winner_score - mean_others))

        # Aggregator
        agg_prompt = (
            f"Question: {question}\n\n"
            f"Winning proposal: {proposals[winner_index]}\n\n"
            f"Critique: {critiques[winner_index]}\n\n"
            f"Synthesize the final answer:"
        )
        answer = self.aggregator.complete(agg_prompt)

        return {
            "answer": answer,
            "winner_index": winner_index,
            "confidence": confidence,
            "proposals": proposals,
            "scores": scores,
        }
