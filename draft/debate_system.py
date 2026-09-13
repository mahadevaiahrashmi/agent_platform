"""Gather independent proposals, score them with a critic, and synthesize the selected answer.

Proposers do not see one another's work; the critic sees every proposal and is
called once; highest score wins with lower index breaking ties; confidence is
the clamped winner score minus the mean of other scores, or the winner score
for one proposer; the aggregator receives the question, winning proposal, and
its critique exactly once; the required result fields are returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


@dataclass
class Proposal:
    index: int
    text: str
    score: float = 0.0
    critique: str = ""


@dataclass
class DebateResult:
    answer: str
    winning_proposal: str
    confidence: float
    scores: List[float]
    critiques: List[str]
    selected_index: int


class DebateSystem:
    def __init__(
        self,
        proposer_llm: Any,
        critic_llm: Any,
        aggregator_llm: Any,
        num_proposers: int = 3,
    ):
        self.proposer_llm = proposer_llm
        self.critic_llm = critic_llm
        self.aggregator_llm = aggregator_llm
        self.num_proposers = num_proposers

    def run(self, question: str) -> DebateResult:
        # Independent proposals – proposers do not see each other
        proposals: List[Proposal] = []
        for i in range(self.num_proposers):
            prompt = (
                f"Propose a clear, self-contained answer to the following question.\n\n"
                f"Question: {question}\n\n"
                f"Proposal:"
            )
            text = self.proposer_llm.complete(prompt)
            proposals.append(Proposal(index=i, text=text.strip()))

        # Critic sees every proposal, called once
        critic_prompt_parts = [
            "Score each proposal from 0.0 to 1.0 and provide a short critique.",
            f"Question: {question}",
            "",
        ]
        for p in proposals:
            critic_prompt_parts.append(f"Proposal {p.index}:\n{p.text}\n")
        critic_prompt_parts.append(
            "Respond with one line per proposal: SCORE: <float> CRITIQUE: <text>"
        )
        critic_raw = self.critic_llm.complete("\n".join(critic_prompt_parts))

        # Parse critic output (best-effort)
        lines = [ln.strip() for ln in critic_raw.splitlines() if ln.strip()]
        for i, p in enumerate(proposals):
            score = 0.5
            critique = ""
            if i < len(lines):
                line = lines[i]
                # crude parse
                import re
                m = re.search(r"(?i)SCORE\s*[:\-]?\s*([0-9.]+)", line)
                if m:
                    try:
                        score = float(m.group(1))
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass
                m2 = re.search(r"(?i)CRITIQUE\s*[:\-]?\s*(.*)$", line)
                if m2:
                    critique = m2.group(1).strip()
                else:
                    critique = line
            p.score = score
            p.critique = critique

        # Highest score wins; lower index breaks ties
        ranked = sorted(proposals, key=lambda p: (-p.score, p.index))
        winner = ranked[0]

        # Confidence: winner - mean(others), clamped; or winner if single
        if len(proposals) == 1:
            confidence = winner.score
        else:
            others = [p.score for p in proposals if p.index != winner.index]
            mean_others = sum(others) / len(others) if others else 0.0
            confidence = max(0.0, min(1.0, winner.score - mean_others))

        # Aggregator receives question, winning proposal, its critique exactly once
        agg_prompt = (
            f"Synthesize a final answer.\n\n"
            f"Question: {question}\n\n"
            f"Winning proposal:\n{winner.text}\n\n"
            f"Critique of winning proposal:\n{winner.critique}\n\n"
            f"Final answer:"
        )
        answer = self.aggregator_llm.complete(agg_prompt).strip()

        return DebateResult(
            answer=answer,
            winning_proposal=winner.text,
            confidence=confidence,
            scores=[p.score for p in proposals],
            critiques=[p.critique for p in proposals],
            selected_index=winner.index,
        )
