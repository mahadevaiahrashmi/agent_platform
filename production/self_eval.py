"""Project 9 — Self-Eval (production, stack-independent)."""
from __future__ import annotations

import json
from typing import List, Optional


class SelfEvalAgent:
    def __init__(
        self,
        worker,
        judge,
        pass_threshold: float = 0.8,
        max_attempts: int = 3,
        *,
        cost_ceiling: Optional[float] = None,
        cost_per_call: float = 1.0,
    ):
        self.worker = worker
        self.judge = judge
        self.pass_threshold = pass_threshold
        self.max_attempts = max_attempts
        self.history: List[dict] = []
        self.cost_ceiling = cost_ceiling
        self.cost_per_call = cost_per_call
        self.spent = 0.0

    def _charge(self) -> None:
        self.spent += self.cost_per_call
        if self.cost_ceiling is not None and self.spent > self.cost_ceiling:
            raise RuntimeError("self-eval cost ceiling exceeded")

    def run(self, task: str, criteria: str) -> dict:
        self.history = []
        prev_output = None
        prev_critique = None
        best_output = ""
        best_score = -1.0
        first_score = None
        last_score = 0.0
        passed = False

        for attempt in range(1, self.max_attempts + 1):
            if attempt == 1:
                prompt = f"Task: {task}\n\nCriteria: {criteria}\n\nOutput:"
            else:
                prompt = (
                    f"Task: {task}\n\nCriteria: {criteria}\n\n"
                    f"Previous output:\n{prev_output}\n\n"
                    f"Critique:\n{prev_critique}\n\nImproved output:"
                )
            output = self.worker.complete(prompt)
            self._charge()
            judge_prompt = (
                f"Criteria: {criteria}\n\nOutput:\n{output}\n\n"
                f'Respond with JSON: {{"score": 0.0-1.0, "critique": "..."}}'
            )
            judge_raw = self.judge.complete(judge_prompt)
            self._charge()
            judge_data = json.loads(judge_raw)
            score = float(judge_data["score"])
            critique = str(judge_data.get("critique", ""))
            if first_score is None:
                first_score = score
            last_score = score
            self.history.append({
                "attempt": attempt, "output": output, "score": score, "critique": critique,
            })
            if score > best_score:
                best_score = score
                best_output = output
            if score >= self.pass_threshold:
                passed = True
                break
            prev_output = output
            prev_critique = critique

        improvement = 0.0
        if first_score is not None and len(self.history) > 1:
            improvement = last_score - first_score
        return {
            "output": best_output, "score": best_score, "passed": passed,
            "attempts": len(self.history), "improvement": improvement,
        }
