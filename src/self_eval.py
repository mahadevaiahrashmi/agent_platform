"""Run a worker/judge refinement loop and retain evaluation history.

The first worker prompt includes task and criteria; every output is judged
against the criteria; passing scores stop immediately; retries include both
the previous output and critique; execution stops at max_attempts; the
best-scoring output is returned, while improvement is the last score minus
the first; every attempt is recorded with 1-based numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class AttemptRecord:
    attempt: int  # 1-based
    output: str
    score: float
    critique: str


@dataclass
class SelfEvalResult:
    output: str
    score: float
    improvement: float
    attempts: List[AttemptRecord] = field(default_factory=list)
    passed: bool = False


class SelfEval:
    def __init__(
        self,
        worker_llm: Any,
        judge_llm: Any,
        pass_threshold: float = 0.8,
        max_attempts: int = 3,
    ):
        self.worker_llm = worker_llm
        self.judge_llm = judge_llm
        self.pass_threshold = pass_threshold
        self.max_attempts = max_attempts

    def run(self, task: str, criteria: str) -> SelfEvalResult:
        attempts: List[AttemptRecord] = []
        prev_output: Optional[str] = None
        prev_critique: Optional[str] = None
        best_output = ""
        best_score = -1.0
        first_score: Optional[float] = None

        for i in range(1, self.max_attempts + 1):
            if i == 1:
                prompt = (
                    f"Complete the following task.\n\n"
                    f"Task: {task}\n\n"
                    f"Criteria for success:\n{criteria}\n\n"
                    f"Output:"
                )
            else:
                prompt = (
                    f"Improve the previous output based on the critique.\n\n"
                    f"Task: {task}\n\n"
                    f"Criteria:\n{criteria}\n\n"
                    f"Previous output:\n{prev_output}\n\n"
                    f"Critique:\n{prev_critique}\n\n"
                    f"Improved output:"
                )

            output = self.worker_llm.complete(prompt).strip()

            # Judge
            judge_prompt = (
                f"Evaluate the output against the criteria. "
                f"Respond with SCORE: <0.0-1.0> CRITIQUE: <text>\n\n"
                f"Criteria:\n{criteria}\n\n"
                f"Output:\n{output}\n"
            )
            judge_raw = self.judge_llm.complete(judge_prompt)
            score, critique = self._parse_judge(judge_raw)

            if first_score is None:
                first_score = score
            if score > best_score:
                best_score = score
                best_output = output

            rec = AttemptRecord(attempt=i, output=output, score=score, critique=critique)
            attempts.append(rec)

            if score >= self.pass_threshold:
                improvement = score - (first_score or score)
                return SelfEvalResult(
                    output=output,
                    score=score,
                    improvement=improvement,
                    attempts=attempts,
                    passed=True,
                )

            prev_output = output
            prev_critique = critique

        # Exhausted attempts – return best
        last_score = attempts[-1].score if attempts else 0.0
        improvement = last_score - (first_score or last_score)
        return SelfEvalResult(
            output=best_output,
            score=best_score,
            improvement=improvement,
            attempts=attempts,
            passed=False,
        )

    def _parse_judge(self, raw: str) -> tuple[float, str]:
        import re
        score = 0.5
        critique = raw.strip()
        m = re.search(r"(?i)SCORE\s*[:\-]?\s*([0-9.]+)", raw)
        if m:
            try:
                score = float(m.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        m2 = re.search(r"(?i)CRITIQUE\s*[:\-]?\s*(.*)$", raw, re.DOTALL)
        if m2:
            critique = m2.group(1).strip()
        return score, critique
