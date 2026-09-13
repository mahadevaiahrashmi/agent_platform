"""Project 9 — Self-Reflective Agent with Auto-Eval.
Execute -> LLM-as-judge -> critique -> regenerate under critique constraints,
keeping improvement metrics and returning the best attempt.

The worker LLM replies with plain text. The judge replies with JSON:
{"score": 0.0-1.0, "critique": "..."}.
"""


class SelfEvalAgent:
    def __init__(self, worker, judge, pass_threshold: float = 0.8,
                 max_attempts: int = 3):
        """Must set up self.history: one entry per attempt,
        {"attempt": int (1-based), "output": str, "score": float,
        "critique": str}."""
        raise NotImplementedError

    def run(self, task: str, criteria: str) -> dict:
        """Generate-judge-refine loop.
        Requirements:
            - Attempt 1: worker prompt contains the task and the criteria.
            - Judge every attempt: judge prompt must contain the criteria and
              the attempt's output; parse score + critique.
            - score >= pass_threshold -> stop immediately.
            - Otherwise regenerate: the next worker prompt MUST contain the
              judge's critique of the previous attempt (this is what makes it
              self-reflective) — plus the previous output.
            - Stop after max_attempts regardless.
            - Return {"output": <output of the BEST-scoring attempt>,
              "score": <its score>, "passed": bool,
              "attempts": <number made>,
              "improvement": <last score - first score, 0.0 for one attempt>}.
        """
        raise NotImplementedError
