"""Project 6 — Cost-Aware Agent Router.
Route tasks to the cheapest capable model, keep a hard token budget,
exit early on confident cheap answers, and report cost analytics.

Each model config: {"client": <llm>, "cost_per_call": float,
"max_complexity": int}. Model clients reply with JSON:
{"answer": "...", "confidence": 0.0-1.0}.
"""


class BudgetExceeded(Exception):
    pass


def estimate_complexity(task: str) -> int:
    """Deterministic complexity heuristic.
    Requirements (exactly):
        - base = number of whitespace-separated words
        - +10 for each of these keywords present (case-insensitive):
          "analyze", "compare", "architecture", "multi-step", "prove"
    """
    raise NotImplementedError


class CostRouter:
    def __init__(self, models: dict[str, dict], budget: float,
                 confidence_exit: float = 0.8):
        """models: name -> config. Must set up self.ledger: list of
        {"task": str, "model": str, "cost": float, "escalated": bool}."""
        raise NotImplementedError

    def route(self, task: str) -> str:
        """Return the name of the CHEAPEST model whose max_complexity >=
        estimate_complexity(task). No capable model -> the most capable one."""
        raise NotImplementedError

    def run_task(self, task: str) -> dict:
        """Execute one task.
        Requirements:
            - Spending another call's cost must never push total spend past
              the budget: raise BudgetExceeded BEFORE calling the model.
            - Call the routed model. If its confidence >= confidence_exit,
              return WITHOUT escalating (early exit).
            - Otherwise escalate ONCE to the most capable (highest
              max_complexity) model, budget permitting; mark escalated=True in
              the ledger entries.
            - Return {"answer": ..., "model": <final model>, "cost": <total
              cost of this task>}.
        """
        raise NotImplementedError

    def analytics(self) -> dict:
        """{"total_cost": float, "calls": int, "by_model": {name: cost},
        "escalation_rate": fraction of tasks that escalated (0.0 if none)}."""
        raise NotImplementedError
