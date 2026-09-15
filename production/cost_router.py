"""Project 6 — Cost Router (production, stack-independent)."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from interfaces import BudgetChecker, InMemoryBudget


class BudgetExceeded(Exception):
    pass


_COMPLEXITY_KEYWORDS = {
    "analyze", "compare", "architecture", "multi-step", "prove",
}


def estimate_complexity(task: str) -> int:
    words = task.lower().split()
    base = len(words)
    bonus = 0
    for kw in _COMPLEXITY_KEYWORDS:
        for w in words:
            if w == kw:
                bonus += 10
    return base + bonus


class CostRouter:
    def __init__(
        self,
        models: dict[str, dict],
        budget: float,
        confidence_exit: float = 0.8,
        *,
        budget_checker: Optional[BudgetChecker] = None,
        task_cost_ceiling: Optional[float] = None,
        fail_open: bool = False,
    ):
        self.models = models
        self.budget = budget
        self.confidence_exit = confidence_exit
        self.ledger: List[dict] = []
        self._budget: BudgetChecker = budget_checker or InMemoryBudget(budget)
        self._spent = 0.0
        self._task_count = 0
        self._escalated_tasks = 0
        self.task_cost_ceiling = task_cost_ceiling
        self.fail_open = fail_open

    def route(self, task: str) -> str:
        complexity = estimate_complexity(task)
        capable = [
            (name, cfg) for name, cfg in self.models.items()
            if cfg["max_complexity"] >= complexity
        ]
        if capable:
            capable.sort(key=lambda x: x[1]["cost_per_call"])
            return capable[0][0]
        return max(self.models.items(), key=lambda x: x[1]["max_complexity"])[0]

    def _charge(self, cost: float) -> None:
        if not self._budget.can_afford(cost):
            raise BudgetExceeded(
                f"Cannot afford cost {cost}; spent {self._budget.spent}, budget {self.budget}"
            )
        self._budget.charge(cost)
        self._spent = self._budget.spent

    def run_task(self, task: str) -> dict:
        self._task_count += 1
        model_name = self.route(task)
        cfg = self.models[model_name]
        cost = cfg["cost_per_call"]

        if self.task_cost_ceiling is not None and cost > self.task_cost_ceiling:
            raise BudgetExceeded(
                f"model {model_name} cost {cost} exceeds task ceiling {self.task_cost_ceiling}"
            )

        if self._spent + cost > self.budget or not self._budget.can_afford(cost):
            raise BudgetExceeded(
                f"Cannot afford {model_name} (cost {cost}); spent {self._spent}, budget {self.budget}"
            )

        try:
            raw = cfg["client"].complete(task)
            data = json.loads(raw)
            answer = data["answer"]
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            if not self.fail_open:
                raise
            for alt_name, alt_cfg in sorted(
                self.models.items(), key=lambda x: -x[1]["max_complexity"]
            ):
                if alt_name == model_name:
                    continue
                alt_cost = alt_cfg["cost_per_call"]
                if not self._budget.can_afford(alt_cost):
                    continue
                try:
                    raw = alt_cfg["client"].complete(task)
                    data = json.loads(raw)
                    answer = data["answer"]
                    self._charge(alt_cost)
                    self.ledger.append({
                        "task": task, "model": alt_name, "cost": alt_cost, "escalated": True,
                    })
                    self._escalated_tasks += 1
                    return {"answer": answer, "model": alt_name, "cost": alt_cost}
                except Exception:
                    continue
            raise

        self._charge(cost)
        self.ledger.append({
            "task": task, "model": model_name, "cost": cost, "escalated": False,
        })
        task_cost = cost
        final_model = model_name

        if confidence >= self.confidence_exit:
            return {"answer": answer, "model": final_model, "cost": task_cost}

        most_capable_name = max(
            self.models.items(), key=lambda x: x[1]["max_complexity"]
        )[0]
        if most_capable_name != model_name:
            esc_cfg = self.models[most_capable_name]
            esc_cost = esc_cfg["cost_per_call"]
            if self.task_cost_ceiling is not None and task_cost + esc_cost > self.task_cost_ceiling:
                return {"answer": answer, "model": final_model, "cost": task_cost}
            if self._budget.can_afford(esc_cost):
                raw2 = esc_cfg["client"].complete(task)
                data2 = json.loads(raw2)
                answer = data2["answer"]
                self._charge(esc_cost)
                task_cost += esc_cost
                final_model = most_capable_name
                self.ledger.append({
                    "task": task, "model": most_capable_name,
                    "cost": esc_cost, "escalated": True,
                })
                self._escalated_tasks += 1

        return {"answer": answer, "model": final_model, "cost": task_cost}

    def analytics(self) -> dict:
        by_model: Dict[str, float] = {}
        for entry in self.ledger:
            by_model[entry["model"]] = by_model.get(entry["model"], 0.0) + entry["cost"]
        rate = self._escalated_tasks / self._task_count if self._task_count else 0.0
        return {
            "total_cost": self._spent,
            "calls": len(self.ledger),
            "by_model": by_model,
            "escalation_rate": rate,
        }
