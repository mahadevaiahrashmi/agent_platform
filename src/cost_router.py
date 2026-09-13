"""Estimate task complexity, choose a model, enforce a hard budget, and report usage.

Complexity uses the exact word-count-plus-keyword heuristic; routing chooses the
cheapest capable model or the most capable fallback; budget is checked before
every call; a confident routed answer exits early; a low-confidence answer may
escalate once to the most capable model only if affordable; the final task
result and ledger support total, per-model, call-count, and task escalation
analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Keyword signals that increase complexity
COMPLEXITY_KEYWORDS = {
    "analyze",
    "compare",
    "reason",
    "multi-step",
    "complex",
    "synthesize",
    "evaluate",
    "critique",
    "plan",
    "design",
}


@dataclass
class ModelSpec:
    name: str
    cost_per_call: float
    capability: int  # higher = more capable
    max_complexity: int  # can handle tasks up to this complexity


@dataclass
class CallRecord:
    model: str
    cost: float
    prompt: str
    response: str
    escalated: bool = False


@dataclass
class TaskResult:
    answer: str
    model_used: str
    confidence: float
    escalated: bool
    total_cost: float
    ledger: List[CallRecord] = field(default_factory=list)


class CostRouter:
    def __init__(
        self,
        models: Sequence[ModelSpec],
        llm_factory: Any,
        budget: float,
        confidence_threshold: float = 0.7,
    ):
        """
        llm_factory(model_name) -> object with .complete(prompt) -> str
        and optionally .last_confidence -> float
        """
        self.models = sorted(models, key=lambda m: (m.cost_per_call, -m.capability))
        self.llm_factory = llm_factory
        self.budget = budget
        self.confidence_threshold = confidence_threshold
        self.spent = 0.0
        self.ledger: List[CallRecord] = []
        self._call_counts: Dict[str, int] = {}
        self._escalations = 0

    def estimate_complexity(self, task: str) -> int:
        """Exact heuristic: word count + keyword bonuses."""
        words = task.lower().split()
        score = len(words)
        for w in words:
            if w in COMPLEXITY_KEYWORDS:
                score += 5
        return score

    def choose_model(self, complexity: int) -> ModelSpec:
        """Cheapest model that can handle the complexity; else most capable."""
        capable = [m for m in self.models if m.max_complexity >= complexity]
        if capable:
            # already sorted by cost ascending
            return capable[0]
        # fallback: most capable
        return max(self.models, key=lambda m: m.capability)

    def _can_afford(self, model: ModelSpec) -> bool:
        return self.spent + model.cost_per_call <= self.budget

    def run(self, task: str) -> TaskResult:
        complexity = self.estimate_complexity(task)
        model = self.choose_model(complexity)
        escalated = False

        if not self._can_afford(model):
            # Try any cheaper model that we can still afford
            affordable = [m for m in self.models if self._can_afford(m)]
            if not affordable:
                return TaskResult(
                    answer="Budget exhausted before any model call.",
                    model_used="",
                    confidence=0.0,
                    escalated=False,
                    total_cost=self.spent,
                    ledger=list(self.ledger),
                )
            model = max(affordable, key=lambda m: m.capability)

        answer, confidence = self._call(model, task)

        # Early exit on high confidence
        if confidence >= self.confidence_threshold:
            return TaskResult(
                answer=answer,
                model_used=model.name,
                confidence=confidence,
                escalated=False,
                total_cost=self.spent,
                ledger=list(self.ledger),
            )

        # Optional single escalation to most capable if affordable
        most_capable = max(self.models, key=lambda m: m.capability)
        if (
            most_capable.name != model.name
            and self._can_afford(most_capable)
        ):
            escalated = True
            self._escalations += 1
            answer, confidence = self._call(most_capable, task, escalated=True)
            model = most_capable

        return TaskResult(
            answer=answer,
            model_used=model.name,
            confidence=confidence,
            escalated=escalated,
            total_cost=self.spent,
            ledger=list(self.ledger),
        )

    def _call(self, model: ModelSpec, prompt: str, escalated: bool = False) -> tuple[str, float]:
        if not self._can_afford(model):
            raise RuntimeError(f"Cannot afford model {model.name}")
        llm = self.llm_factory(model.name)
        response = llm.complete(prompt)
        conf = getattr(llm, "last_confidence", 0.5)
        cost = model.cost_per_call
        self.spent += cost
        self._call_counts[model.name] = self._call_counts.get(model.name, 0) + 1
        rec = CallRecord(
            model=model.name,
            cost=cost,
            prompt=prompt,
            response=response,
            escalated=escalated,
        )
        self.ledger.append(rec)
        return response, float(conf)

    # Analytics helpers
    @property
    def total_cost(self) -> float:
        return self.spent

    def calls_per_model(self) -> Dict[str, int]:
        return dict(self._call_counts)

    @property
    def escalation_count(self) -> int:
        return self._escalations
