"""Project 2 — ReAct Planning Agent.

Observe -> think -> act loop with a hard iteration cap, unknown-tool recovery,
and graceful degradation instead of infinite looping.

The LLM is called with the running trace and must reply with JSON:
    {"thought": "...", "action": "<tool name>", "args": {...}}   -- act
    {"thought": "...", "final": "<answer>"}                       -- finish
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


class ReActAgent:
    def __init__(self, llm, tools: dict, max_iterations: int = 5):
        """tools maps name -> callable(**args) -> str.

        Must set up self.trace: list of step dicts, in order, each
        {"thought": str, "action": str | None, "observation": str | None}.
        """
        self.llm = llm
        self.tools: Dict[str, Callable[..., Any]] = tools
        self.max_iterations = max_iterations
        self.trace: List[dict] = []

    def run(self, goal: str) -> dict:
        """Run the ReAct loop for a goal.

        Requirements:
            - Each iteration: call the LLM with the goal plus all prior
              thoughts/observations, parse its JSON decision.
            - "final" decision -> return {"status": "done", "answer": final,
              "iterations": n}.
            - Action decision -> execute the tool, append the observation to
              the trace, continue.
            - Unknown tool or tool exception MUST NOT crash the loop: record
              an error observation ("ERROR: ...") and continue, letting the
              model recover.
            - Unparseable LLM output counts as an iteration with observation
              "ERROR: invalid decision format".
            - After max_iterations without "final": degrade gracefully -->
              return {"status": "max_iterations", "answer": <best-effort
              summary of the trace, non-empty>, "iterations": max_iterations}.
              Never loop past the cap; never raise.
        """
        self.trace = []

        for i in range(1, self.max_iterations + 1):
            prompt = self._build_prompt(goal)
            raw = self.llm.complete(prompt)

            thought = ""
            action: Optional[str] = None
            observation: Optional[str] = None
            final_answer: Optional[str] = None

            try:
                data = json.loads(raw)
                thought = str(data.get("thought", ""))
                if "final" in data:
                    final_answer = str(data["final"])
                elif "action" in data:
                    action = str(data["action"])
                    args = data.get("args") or {}
                    if not isinstance(args, dict):
                        args = {}
                    if action not in self.tools:
                        observation = f"ERROR: unknown tool '{action}'"
                    else:
                        try:
                            observation = str(self.tools[action](**args))
                        except Exception as exc:
                            observation = f"ERROR: {exc}"
                else:
                    observation = "ERROR: invalid decision format"
            except (json.JSONDecodeError, TypeError, ValueError):
                observation = "ERROR: invalid decision format"

            step = {
                "thought": thought,
                "action": action,
                "observation": observation,
            }
            self.trace.append(step)

            if final_answer is not None:
                return {
                    "status": "done",
                    "answer": final_answer,
                    "iterations": i,
                }

        # Exhausted
        answer = self._best_effort_answer()
        return {
            "status": "max_iterations",
            "answer": answer,
            "iterations": self.max_iterations,
        }

    def _build_prompt(self, goal: str) -> str:
        parts = [
            f"Goal: {goal}",
            "",
            "Reply with JSON only:",
            '  {"thought": "...", "action": "<tool>", "args": {...}}  to act',
            '  {"thought": "...", "final": "<answer>"}                 to finish',
            "",
            "Available tools: " + ", ".join(sorted(self.tools.keys())),
        ]
        if self.trace:
            parts.append("")
            parts.append("Prior steps:")
            for idx, step in enumerate(self.trace, 1):
                parts.append(f"  Step {idx}: thought={step['thought']!r}")
                if step["action"]:
                    parts.append(f"    action={step['action']}")
                if step["observation"] is not None:
                    parts.append(f"    observation={step['observation']}")
        parts.append("")
        parts.append("Your decision (JSON):")
        return "\n".join(parts)

    def _best_effort_answer(self) -> str:
        observations = [
            s["observation"]
            for s in self.trace
            if s.get("observation") and not str(s["observation"]).startswith("ERROR")
        ]
        if observations:
            return "Based on observations: " + " | ".join(observations)
        thoughts = [s["thought"] for s in self.trace if s.get("thought")]
        if thoughts:
            return "Partial progress: " + " | ".join(thoughts)
        return "Unable to complete the goal within the iteration limit."
