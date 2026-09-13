"""Bounded ReAct observe-think-act cycle over a supplied tool set.

Every decision sees the goal and prior trace; final decisions stop immediately;
tool results feed the next iteration; unknown tools, tool exceptions, and
invalid decisions become error observations; the LLM call count never exceeds
max_iterations; exhaustion returns a non-empty best-effort answer rather than
raising.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Observation:
    content: str
    is_error: bool = False


@dataclass
class Decision:
    kind: str  # "tool" | "final"
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    answer: Optional[str] = None
    raw: str = ""


@dataclass
class ReActResult:
    answer: str
    trace: List[dict] = field(default_factory=list)
    iterations: int = 0
    llm_calls: int = 0


def _parse_decision(text: str) -> Decision:
    """Parse LLM output into a Decision.

    Expected formats (flexible):
      FINAL: <answer>
      TOOL: <name> ARGS: <json>
      or free-form JSON {"action": "final"|"tool", ...}
    """
    text = text.strip()
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            action = data.get("action") or data.get("kind") or data.get("type")
            if action in ("final", "answer", "done"):
                return Decision(
                    kind="final",
                    answer=str(data.get("answer") or data.get("content") or data.get("result") or ""),
                    raw=text,
                )
            if action in ("tool", "call", "action"):
                return Decision(
                    kind="tool",
                    tool_name=str(data.get("tool") or data.get("name") or data.get("tool_name") or ""),
                    tool_args=data.get("args") or data.get("arguments") or data.get("parameters") or {},
                    raw=text,
                )
    except (json.JSONDecodeError, TypeError):
        pass

    # FINAL: pattern
    m = re.match(r"(?i)^\s*FINAL\s*[:\-]\s*(.*)$", text, re.DOTALL)
    if m:
        return Decision(kind="final", answer=m.group(1).strip(), raw=text)

    # TOOL: name ARGS: {...}
    m = re.match(
        r"(?i)^\s*TOOL\s*[:\-]\s*(\w+)\s*(?:ARGS?\s*[:\-]\s*)?(.*)$",
        text,
        re.DOTALL,
    )
    if m:
        name = m.group(1).strip()
        args_str = m.group(2).strip()
        args: dict = {}
        if args_str:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"input": args_str}
        return Decision(kind="tool", tool_name=name, tool_args=args, raw=text)

    # Fallback: treat as final answer
    return Decision(kind="final", answer=text, raw=text)


def run_react(
    llm: Any,
    goal: str,
    tools: Dict[str, Callable[..., Any]],
    max_iterations: int = 5,
) -> ReActResult:
    """Run a bounded ReAct loop.

    The LLM is called at most ``max_iterations`` times.  On exhaustion a
    non-empty best-effort answer is returned.
    """
    trace: List[dict] = []
    observations: List[Observation] = []
    llm_calls = 0
    best_answer = "I was unable to fully solve the goal within the iteration limit."

    for i in range(max_iterations):
        # Build prompt with goal + prior trace
        prompt_parts = [
            f"Goal: {goal}",
            "",
            "You may either call a tool or give a final answer.",
            "To call a tool respond with: TOOL: <name> ARGS: <json>",
            "To finish respond with: FINAL: <answer>",
            "",
            "Available tools:",
        ]
        for name, fn in tools.items():
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            prompt_parts.append(f"  - {name}: {doc}")
        if observations:
            prompt_parts.append("")
            prompt_parts.append("Prior observations:")
            for idx, obs in enumerate(observations, 1):
                tag = "ERROR" if obs.is_error else "OK"
                prompt_parts.append(f"  [{idx}] ({tag}) {obs.content}")
        prompt_parts.append("")
        prompt_parts.append("Your decision:")
        prompt = "\n".join(prompt_parts)

        try:
            raw = llm.complete(prompt)
            llm_calls += 1
        except Exception as exc:
            observations.append(Observation(content=f"LLM error: {exc}", is_error=True))
            trace.append({"iteration": i + 1, "error": str(exc)})
            continue

        decision = _parse_decision(raw)
        entry = {
            "iteration": i + 1,
            "raw": raw,
            "kind": decision.kind,
            "tool_name": decision.tool_name,
            "answer": decision.answer,
        }
        trace.append(entry)

        if decision.kind == "final":
            answer = (decision.answer or "").strip() or best_answer
            return ReActResult(answer=answer, trace=trace, iterations=i + 1, llm_calls=llm_calls)

        # Tool call
        name = decision.tool_name or ""
        if name not in tools:
            obs = Observation(content=f"Unknown tool: {name}", is_error=True)
            observations.append(obs)
            continue

        try:
            result = tools[name](**(decision.tool_args or {}))
            obs = Observation(content=str(result), is_error=False)
            observations.append(obs)
            # Keep a best-effort answer from successful tool results
            if result is not None:
                best_answer = str(result)
        except Exception as exc:
            obs = Observation(content=f"Tool {name} raised: {exc}", is_error=True)
            observations.append(obs)

    # Exhausted iterations – return best effort
    return ReActResult(
        answer=best_answer,
        trace=trace,
        iterations=max_iterations,
        llm_calls=llm_calls,
    )
