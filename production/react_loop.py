"""Project 2 — ReAct (production, stack-independent)."""
from __future__ import annotations
import json, time, uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional
from config import AgentConfig, CancellationToken, CancelledError, Deadline, RunState
from errors import classify_tool_exception

def _run_with_timeout(fn: Callable, kwargs: dict, timeout: Optional[float]) -> Any:
    if timeout is None or timeout <= 0:
        return fn(**kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(f"tool timed out after {timeout}s") from exc

class ReActAgent:
    def __init__(self, llm, tools: dict, max_iterations: int = 5,
                 tool_timeout: Optional[float] = None, overall_timeout: Optional[float] = None,
                 *, config: Optional[AgentConfig] = None,
                 cancel_token: Optional[CancellationToken] = None,
                 clock: Optional[Callable[[], float]] = None,
                 idempotency_key: Optional[str] = None):
        self.llm = llm
        self.tools: Dict[str, Callable[..., Any]] = tools
        self.config = config or AgentConfig(
            max_iterations=max_iterations, tool_timeout=tool_timeout,
            overall_timeout=overall_timeout)
        if config is None:
            self.max_iterations = max_iterations
            self.tool_timeout = tool_timeout
            self.overall_timeout = overall_timeout
        else:
            self.max_iterations = self.config.max_iterations
            self.tool_timeout = self.config.tool_timeout
            self.overall_timeout = self.config.overall_timeout
        self.cancel_token = cancel_token or CancellationToken()
        self._clock = clock or time.monotonic
        self.idempotency_key = idempotency_key
        self.trace: List[dict] = []
        self.last_run: Optional[RunState] = None

    def run(self, goal: str) -> dict:
        run_id = str(uuid.uuid4())
        deadline = Deadline(self.overall_timeout, self._clock)
        self.trace = []
        state = RunState(run_id=run_id, goal=goal, status="running")
        try:
            for i in range(1, self.max_iterations + 1):
                self.cancel_token.raise_if_cancelled()
                deadline.raise_if_expired()
                prompt = self._build_prompt(goal)
                raw = self.llm.complete(prompt)
                thought, action, observation, final_answer = "", None, None, None
                try:
                    data = json.loads(raw)
                    thought = str(data.get("thought", ""))
                    if "final" in data:
                        final_answer = str(data["final"])
                        if self.config.max_output_chars and final_answer:
                            final_answer = final_answer[: self.config.max_output_chars]
                    elif "action" in data:
                        action = str(data["action"])
                        args = data.get("args") or {}
                        if not isinstance(args, dict):
                            args = {}
                        observation = self._invoke_tool(action, args)
                    else:
                        observation = "ERROR: invalid decision format"
                except (json.JSONDecodeError, TypeError, ValueError):
                    observation = "ERROR: invalid decision format"
                except (CancelledError, TimeoutError) as exc:
                    state.status = "cancelled" if isinstance(exc, CancelledError) else "timed_out"
                    state.iterations = i
                    state.trace = list(self.trace)
                    state.answer = self._best_effort_answer()
                    self.last_run = state
                    return {"status": state.status, "answer": state.answer,
                            "iterations": i, "run_id": run_id}
                step = {"thought": thought, "action": action, "observation": observation}
                self.trace.append(step)
                state.iterations = i
                state.trace = list(self.trace)
                if final_answer is not None:
                    state.status = "done"
                    state.answer = final_answer
                    self.last_run = state
                    return {"status": "done", "answer": final_answer, "iterations": i, "run_id": run_id}
            answer = self._best_effort_answer()
            state.status = "max_iterations"
            state.answer = answer
            self.last_run = state
            return {"status": "max_iterations", "answer": answer,
                    "iterations": self.max_iterations, "run_id": run_id}
        except CancelledError:
            state.status = "cancelled"
            state.answer = self._best_effort_answer()
            self.last_run = state
            return {"status": "cancelled", "answer": state.answer,
                    "iterations": state.iterations, "run_id": run_id}
        except TimeoutError:
            state.status = "timed_out"
            state.answer = self._best_effort_answer()
            self.last_run = state
            return {"status": "timed_out", "answer": state.answer,
                    "iterations": state.iterations, "run_id": run_id}

    def _tool_timeout_for(self, name: str) -> Optional[float]:
        if name in self.config.per_tool_timeouts:
            return self.config.per_tool_timeouts[name]
        return self.tool_timeout

    def _invoke_tool(self, action: str, args: dict) -> str:
        if self.config.tool_denylist and action in self.config.tool_denylist:
            return f"ERROR: tool '{action}' denied"
        if self.config.tool_allowlist is not None and action not in self.config.tool_allowlist:
            return f"ERROR: tool '{action}' not allowed"
        if action not in self.tools:
            return f"ERROR: unknown tool '{action}'"
        call_args = dict(args)
        if self.idempotency_key and "idempotency_key" not in call_args:
            call_args["idempotency_key"] = self.idempotency_key
        try:
            result = _run_with_timeout(self.tools[action], call_args, self._tool_timeout_for(action))
            text = str(result)
            if self.config.max_output_chars:
                text = text[: self.config.max_output_chars]
            return text
        except TypeError:
            try:
                result = _run_with_timeout(self.tools[action], args, self._tool_timeout_for(action))
                text = str(result)
                if self.config.max_output_chars:
                    text = text[: self.config.max_output_chars]
                return text
            except Exception as exc:
                return f"ERROR: {classify_tool_exception(exc)}"
        except Exception as exc:
            return f"ERROR: {classify_tool_exception(exc)}"

    def _build_prompt(self, goal: str) -> str:
        parts = [f"Goal: {goal}", "", "Reply with JSON only:",
                 '  {"thought": "...", "action": "<tool>", "args": {...}}  to act',
                 '  {"thought": "...", "final": "<answer>"}                 to finish',
                 "", "Available tools: " + ", ".join(sorted(self.tools.keys()))]
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
        observations = [s["observation"] for s in self.trace
                        if s.get("observation") and not str(s["observation"]).startswith("ERROR")]
        if observations:
            return "Based on observations: " + " | ".join(observations)
        thoughts = [s["thought"] for s in self.trace if s.get("thought")]
        if thoughts:
            return "Partial progress: " + " | ".join(thoughts)
        return "Unable to complete the goal within the iteration limit."

    def checkpoint(self) -> Optional[RunState]:
        return self.last_run
