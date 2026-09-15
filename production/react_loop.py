"""Project 2 — ReAct Planning Agent.
Observe -> think -> act loop with a hard iteration cap, unknown-tool recovery,
and graceful degradation instead of infinite looping.

The LLM is called with the running trace and must reply with JSON:
    {"thought": "...", "action": "<tool name>", "args": {...}}   -- act
    {"thought": "...", "final": "<answer>"}                       -- finish
"""
from __future__ import annotations
import json, time, uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional

class CancelledError(Exception):
    pass

class CancellationToken:
    def __init__(self):
        self._cancelled = False
        self.reason = None
    def cancel(self, reason: str = "cancelled"):
        self._cancelled = True
        self.reason = reason
    @property
    def is_cancelled(self):
        return self._cancelled
    def raise_if_cancelled(self):
        if self._cancelled:
            raise CancelledError(self.reason or "cancelled")

class Deadline:
    def __init__(self, seconds, clock):
        self._clock = clock
        self._deadline = None if seconds is None or seconds <= 0 else clock() + seconds
    def expired(self):
        return self._deadline is not None and self._clock() >= self._deadline
    def raise_if_expired(self):
        if self.expired():
            raise TimeoutError("overall deadline exceeded")

class AgentConfig:
    def __init__(self, max_iterations=5, tool_timeout=None, overall_timeout=None,
                 per_tool_timeouts=None, max_output_chars=None,
                 tool_allowlist=None, tool_denylist=None):
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.overall_timeout = overall_timeout
        self.per_tool_timeouts = per_tool_timeouts or {}
        self.max_output_chars = max_output_chars
        self.tool_allowlist = tool_allowlist
        self.tool_denylist = tool_denylist

class RunState:
    def __init__(self, run_id, goal, status, iterations=0, answer=None, trace=None, meta=None):
        self.run_id = run_id
        self.goal = goal
        self.status = status
        self.iterations = iterations
        self.answer = answer
        self.trace = trace or []
        self.meta = meta or {}

def classify_tool_exception(exc):
    return exc

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
        """tools maps name -> callable(**args) -> str.
Must set up self.trace: list of step dicts, in order, each
{"thought": str, "action": str | None, "observation": str | None}."""
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
      Never loop past the cap; never raise."""
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
