"""Project 2 — ReAct Planning Agent.
Observe -> think -> act loop with a hard iteration cap, unknown-tool recovery,
and graceful degradation instead of infinite looping.

The LLM is called with the running trace and must reply with JSON:
    {"thought": "...", "action": "<tool name>", "args": {...}}   -- act
    {"thought": "...", "final": "<answer>"}                       -- finish
"""


class ReActAgent:
    def __init__(self, llm, tools: dict, max_iterations: int = 5):
        """tools maps name -> callable(**args) -> str.
        Must set up self.trace: list of step dicts, in order, each
        {"thought": str, "action": str | None, "observation": str | None}.
        """
        raise NotImplementedError

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
        raise NotImplementedError
