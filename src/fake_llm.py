"""Deterministic fake LLM used by all agent-project grading suites.

Scoring never calls a real model: tests inject scripted responses, so grading
is deterministic, instant, and free. Candidates code against this interface.
"""


class FakeLLM:
    """Scripted LLM client. Pops one response per complete() call.

    responses: list of str (returned as-is) or Exception (raised) —
    lets tests simulate malformed JSON, transient failures, etc.
    """

    def __init__(self, responses, name="fake-small", cost_per_1k=(0.15, 0.60)):
        self._responses = list(responses)
        self.name = name
        self.cost_in_per_1k, self.cost_out_per_1k = cost_per_1k
        self.calls = []  # [{"prompt": ..., "response": ...|None, "error": ...|None}]

    def complete(self, prompt: str) -> str:
        if not self._responses:
            raise RuntimeError("FakeLLM exhausted its scripted responses")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            self.calls.append({"prompt": prompt, "response": None, "error": repr(resp)})
            raise resp
        self.calls.append({"prompt": prompt, "response": resp, "error": None})
        return resp

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def token_usage(self) -> tuple[int, int]:
        """Crude deterministic token count: whitespace-split words."""
        tokens_in = sum(len(c["prompt"].split()) for c in self.calls)
        tokens_out = sum(len((c["response"] or "").split()) for c in self.calls)
        return tokens_in, tokens_out

    def cost(self) -> float:
        tokens_in, tokens_out = self.token_usage()
        return (tokens_in / 1000) * self.cost_in_per_1k + (tokens_out / 1000) * self.cost_out_per_1k
