"""Deterministic FakeLLM for offline testing.

Returns scripted text or raises scripted exceptions, records calls.
Do not modify this file in the real challenge; provided here for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union


@dataclass
class LLMCall:
    prompt: str
    kwargs: dict


class FakeLLM:
    """Scripted LLM that returns predetermined responses in order."""

    def __init__(
        self,
        responses: Optional[Sequence[Union[str, Exception, Callable[[str], str]]]] = None,
        default: str = "OK",
    ):
        self.responses = list(responses or [])
        self.default = default
        self.calls: List[LLMCall] = []
        self._index = 0

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(LLMCall(prompt=prompt, kwargs=kwargs))
        if self._index < len(self.responses):
            resp = self.responses[self._index]
            self._index += 1
            if isinstance(resp, Exception):
                raise resp
            if callable(resp):
                return resp(prompt)
            return str(resp)
        return self.default

    def reset(self) -> None:
        self.calls.clear()
        self._index = 0

    @property
    def call_count(self) -> int:
        return len(self.calls)
