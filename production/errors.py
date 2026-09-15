"""Structured error taxonomy for tools and agent runs."""

from __future__ import annotations


class AgentError(Exception):
    code: str = "agent_error"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code:
            self.code = code


class ToolTimeoutError(AgentError):
    code = "tool_timeout"


class ToolPermissionError(AgentError):
    code = "tool_permission"


class ToolValidationError(AgentError):
    code = "tool_validation"


class ToolRuntimeError(AgentError):
    code = "tool_runtime"


class CircuitOpenError(AgentError):
    code = "circuit_open"


class BudgetExceededError(AgentError):
    code = "budget_exceeded"


def classify_tool_exception(exc: BaseException) -> AgentError:
    if isinstance(exc, AgentError):
        return exc
    if isinstance(exc, TimeoutError):
        return ToolTimeoutError(str(exc))
    if isinstance(exc, PermissionError):
        return ToolPermissionError(str(exc))
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ToolValidationError(str(exc))
    return ToolRuntimeError(str(exc))
