"""Project 1 — Structured Output (production, stack-independent)."""
from __future__ import annotations
import json, time
from typing import Optional
from pydantic import BaseModel, ValidationError
from errors import CircuitOpenError
from interfaces import MetricsHook, NullMetrics

class ExtractionError(Exception):
    """Raised when the LLM cannot produce schema-valid output within retries."""

class StructuredAgent:
    def __init__(self, llm, schema: type[BaseModel], max_retries: int = 2, *,
                 circuit_failure_threshold: int = 0, metrics: Optional[MetricsHook] = None,
                 schema_version: str = "1"):
        self.llm = llm
        self.schema = schema
        self.max_retries = max_retries
        self.failures: list[dict] = []
        self.circuit_failure_threshold = circuit_failure_threshold
        self._consecutive_failures = 0
        self.metrics: MetricsHook = metrics or NullMetrics()
        self.schema_version = schema_version

    def build_prompt(self, text: str, previous_error: str | None = None) -> str:
        schema_json = json.dumps(self.schema.model_json_schema(), indent=2)
        parts = [
            "Extract structured data matching the following JSON schema.",
            "Return ONLY valid JSON that conforms to the schema. No markdown, no commentary.",
            "", "Schema:", schema_json, "", "Source text:", text,
        ]
        if previous_error is not None:
            parts.extend(["", f"Previous error: {previous_error}",
                          "Please correct the output based on the error above."])
        return "\n".join(parts)

    def _validate_response(self, raw: str) -> BaseModel:
        try:
            return self.schema.model_validate_json(raw)
        except Exception:
            data = json.loads(raw)
            return self.schema.model_validate(data)

    def extract(self, text: str) -> BaseModel:
        if (self.circuit_failure_threshold > 0 and
                self._consecutive_failures >= self.circuit_failure_threshold):
            raise CircuitOpenError(
                f"circuit open after {self._consecutive_failures} consecutive failures")
        previous_error = None
        total_attempts = self.max_retries + 1
        t0 = time.monotonic()
        for attempt in range(1, total_attempts + 1):
            prompt = self.build_prompt(text, previous_error)
            raw = self.llm.complete(prompt)
            try:
                obj = self._validate_response(raw)
                self._consecutive_failures = 0
                self.metrics.record("structured_output.success", 1.0, attempts=attempt,
                                    schema=self.schema.__name__)
                self.metrics.record("structured_output.latency_ms", (time.monotonic() - t0) * 1000)
                try:
                    object.__setattr__(obj, "__schema_version__", self.schema_version)
                except Exception:
                    pass
                return obj
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                error_str = f"{type(exc).__name__}: {exc}"
                self.failures.append({"attempt": attempt, "raw": raw, "error": error_str})
                previous_error = error_str
                self.metrics.record("structured_output.retry", 1.0, attempt=attempt,
                                    error_type=type(exc).__name__)
                if attempt == total_attempts:
                    self._consecutive_failures += 1
                    self.metrics.record("structured_output.failure", 1.0)
                    raise ExtractionError(f"Failed after {total_attempts} attempts") from exc
        raise ExtractionError("Unexpected fall-through")

    def reset_circuit(self) -> None:
        self._consecutive_failures = 0
