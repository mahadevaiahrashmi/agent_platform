"""Project 1 — Structured Output Agent.

Make LLM output reliable: enforce a Pydantic schema, retry on parse/validation
errors feeding the error back to the model, and log every failure.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, ValidationError


class ExtractionError(Exception):
    """Raised when the LLM cannot produce schema-valid output within retries."""


class StructuredAgent:
    def __init__(self, llm, schema: type[BaseModel], max_retries: int = 2):
        """Initialize with an LLM client, a Pydantic model class, and a retry cap.

        Must set up:
            - self.llm, self.schema, self.max_retries
            - self.failures: list of dicts logging every failed attempt
              ({"attempt": int, "raw": str, "error": str})
        """
        self.llm = llm
        self.schema = schema
        self.max_retries = max_retries
        self.failures: list[dict] = []

    def build_prompt(self, text: str, previous_error: str | None = None) -> str:
        """Build the extraction prompt.

        Requirements:
            - MUST include the schema's JSON structure (use
              self.schema.model_json_schema()) so the model knows the contract.
            - MUST include the source text.
            - When retrying, MUST include the previous validation error verbatim
              so the model can correct itself. Retries that don't feed the error
              back are scored as incorrect.
        """
        schema_json = json.dumps(self.schema.model_json_schema(), indent=2)
        parts = [
            "Extract structured data matching the following JSON schema.",
            "Return ONLY valid JSON that conforms to the schema.",
            "",
            "Schema:",
            schema_json,
            "",
            "Source text:",
            text,
        ]
        if previous_error is not None:
            parts.extend(
                [
                    "",
                    f"Previous error: {previous_error}",
                    "Please correct the output based on the error above.",
                ]
            )
        return "\n".join(parts)

    def extract(self, text: str) -> BaseModel:
        """Extract a validated instance of self.schema from text.

        Requirements:
            - Call the LLM, parse the response as JSON, validate with the schema
              (schema.model_validate_json or equivalent).
            - On parse/validation failure: log to self.failures and retry with
              the error fed back, up to self.max_retries retries
              (max_retries + 1 total attempts).
            - After exhausting retries, raise ExtractionError. All failures must
              remain logged in self.failures.
        """
        previous_error: Optional[str] = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            prompt = self.build_prompt(text, previous_error)
            raw = self.llm.complete(prompt)
            try:
                # Prefer model_validate_json when available
                try:
                    return self.schema.model_validate_json(raw)
                except Exception:
                    data = json.loads(raw)
                    return self.schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                error_str = f"{type(exc).__name__}: {exc}"
                self.failures.append(
                    {"attempt": attempt, "raw": raw, "error": error_str}
                )
                previous_error = error_str
                if attempt == total_attempts:
                    raise ExtractionError(
                        f"Failed after {total_attempts} attempts"
                    ) from exc

        raise ExtractionError("Unexpected fall-through")
