"""Produce a Pydantic-validated object from an LLM response.

Prompts contain the source text and JSON schema; JSON parsing and schema
validation are enforced; each failed attempt logs its number, raw response,
and error; the next prompt includes the previous error; max_retries + 1 total
attempts are allowed before ExtractionError.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ExtractionError(Exception):
    """Raised when structured extraction fails after all retries."""

    def __init__(self, message: str, attempts: int, last_raw: str | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw


def _schema_for_model(model: Type[BaseModel]) -> dict:
    return model.model_json_schema()


def extract_structured(
    llm: Any,
    source_text: str,
    model: Type[T],
    max_retries: int = 2,
) -> T:
    """Extract a validated Pydantic model from source_text via the LLM.

    The prompt always includes the source text and the JSON schema of `model`.
    On failure the next prompt includes the previous error.  Exactly
    ``max_retries + 1`` attempts are made before raising ExtractionError.
    """
    schema = _schema_for_model(model)
    schema_str = json.dumps(schema, indent=2)
    previous_error: str | None = None
    last_raw: str | None = None
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        prompt_parts = [
            "Extract structured data from the following source text.",
            "Return ONLY valid JSON that matches the schema exactly.",
            "",
            "Source text:",
            source_text,
            "",
            "JSON Schema:",
            schema_str,
        ]
        if previous_error is not None:
            prompt_parts.extend(
                [
                    "",
                    f"Previous attempt failed with error: {previous_error}",
                    "Please correct the output.",
                ]
            )
        prompt = "\n".join(prompt_parts)

        try:
            raw = llm.complete(prompt)
            last_raw = raw
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            data = json.loads(cleaned)
            result = model.model_validate(data)
            return result
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.info(
                "structured_output attempt %s failed: raw=%r error=%s",
                attempt,
                last_raw,
                error_msg,
            )
            previous_error = error_msg
            if attempt == total_attempts:
                raise ExtractionError(
                    f"Failed to extract valid {model.__name__} after {total_attempts} attempts",
                    attempts=total_attempts,
                    last_raw=last_raw,
                ) from exc

    # Unreachable, but satisfies type checkers
    raise ExtractionError("Unexpected fall-through", attempts=total_attempts, last_raw=last_raw)
