"""Project 1 — Structured Output Agent.
Make LLM output reliable: enforce a Pydantic schema, retry on parse/validation
errors feeding the error back to the model, and log every failure.
"""
from pydantic import BaseModel


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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
