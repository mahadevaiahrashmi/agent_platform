import pytest
from pydantic import BaseModel
from fake_llm import FakeLLM
from structured_output import StructuredAgent, ExtractionError


class Invoice(BaseModel):
    number: str
    total: float
    paid: bool


VALID = '{"number": "INV-1", "total": 99.5, "paid": false}'
BAD_JSON = 'Sure! Here is the invoice: {"number": "INV-1", total: '
WRONG_TYPE = '{"number": "INV-1", "total": "a lot", "paid": false}'


def test_valid_first_try():
    llm = FakeLLM([VALID])
    agent = StructuredAgent(llm, Invoice)
    result = agent.extract("Invoice INV-1 for $99.50, unpaid")
    assert isinstance(result, Invoice)
    assert result.total == 99.5
    assert llm.call_count == 1
    assert agent.failures == []


def test_prompt_contains_schema_and_text():
    llm = FakeLLM([VALID])
    agent = StructuredAgent(llm, Invoice)
    agent.extract("Invoice INV-1 for $99.50, unpaid")
    prompt = llm.calls[0]["prompt"]
    assert "number" in prompt and "total" in prompt and "paid" in prompt
    assert "Invoice INV-1" in prompt


def test_retry_on_bad_json_feeds_error_back():
    llm = FakeLLM([BAD_JSON, VALID])
    agent = StructuredAgent(llm, Invoice)
    result = agent.extract("Invoice INV-1 for $99.50, unpaid")
    assert result.number == "INV-1"
    assert llm.call_count == 2
    assert len(agent.failures) == 1
    retry_prompt = llm.calls[1]["prompt"]
    assert agent.failures[0]["error"].split(":")[0].lower() in retry_prompt.lower() \
        or "error" in retry_prompt.lower()


def test_retry_on_type_error():
    llm = FakeLLM([WRONG_TYPE, VALID])
    agent = StructuredAgent(llm, Invoice)
    result = agent.extract("Invoice INV-1 for $99.50, unpaid")
    assert result.total == 99.5
    assert len(agent.failures) == 1


def test_exhausted_retries_raises_and_logs():
    llm = FakeLLM([BAD_JSON, BAD_JSON, WRONG_TYPE])
    agent = StructuredAgent(llm, Invoice, max_retries=2)
    with pytest.raises(ExtractionError):
        agent.extract("Invoice INV-1")
    assert llm.call_count == 3
    assert len(agent.failures) == 3
    assert all({"attempt", "raw", "error"} <= set(f) for f in agent.failures)
