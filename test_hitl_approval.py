import json
import pytest
from fake_llm import FakeLLM
from hitl_approval import ApprovalAgent, UnknownTicket

CONFIDENT = json.dumps({"answer": "Paris", "confidence": 0.95, "action": None})
UNSURE = json.dumps({"answer": "maybe Lyon?", "confidence": 0.4, "action": None})
PROTECTED = json.dumps({"answer": "refund issued", "confidence": 0.99, "action": "refund"})


def test_confident_request_auto_completes():
    agent = ApprovalAgent(FakeLLM([CONFIDENT]))
    result = agent.handle("capital of France?")
    assert result["status"] == "completed"
    assert result["answer"] == "Paris"
    events = [e["event"] for e in agent.audit_log]
    assert events == ["request", "completed"]


def test_low_confidence_pauses():
    agent = ApprovalAgent(FakeLLM([UNSURE]))
    result = agent.handle("capital of gaul?")
    assert result["status"] == "pending"
    assert result["reason"] == "low_confidence"
    assert result["ticket"]


def test_protected_action_pauses_even_when_confident():
    agent = ApprovalAgent(FakeLLM([PROTECTED]))
    result = agent.handle("refund order 42")
    assert result["status"] == "pending"
    assert result["reason"] == "protected_action"


def test_resume_approved_completes_with_pending_answer():
    agent = ApprovalAgent(FakeLLM([PROTECTED]))
    ticket = agent.handle("refund order 42")["ticket"]
    result = agent.resume(ticket, approved=True, human_note="verified receipt")
    assert result == {"status": "completed", "answer": "refund issued"}
    events = [e["event"] for e in agent.audit_log]
    assert events == ["request", "paused", "human_decision", "completed"]


def test_resume_rejected_aborts():
    agent = ApprovalAgent(FakeLLM([UNSURE]))
    ticket = agent.handle("capital?")["ticket"]
    result = agent.resume(ticket, approved=False, human_note="too vague")
    assert result == {"status": "aborted", "answer": None}
    assert [e["event"] for e in agent.audit_log][-1] == "aborted"


def test_ticket_single_use_and_unknown():
    agent = ApprovalAgent(FakeLLM([UNSURE]))
    ticket = agent.handle("q")["ticket"]
    agent.resume(ticket, approved=True)
    with pytest.raises(UnknownTicket):
        agent.resume(ticket, approved=True)
    with pytest.raises(UnknownTicket):
        agent.resume("no-such-ticket", approved=True)


def test_audit_trail_filter_and_note():
    agent = ApprovalAgent(FakeLLM([UNSURE, CONFIDENT]))
    ticket = agent.handle("q1")["ticket"]
    agent.handle("q2")
    agent.resume(ticket, approved=True, human_note="checked")
    trail = agent.audit_trail(ticket)
    assert [e["event"] for e in trail] == ["paused", "human_decision", "completed"]
    decision = trail[1]["detail"]
    assert decision["approved"] is True and decision["note"] == "checked"
