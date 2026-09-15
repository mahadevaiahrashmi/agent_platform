"""Project 5 — Human-in-the-Loop Approval Agent.
Uncertainty detection -> pause -> request human input -> resume with validated
context, with a complete, ordered audit trail.

The LLM replies with JSON: {"answer": "...", "confidence": 0.0-1.0,
"action": "<name>"|null}.
"""
from __future__ import annotations

import itertools
import json
import time
from typing import Callable, Dict, List, Optional, Set

class InMemoryTicketStore:
    def __init__(self):
        self._data = {}
    def save(self, ticket_id, data):
        self._data[ticket_id] = dict(data)
    def load(self, ticket_id):
        return self._data.get(ticket_id)
    def delete(self, ticket_id):
        self._data.pop(ticket_id, None)



class UnknownTicket(Exception):
    pass


class TicketExpired(Exception):
    pass


class ApprovalAgent:
    def __init__(
        self,
        llm,
        confidence_threshold: float = 0.75,
        protected_actions: set[str] = frozenset({"delete", "send_email", "refund"}),
        *,
        ticket_ttl_seconds: Optional[float] = None,
        required_approvals: int = 1,
        store=None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """Must set up:
            - self.audit_log: append-only list of events, each
              {"event": str, "detail": dict} — events in the order they happen.
            - internal storage for paused tickets.
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.protected_actions: Set[str] = set(protected_actions)
        self.audit_log: List[dict] = []
        self._tickets: Dict[str, dict] = {}
        self._id_gen = itertools.count(1)
        self.ticket_ttl_seconds = ticket_ttl_seconds
        self.required_approvals = max(1, required_approvals)
        self.store = store or InMemoryTicketStore()
        self._clock = clock or time.monotonic

    def handle(self, request: str) -> dict:
        """Process a request.
Requirements:
    - Call the LLM once, parse its JSON.
    - Log {"event": "request", ...} first, always.
    - AUTO path: confidence >= threshold AND action not protected ->
      log "completed" and return
      {"status": "completed", "answer": ..., "ticket": None}.
    - PAUSE path: low confidence OR protected action -> create a
      ticket id, log "paused" with a "reason" of "low_confidence" or
      "protected_action", store the pending answer/action, and return
      {"status": "pending", "ticket": <id>, "reason": ...}.
      The protected action MUST NOT be considered executed."""
        self.audit_log.append({"event": "request", "detail": {"request": request}})
        raw = self.llm.complete(request)
        data = json.loads(raw)
        answer = data.get("answer")
        confidence = float(data.get("confidence", 0.0))
        action = data.get("action")
        is_protected = action is not None and action in self.protected_actions
        low_conf = confidence < self.confidence_threshold
        if not low_conf and not is_protected:
            self.audit_log.append({"event": "completed", "detail": {"answer": answer}})
            return {"status": "completed", "answer": answer, "ticket": None}
        ticket_id = f"ticket-{next(self._id_gen)}"
        reason = "protected_action" if is_protected else "low_confidence"
        record = {
            "answer": answer, "action": action, "resolved": False,
            "approvals": 0, "rejections": 0, "created_at": self._clock(), "reason": reason,
        }
        self._tickets[ticket_id] = record
        self.store.save(ticket_id, record)
        self.audit_log.append({"event": "paused", "detail": {"ticket": ticket_id, "reason": reason}})
        return {"status": "pending", "ticket": ticket_id, "reason": reason}

    def resume(self, ticket: str, approved: bool, human_note: str = "") -> dict:
        """Resume a paused ticket with the human decision.
Requirements:
    - Unknown/already-resolved ticket -> raise UnknownTicket.
    - Log "human_decision" (with approved + note), then:
      approved -> log "completed", return {"status": "completed",
      "answer": <pending answer>}.
      rejected -> log "aborted", return {"status": "aborted",
      "answer": None}.
    - A ticket can be resumed exactly once."""
        pending = self._tickets.get(ticket) or self.store.load(ticket)
        if pending is None or pending.get("resolved"):
            raise UnknownTicket(f"Unknown or already resolved ticket: {ticket}")
        if self.ticket_ttl_seconds is not None:
            age = self._clock() - float(pending.get("created_at", self._clock()))
            if age > self.ticket_ttl_seconds:
                pending["resolved"] = True
                self._tickets[ticket] = pending
                self.store.save(ticket, pending)
                self.audit_log.append({"event": "expired", "detail": {"ticket": ticket}})
                raise TicketExpired(f"Ticket {ticket} expired")
        self.audit_log.append({
            "event": "human_decision",
            "detail": {"ticket": ticket, "approved": approved, "note": human_note},
        })
        if approved:
            pending["approvals"] = int(pending.get("approvals", 0)) + 1
        else:
            pending["rejections"] = int(pending.get("rejections", 0)) + 1
            pending["resolved"] = True
            self._tickets[ticket] = pending
            self.store.save(ticket, pending)
            self.audit_log.append({"event": "aborted", "detail": {"ticket": ticket}})
            return {"status": "aborted", "answer": None}
        if pending["approvals"] >= self.required_approvals:
            pending["resolved"] = True
            self._tickets[ticket] = pending
            self.store.save(ticket, pending)
            self.audit_log.append({
                "event": "completed",
                "detail": {"ticket": ticket, "answer": pending["answer"]},
            })
            return {"status": "completed", "answer": pending["answer"]}
        self._tickets[ticket] = pending
        self.store.save(ticket, pending)
        return {
            "status": "pending", "ticket": ticket, "reason": "awaiting_more_approvals",
            "approvals": pending["approvals"], "required": self.required_approvals,
        }

    def audit_trail(self, ticket: str | None = None) -> list[dict]:
        """Full audit log, or only events whose detail carries this ticket."""
        if ticket is None:
            return list(self.audit_log)
        return [e for e in self.audit_log if e.get("detail", {}).get("ticket") == ticket]
