"""Gate low-confidence or protected actions behind resumable approval tickets.

A request event is always logged first; confident, unprotected work completes
automatically; low confidence or a protected action pauses with the specified
reason and does not execute the action; approval returns the pending answer,
rejection aborts, and each ticket resolves once; audit events remain ordered
and can be filtered by ticket, including the human note.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TicketStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO = "auto"


@dataclass
class AuditEvent:
    ticket_id: str
    event: str
    timestamp: float
    detail: Optional[str] = None
    human_note: Optional[str] = None


@dataclass
class Ticket:
    id: str
    action: str
    answer: Any
    reason: str
    status: TicketStatus = TicketStatus.PENDING
    confidence: float = 0.0
    protected: bool = False


class HITLApproval:
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        protected_actions: Optional[set] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.confidence_threshold = confidence_threshold
        self.protected_actions = protected_actions or set()
        self._clock = clock
        self._tickets: Dict[str, Ticket] = {}
        self._audit: List[AuditEvent] = []
        self._id_counter = itertools.count(1)

    def _log(self, ticket_id: str, event: str, detail: str | None = None, human_note: str | None = None) -> None:
        self._audit.append(
            AuditEvent(
                ticket_id=ticket_id,
                event=event,
                timestamp=self._clock(),
                detail=detail,
                human_note=human_note,
            )
        )

    def request(
        self,
        action: str,
        answer: Any,
        confidence: float,
        *,
        reason: str = "low_confidence",
    ) -> dict:
        """Submit work for possible human approval.

        Always logs a request event.  If confidence is high enough and the
        action is not protected, the work is auto-approved and executed
        (returned as completed).  Otherwise a pending ticket is created.
        """
        tid = f"t-{next(self._id_counter)}"
        protected = action in self.protected_actions
        self._log(tid, "request", detail=f"action={action} confidence={confidence}")

        needs_approval = confidence < self.confidence_threshold or protected
        if not needs_approval:
            ticket = Ticket(
                id=tid,
                action=action,
                answer=answer,
                reason="auto",
                status=TicketStatus.AUTO,
                confidence=confidence,
                protected=protected,
            )
            self._tickets[tid] = ticket
            self._log(tid, "auto_approved")
            return {
                "status": "completed",
                "ticket_id": tid,
                "answer": answer,
            }

        # Pause – do not execute
        actual_reason = reason
        if protected and confidence >= self.confidence_threshold:
            actual_reason = "protected_action"
        ticket = Ticket(
            id=tid,
            action=action,
            answer=answer,
            reason=actual_reason,
            status=TicketStatus.PENDING,
            confidence=confidence,
            protected=protected,
        )
        self._tickets[tid] = ticket
        self._log(tid, "paused", detail=actual_reason)
        return {
            "status": "pending",
            "ticket_id": tid,
            "reason": actual_reason,
            "answer": None,  # not executed yet
        }

    def approve(self, ticket_id: str, note: str = "") -> dict:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise KeyError(f"Unknown ticket: {ticket_id}")
        if ticket.status != TicketStatus.PENDING:
            raise RuntimeError(f"Ticket {ticket_id} already resolved as {ticket.status}")
        ticket.status = TicketStatus.APPROVED
        self._log(ticket_id, "approved", human_note=note or None)
        return {
            "status": "completed",
            "ticket_id": ticket_id,
            "answer": ticket.answer,
        }

    def reject(self, ticket_id: str, note: str = "") -> dict:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise KeyError(f"Unknown ticket: {ticket_id}")
        if ticket.status != TicketStatus.PENDING:
            raise RuntimeError(f"Ticket {ticket_id} already resolved as {ticket.status}")
        ticket.status = TicketStatus.REJECTED
        self._log(ticket_id, "rejected", human_note=note or None)
        return {
            "status": "aborted",
            "ticket_id": ticket_id,
            "answer": None,
        }

    def audit_log(self, ticket_id: Optional[str] = None) -> List[AuditEvent]:
        """Return audit events, optionally filtered by ticket_id, in order."""
        if ticket_id is None:
            return list(self._audit)
        return [e for e in self._audit if e.ticket_id == ticket_id]
