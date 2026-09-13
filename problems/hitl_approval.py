"""Project 5 — Human-in-the-Loop Approval Agent.
Uncertainty detection -> pause -> request human input -> resume with validated
context, with a complete, ordered audit trail.

The LLM replies with JSON: {"answer": "...", "confidence": 0.0-1.0,
"action": "<name>"|null}.
"""


class UnknownTicket(Exception):
    pass


class ApprovalAgent:
    def __init__(self, llm, confidence_threshold: float = 0.75,
                 protected_actions: set[str] = frozenset({"delete", "send_email", "refund"})):
        """Must set up:
            - self.audit_log: append-only list of events, each
              {"event": str, "detail": dict} — events in the order they happen.
            - internal storage for paused tickets.
        """
        raise NotImplementedError

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
              The protected action MUST NOT be considered executed.
        """
        raise NotImplementedError

    def resume(self, ticket: str, approved: bool, human_note: str = "") -> dict:
        """Resume a paused ticket with the human decision.
        Requirements:
            - Unknown/already-resolved ticket -> raise UnknownTicket.
            - Log "human_decision" (with approved + note), then:
              approved -> log "completed", return {"status": "completed",
              "answer": <pending answer>}.
              rejected -> log "aborted", return {"status": "aborted",
              "answer": None}.
            - A ticket can be resumed exactly once.
        """
        raise NotImplementedError

    def audit_trail(self, ticket: str | None = None) -> list[dict]:
        """Full audit log, or only events whose detail carries this ticket."""
        raise NotImplementedError
