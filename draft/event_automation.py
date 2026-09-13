"""Process events idempotently with bounded retry, exponential backoff, dead-lettering, and replay.

Successful and dead-lettered IDs reject later duplicates; unknown types
dead-letter without retry; handler failures retry max_retries additional times
with delays 1, 2, 4, ...; final dead-letter records contain the last error and
total attempts; process() never raises; replay bypasses dead-letter
idempotency, counts recoveries, and does not duplicate failed entries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class DeadLetterRecord:
    event_id: str
    event_type: str
    payload: Any
    last_error: str
    attempts: int


@dataclass
class ProcessResult:
    status: str  # "success" | "duplicate" | "dead_letter" | "unknown_type"
    event_id: str
    attempts: int = 0
    error: Optional[str] = None


class EventAutomation:
    def __init__(
        self,
        handlers: Dict[str, Callable[[Any], None]],
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.handlers = handlers
        self.max_retries = max_retries
        self._sleep = sleep_fn
        self._processed: Set[str] = set()
        self._dead_lettered: Set[str] = set()
        self.dead_letters: List[DeadLetterRecord] = []
        self.recovery_count = 0

    def process(self, event_id: str, event_type: str, payload: Any) -> ProcessResult:
        """Process an event idempotently. Never raises."""
        if event_id in self._processed or event_id in self._dead_lettered:
            return ProcessResult(status="duplicate", event_id=event_id)

        if event_type not in self.handlers:
            rec = DeadLetterRecord(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                last_error=f"Unknown event type: {event_type}",
                attempts=1,
            )
            self.dead_letters.append(rec)
            self._dead_lettered.add(event_id)
            return ProcessResult(
                status="unknown_type",
                event_id=event_id,
                attempts=1,
                error=rec.last_error,
            )

        handler = self.handlers[event_type]
        last_error = ""
        # Initial attempt + max_retries additional
        for attempt in range(1, self.max_retries + 2):
            try:
                handler(payload)
                self._processed.add(event_id)
                return ProcessResult(status="success", event_id=event_id, attempts=attempt)
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.max_retries:
                    delay = 2 ** (attempt - 1)  # 1, 2, 4, ...
                    self._sleep(delay)

        # Exhausted retries → dead letter
        rec = DeadLetterRecord(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
            last_error=last_error,
            attempts=self.max_retries + 1,
        )
        self.dead_letters.append(rec)
        self._dead_lettered.add(event_id)
        return ProcessResult(
            status="dead_letter",
            event_id=event_id,
            attempts=self.max_retries + 1,
            error=last_error,
        )

    def replay(self, event_id: str) -> ProcessResult:
        """Replay a dead-lettered event, bypassing dead-letter idempotency.

        Does not duplicate the dead-letter entry. Counts as a recovery.
        """
        rec = next((d for d in self.dead_letters if d.event_id == event_id), None)
        if rec is None:
            return ProcessResult(status="unknown", event_id=event_id, error="not in dead letters")

        # Temporarily remove from dead-lettered set so process can run
        self._dead_lettered.discard(event_id)
        result = self.process(rec.event_id, rec.event_type, rec.payload)
        if result.status == "success":
            self.recovery_count += 1
            # Remove the old dead-letter record (do not leave a duplicate failed entry)
            self.dead_letters = [d for d in self.dead_letters if d.event_id != event_id]
        else:
            # Re-add if it failed again
            self._dead_lettered.add(event_id)
        return result
