"""Project 7 — Event-Triggered Automation Agent.

Consume webhook/queue events with idempotent execution, bounded retries with
exponential backoff, and a dead-letter queue. No LLM involved — this grades
production automation engineering.

An event is {"id": str, "type": str, "payload": dict}.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set


class EventProcessor:
    def __init__(self, handlers: dict, max_retries: int = 3, sleep=None):
        """handlers: event type -> callable(payload) -> result.
        sleep: injectable sleep function (tests pass a recorder; default
        time.sleep). Must set up:
            - self.processed: {event_id: result} of successful events
            - self.dead_letter: list of {"event": event, "error": str,
              "attempts": int}
        """
        self.handlers = handlers
        self.max_retries = max_retries
        self.sleep = sleep if sleep is not None else time.sleep
        self.processed: Dict[str, Any] = {}
        self.dead_letter: List[dict] = []
        self._seen_ids: Set[str] = set()  # success OR dead-lettered

    def process(self, event: dict) -> dict:
        """Process one event.

        Requirements:
            - IDEMPOTENT: an event id seen before (success OR dead-lettered)
              returns {"status": "duplicate"} without invoking the handler.
            - Unknown event type -> straight to dead_letter (no retries),
              return {"status": "dead_letter"}.
            - Handler exceptions: retry up to max_retries additional attempts,
              calling self.sleep(2 ** attempt) between attempts (1, 2, 4...).
            - Success -> {"status": "ok", "result": ...} and record in
              self.processed.
            - Still failing after retries -> append to dead_letter with the
              LAST error string and total attempt count, return
              {"status": "dead_letter"}.
            - process() never raises.
        """
        event_id = event["id"]
        if event_id in self._seen_ids:
            return {"status": "duplicate"}

        event_type = event.get("type")
        payload = event.get("payload")

        if event_type not in self.handlers:
            self.dead_letter.append(
                {
                    "event": event,
                    "error": f"Unknown event type: {event_type}",
                    "attempts": 1,
                }
            )
            self._seen_ids.add(event_id)
            return {"status": "dead_letter"}

        handler = self.handlers[event_type]
        last_error = ""
        # attempt 0 is first try; then max_retries more
        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                result = handler(payload)
                self.processed[event_id] = result
                self._seen_ids.add(event_id)
                return {"status": "ok", "result": result}
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    # sleep(2 ** attempt) => 1, 2, 4, ...
                    self.sleep(2 ** attempt)

        # Exhausted
        self.dead_letter.append(
            {
                "event": event,
                "error": last_error,
                "attempts": total_attempts,
            }
        )
        self._seen_ids.add(event_id)
        return {"status": "dead_letter"}

    def replay_dead_letter(self) -> int:
        """Retry every dead-lettered event once more through process()
        (idempotency must not block the replay). Return how many succeeded.
        Events that fail again remain dead-lettered exactly once (no dupes)."""
        to_replay = list(self.dead_letter)
        self.dead_letter = []
        recovered = 0
        for entry in to_replay:
            event = entry["event"]
            event_id = event["id"]
            # Allow re-processing by removing from seen set
            self._seen_ids.discard(event_id)
            result = self.process(event)
            if result.get("status") == "ok":
                recovered += 1
            # if it failed again, process() already put it back in dead_letter
        return recovered
