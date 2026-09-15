"""Project 7 — Event automation (production, stack-independent)."""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional

from interfaces import EventStore, InMemoryEventStore


class EventProcessor:
    def __init__(
        self,
        handlers: dict,
        max_retries: int = 3,
        sleep=None,
        *,
        jitter: bool = False,
        max_event_age_seconds: Optional[float] = None,
        max_replay_attempts: int = 3,
        store: Optional[EventStore] = None,
        clock: Optional[Callable[[], float]] = None,
        rng: Optional[Callable[[], float]] = None,
    ):
        self.handlers = handlers
        self.max_retries = max_retries
        self.sleep = sleep if sleep is not None else time.sleep
        self.processed: Dict[str, Any] = {}
        self.dead_letter: List[dict] = []
        self._store = store or InMemoryEventStore()
        self.jitter = jitter
        self.max_event_age_seconds = max_event_age_seconds
        self.max_replay_attempts = max_replay_attempts
        self._replay_counts: Dict[str, int] = {}
        self._clock = clock or time.monotonic
        self._rng = rng or random.random

    def process(self, event: dict) -> dict:
        event_id = event["id"]
        if self._store.is_seen(event_id):
            return {"status": "duplicate"}

        if self.max_event_age_seconds is not None and "ts" in event:
            age = self._clock() - float(event["ts"])
            if age > self.max_event_age_seconds:
                self.dead_letter.append({
                    "event": event, "error": "event expired", "attempts": 1,
                })
                self._store.mark_seen(event_id)
                return {"status": "dead_letter"}

        event_type = event.get("type")
        payload = event.get("payload")

        if event_type not in self.handlers:
            self.dead_letter.append({
                "event": event,
                "error": f"Unknown event type: {event_type}",
                "attempts": 1,
            })
            self._store.mark_seen(event_id)
            return {"status": "dead_letter"}

        handler = self.handlers[event_type]
        last_error = ""
        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                result = handler(payload)
                self.processed[event_id] = result
                self._store.mark_seen(event_id)
                return {"status": "ok", "result": result}
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    delay = float(2 ** attempt)
                    if self.jitter:
                        delay = delay * (0.5 + self._rng())
                    self.sleep(delay)

        self.dead_letter.append({
            "event": event, "error": last_error, "attempts": total_attempts,
        })
        self._store.mark_seen(event_id)
        return {"status": "dead_letter"}

    def replay_dead_letter(self) -> int:
        to_replay = list(self.dead_letter)
        self.dead_letter = []
        recovered = 0
        for entry in to_replay:
            event = entry["event"]
            event_id = event["id"]
            count = self._replay_counts.get(event_id, 0) + 1
            self._replay_counts[event_id] = count
            if count > self.max_replay_attempts:
                self.dead_letter.append({
                    "event": event,
                    "error": entry.get("error", "poison"),
                    "attempts": entry.get("attempts", 0),
                    "poison": True,
                })
                self._store.mark_seen(event_id)
                continue
            self._store.unmark(event_id)
            result = self.process(event)
            if result.get("status") == "ok":
                recovered += 1
        return recovered
