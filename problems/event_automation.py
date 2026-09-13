"""Project 7 — Event-Triggered Automation Agent.
Consume webhook/queue events with idempotent execution, bounded retries with
exponential backoff, and a dead-letter queue. No LLM involved — this grades
production automation engineering.

An event is {"id": str, "type": str, "payload": dict}.
"""


class EventProcessor:
    def __init__(self, handlers: dict, max_retries: int = 3, sleep=None):
        """handlers: event type -> callable(payload) -> result.
        sleep: injectable sleep function (tests pass a recorder; default
        time.sleep). Must set up:
            - self.processed: {event_id: result} of successful events
            - self.dead_letter: list of {"event": event, "error": str,
              "attempts": int}
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def replay_dead_letter(self) -> int:
        """Retry every dead-lettered event once more through process()
        (idempotency must not block the replay). Return how many succeeded.
        Events that fail again remain dead-lettered exactly once (no dupes)."""
        raise NotImplementedError
