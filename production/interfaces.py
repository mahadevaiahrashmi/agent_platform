"""Pluggable interfaces — in-memory defaults; swap backends without changing agents."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol


class BudgetChecker(Protocol):
    def can_afford(self, cost: float) -> bool: ...
    def charge(self, cost: float) -> None: ...
    @property
    def spent(self) -> float: ...


class InMemoryBudget:
    def __init__(self, limit: float):
        self.limit = limit
        self._spent = 0.0

    def can_afford(self, cost: float) -> bool:
        return self._spent + cost <= self.limit

    def charge(self, cost: float) -> None:
        if not self.can_afford(cost):
            raise RuntimeError("budget exceeded")
        self._spent += cost

    @property
    def spent(self) -> float:
        return self._spent


class RelevanceScorer(Protocol):
    def score(self, query: str, text: str) -> float: ...


class TokenOverlapScorer:
    def score(self, query: str, text: str) -> float:
        q = query.lower().split()
        if not q:
            return 0.0
        t = set(text.lower().split())
        return sum(1 for w in q if w in t) / len(q)


class SpanExporter(Protocol):
    def export(self, spans: List[dict]) -> None: ...


class NullSpanExporter:
    def export(self, spans: List[dict]) -> None:
        return None


class InMemorySpanExporter:
    def __init__(self) -> None:
        self.exported: List[dict] = []

    def export(self, spans: List[dict]) -> None:
        self.exported.extend(spans)


class TicketStore(Protocol):
    def save(self, ticket_id: str, data: dict) -> None: ...
    def load(self, ticket_id: str) -> Optional[dict]: ...
    def delete(self, ticket_id: str) -> None: ...


class InMemoryTicketStore:
    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}

    def save(self, ticket_id: str, data: dict) -> None:
        self._data[ticket_id] = dict(data)

    def load(self, ticket_id: str) -> Optional[dict]:
        return self._data.get(ticket_id)

    def delete(self, ticket_id: str) -> None:
        self._data.pop(ticket_id, None)


class EventStore(Protocol):
    def mark_seen(self, event_id: str) -> None: ...
    def is_seen(self, event_id: str) -> bool: ...
    def unmark(self, event_id: str) -> None: ...


class InMemoryEventStore:
    def __init__(self) -> None:
        self._seen: set = set()

    def mark_seen(self, event_id: str) -> None:
        self._seen.add(event_id)

    def is_seen(self, event_id: str) -> bool:
        return event_id in self._seen

    def unmark(self, event_id: str) -> None:
        self._seen.discard(event_id)


class MetricsHook(Protocol):
    def record(self, name: str, value: float = 1.0, **labels: Any) -> None: ...


class NullMetrics:
    def record(self, name: str, value: float = 1.0, **labels: Any) -> None:
        return None


class ListMetrics:
    def __init__(self) -> None:
        self.events: List[dict] = []

    def record(self, name: str, value: float = 1.0, **labels: Any) -> None:
        self.events.append({"name": name, "value": value, **labels})


Redactor = Callable[[str], str]


def identity_redactor(text: str) -> str:
    return text
