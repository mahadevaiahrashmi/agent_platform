"""Project 4 — Memory-Enabled Conversational Agent.
Short-term rolling buffer + long-term recall with relevance scoring,
LLM-based compression of overflow, and cross-session persistence.

No real embeddings: relevance = deterministic token-overlap score
(|query words ∩ memory words| / |query words|), lowercase, whitespace-split.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

def identity_redactor(text: str) -> str:
    return text

class TokenOverlapScorer:
    def score(self, query: str, text: str) -> float:
        q = query.lower().split()
        if not q:
            return 0.0
        t = set(text.lower().split())
        return sum(1 for w in q if w in t) / len(q)



class Memory:
    def __init__(
        self,
        llm,
        short_window: int = 4,
        *,
        scorer=None,
        redactor=None,
        long_term_ttl_seconds: Optional[float] = None,
        max_long_term: Optional[int] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """llm is used only by compress(). Must set up:
- self.short_term: list of the most recent `short_window` turns
- self.long_term: list of {"text": str, "source": "turn"|"summary"}"""
        self.llm = llm
        self.short_window = short_window
        self.short_term: List[str] = []
        self.long_term: List[dict] = []
        self.scorer = scorer or TokenOverlapScorer()
        self.redactor = redactor or identity_redactor
        self.long_term_ttl_seconds = long_term_ttl_seconds
        self.max_long_term = max_long_term
        self._clock = clock or time.monotonic

    def add_turn(self, text: str) -> None:
        """Append a turn.
Requirements:
    - self.short_term holds at most `short_window` turns (most recent,
      in order).
    - A turn evicted from the buffer is moved to long_term with
      source "turn" — nothing is ever silently dropped."""
        safe = self.redactor(text)
        self.short_term.append(safe)
        while len(self.short_term) > self.short_window:
            evicted = self.short_term.pop(0)
            self.long_term.append({"text": evicted, "source": "turn", "ts": self._clock()})
            self._enforce_long_term_limits()

    def relevance(self, query: str, text: str) -> float:
        """Token-overlap score as defined in the module docstring.
Empty query -> 0.0."""
        return self.scorer.score(query, text)

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Top-k long_term texts by relevance, highest first.
Requirements:
    - Entries scoring 0 are never returned.
    - Ties keep insertion order (stable)."""
        self._purge_expired()
        scored = []
        for idx, entry in enumerate(self.long_term):
            score = self.relevance(query, entry["text"])
            if score > 0.0:
                scored.append((score, idx, entry["text"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [text for _, _, text in scored[:k]]

    def compress(self) -> None:
        """Summarize long_term "turn" entries into one "summary" entry.
Requirements:
    - Call self.llm.complete() once with a prompt containing every
      long_term turn text; the response replaces those entries as one
      {"text": <response>, "source": "summary"}.
    - Existing "summary" entries are preserved (not re-compressed).
    - No-op (no LLM call) when there are no "turn" entries."""
        turns = [e for e in self.long_term if e["source"] == "turn"]
        if not turns:
            return
        prompt_parts = ["Summarize the following turns:"]
        for e in turns:
            prompt_parts.append(e["text"])
        prompt = "\n".join(prompt_parts)
        summary = self.llm.complete(prompt)
        kept = [e for e in self.long_term if e["source"] == "summary"]
        kept.append({"text": summary, "source": "summary", "ts": self._clock()})
        self.long_term = kept
        self._enforce_long_term_limits()

    def save(self, path: str) -> None:
        """Persist short_term + long_term as JSON."""
        data = {
            "short_window": self.short_window,
            "short_term": self.short_term,
            "long_term": self.long_term,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, llm, path: str) -> "Memory":
        """Restore a Memory (same short_window semantics) from save()'s JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(llm, short_window=data.get("short_window", 4))
        m.short_term = list(data.get("short_term", []))
        m.long_term = list(data.get("long_term", []))
        return m

    def _purge_expired(self) -> None:
        if not self.long_term_ttl_seconds:
            return
        now = self._clock()
        self.long_term = [
            e for e in self.long_term
            if e.get("ts") is None or (now - float(e["ts"])) <= self.long_term_ttl_seconds
        ]

    def _enforce_long_term_limits(self) -> None:
        self._purge_expired()
        if self.max_long_term is not None and len(self.long_term) > self.max_long_term:
            overflow = len(self.long_term) - self.max_long_term
            self.long_term = self.long_term[overflow:]
