"""Project 4 — Memory-Enabled Conversational Agent.

Short-term rolling buffer + long-term recall with relevance scoring,
LLM-based compression of overflow, and cross-session persistence.

No real embeddings: relevance = deterministic token-overlap score
(|query words ∩ memory words| / |query words|), lowercase, whitespace-split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


class Memory:
    def __init__(self, llm, short_window: int = 4):
        """llm is used only by compress(). Must set up:
            - self.short_term: list of the most recent `short_window` turns
            - self.long_term: list of {"text": str, "source": "turn"|"summary"}
        """
        self.llm = llm
        self.short_window = short_window
        self.short_term: List[str] = []
        self.long_term: List[dict] = []

    def add_turn(self, text: str) -> None:
        """Append a turn.

        Requirements:
            - self.short_term holds at most `short_window` turns (most recent,
              in order).
            - A turn evicted from the buffer is moved to long_term with
              source "turn" — nothing is ever silently dropped.
        """
        self.short_term.append(text)
        while len(self.short_term) > self.short_window:
            evicted = self.short_term.pop(0)
            self.long_term.append({"text": evicted, "source": "turn"})

    def relevance(self, query: str, text: str) -> float:
        """Token-overlap score as defined in the module docstring.
        Empty query -> 0.0."""
        q_words = query.lower().split()
        if not q_words:
            return 0.0
        t_words = set(text.lower().split())
        overlap = sum(1 for w in q_words if w in t_words)
        return overlap / len(q_words)

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Top-k long_term texts by relevance, highest first.

        Requirements:
            - Entries scoring 0 are never returned.
            - Ties keep insertion order (stable).
        """
        scored = []
        for idx, entry in enumerate(self.long_term):
            score = self.relevance(query, entry["text"])
            if score > 0.0:
                scored.append((score, idx, entry["text"]))
        # Highest score first; lower insertion index breaks ties
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [text for _, _, text in scored[:k]]

    def compress(self) -> None:
        """Summarize long_term "turn" entries into one "summary" entry.

        Requirements:
            - Call self.llm.complete() once with a prompt containing every
              long_term turn text; the response replaces those entries as one
              {"text": <response>, "source": "summary"}.
            - Existing "summary" entries are preserved (not re-compressed).
            - No-op (no LLM call) when there are no "turn" entries.
        """
        turns = [e for e in self.long_term if e["source"] == "turn"]
        if not turns:
            return
        prompt_parts = ["Summarize the following turns:"]
        for e in turns:
            prompt_parts.append(e["text"])
        prompt = "\n".join(prompt_parts)
        summary = self.llm.complete(prompt)
        kept = [e for e in self.long_term if e["source"] == "summary"]
        kept.append({"text": summary, "source": "summary"})
        self.long_term = kept

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
