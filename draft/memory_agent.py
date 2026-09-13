"""Maintain rolling short-term turns and searchable, compressible, persistent long-term memory.

Evicted turns move to long-term storage; relevance is lowercase whitespace-token
overlap with an empty-query score of 0.0; recall filters zero scores, ranks
highest first, and keeps insertion order on ties; compression replaces only
long-term turn entries with one LLM summary while preserving existing summaries;
save/load restores both memory tiers and window semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Turn:
    role: str
    content: str
    is_summary: bool = False


@dataclass
class MemoryState:
    short_term: List[Turn] = field(default_factory=list)
    long_term: List[Turn] = field(default_factory=list)
    window_size: int = 10


def _tokenize(text: str) -> set:
    return set(text.lower().split())


def _overlap_score(query: str, content: str) -> float:
    if not query.strip():
        return 0.0
    q = _tokenize(query)
    c = _tokenize(content)
    if not q:
        return 0.0
    return len(q & c) / len(q)


class MemoryAgent:
    def __init__(self, window_size: int = 10, llm: Any = None):
        self.window_size = window_size
        self.llm = llm
        self.short_term: List[Turn] = []
        self.long_term: List[Turn] = []

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn; evict oldest short-term turns into long-term when over window."""
        self.short_term.append(Turn(role=role, content=content))
        while len(self.short_term) > self.window_size:
            evicted = self.short_term.pop(0)
            self.long_term.append(evicted)

    def recall(self, query: str, k: int = 5) -> List[Turn]:
        """Return up to k long-term turns ranked by token-overlap relevance.

        Zero-score turns are filtered out.  Highest score first; insertion order
        breaks ties.
        """
        scored = []
        for idx, turn in enumerate(self.long_term):
            score = _overlap_score(query, turn.content)
            if score > 0.0:
                scored.append((score, idx, turn))
        # Highest score first, then original insertion order (lower idx)
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [t for _, _, t in scored[:k]]

    def compress(self) -> None:
        """Replace non-summary long-term turns with a single LLM summary.

        Existing summary entries are preserved.  Requires an llm.
        """
        if self.llm is None:
            raise RuntimeError("compress requires an llm")
        to_summarize = [t for t in self.long_term if not t.is_summary]
        if not to_summarize:
            return
        text = "\n".join(f"{t.role}: {t.content}" for t in to_summarize)
        prompt = (
            "Summarize the following conversation turns into a concise paragraph "
            "that preserves key facts and decisions.\n\n" + text
        )
        summary = self.llm.complete(prompt)
        # Keep existing summaries, replace the rest with one new summary
        kept = [t for t in self.long_term if t.is_summary]
        kept.append(Turn(role="system", content=summary, is_summary=True))
        self.long_term = kept

    def save(self, path: str | Path) -> None:
        state = {
            "window_size": self.window_size,
            "short_term": [asdict(t) for t in self.short_term],
            "long_term": [asdict(t) for t in self.long_term],
        }
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.window_size = data.get("window_size", self.window_size)
        self.short_term = [Turn(**t) for t in data.get("short_term", [])]
        self.long_term = [Turn(**t) for t in data.get("long_term", [])]
