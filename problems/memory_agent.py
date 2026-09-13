"""Project 4 — Memory-Enabled Conversational Agent.
Short-term rolling buffer + long-term recall with relevance scoring,
LLM-based compression of overflow, and cross-session persistence.

No real embeddings: relevance = deterministic token-overlap score
(|query words ∩ memory words| / |query words|), lowercase, whitespace-split.
"""


class Memory:
    def __init__(self, llm, short_window: int = 4):
        """llm is used only by compress(). Must set up:
            - self.short_term: list of the most recent `short_window` turns
            - self.long_term: list of {"text": str, "source": "turn"|"summary"}
        """
        raise NotImplementedError

    def add_turn(self, text: str) -> None:
        """Append a turn.
        Requirements:
            - self.short_term holds at most `short_window` turns (most recent,
              in order).
            - A turn evicted from the buffer is moved to long_term with
              source "turn" — nothing is ever silently dropped.
        """
        raise NotImplementedError

    def relevance(self, query: str, text: str) -> float:
        """Token-overlap score as defined in the module docstring.
        Empty query -> 0.0."""
        raise NotImplementedError

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Top-k long_term texts by relevance, highest first.
        Requirements:
            - Entries scoring 0 are never returned.
            - Ties keep insertion order (stable).
        """
        raise NotImplementedError

    def compress(self) -> None:
        """Summarize long_term "turn" entries into one "summary" entry.
        Requirements:
            - Call self.llm.complete() once with a prompt containing every
              long_term turn text; the response replaces those entries as one
              {"text": <response>, "source": "summary"}.
            - Existing "summary" entries are preserved (not re-compressed).
            - No-op (no LLM call) when there are no "turn" entries.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist short_term + long_term as JSON."""
        raise NotImplementedError

    @classmethod
    def load(cls, llm, path: str) -> "Memory":
        """Restore a Memory (same short_window semantics) from save()'s JSON."""
        raise NotImplementedError
