from __future__ import annotations

from collections import Counter, deque


class PredictionBuffer:
    def __init__(self, max_size: int, min_votes: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if min_votes <= 0:
            raise ValueError("min_votes must be positive")
        self.max_size = max_size
        self.min_votes = min_votes
        self._items: deque[str] = deque(maxlen=max_size)

    def add(self, label: str) -> str | None:
        self._items.append(label)
        label_counts = Counter(self._items)
        winner, votes = label_counts.most_common(1)[0]
        if votes >= self.min_votes:
            return winner
        return None

    def clear(self) -> None:
        self._items.clear()

    def values(self) -> tuple[str, ...]:
        return tuple(self._items)
