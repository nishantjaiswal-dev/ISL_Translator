from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceFilter:
    threshold: float = 0.80

    def accepts(self, confidence: float) -> bool:
        return confidence >= self.threshold
