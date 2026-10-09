from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class WordBuilder:
    hold_time_seconds: float
    delete_labels: Iterable[str]
    space_labels: Iterable[str]
    clear_labels: Iterable[str]
    ignore_labels: Iterable[str]
    current_word: str = ""
    sentence: str = ""
    _active_label: str | None = None
    _active_since: float = 0.0
    _accepted_active: bool = False
    _delete_set: set[str] = field(init=False, repr=False)
    _space_set: set[str] = field(init=False, repr=False)
    _clear_set: set[str] = field(init=False, repr=False)
    _ignore_set: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._delete_set = {label.lower() for label in self.delete_labels}
        self._space_set = {label.lower() for label in self.space_labels}
        self._clear_set = {label.lower() for label in self.clear_labels}
        self._ignore_set = {label.lower() for label in self.ignore_labels}

    def observe(self, label: str, confidence: float, now: float | None = None) -> str | None:
        del confidence
        now = time.monotonic() if now is None else now
        normalized = label.strip()
        if not normalized:
            self.reset_hold()
            return None
        if normalized != self._active_label:
            self._active_label = normalized
            self._active_since = now
            self._accepted_active = False
            return None
        if self._accepted_active or now - self._active_since < self.hold_time_seconds:
            return None

        self._accepted_active = True
        return self.accept_label(normalized)

    def accept_label(self, label: str) -> str | None:
        normalized = label.strip()
        key = normalized.lower()
        if not normalized or key in self._ignore_set:
            return None
        if key in self._delete_set:
            self.delete_last()
            return "Deleted last character"
        if key in self._space_set:
            self.accept_space()
            return "Added space"
        if key in self._clear_set:
            self.clear_all()
            return "Cleared text"

        character = self._label_to_character(normalized)
        self.current_word += character
        return f"Added {character}"

    def accept_space(self, spell_corrector=None) -> str:
        return self.commit_word(spell_corrector)

    def commit_word(self, spell_corrector=None) -> str:
        word = self.current_word.strip()
        if not word:
            self._append_sentence_space()
            return "Added space"
        corrected = spell_corrector.correct(word) if spell_corrector else word
        self._append_to_sentence(corrected)
        self.current_word = ""
        return f"Committed {corrected}"

    def delete_last(self) -> None:
        if self.current_word:
            self.current_word = self.current_word[:-1]
            return
        self.sentence = self.sentence.rstrip()
        if not self.sentence:
            return
        parts = self.sentence.split(" ")
        parts = parts[:-1]
        self.sentence = " ".join(parts)
        if self.sentence:
            self.sentence += " "

    def clear_current_word(self) -> None:
        self.current_word = ""

    def clear_all(self) -> None:
        self.current_word = ""
        self.sentence = ""
        self.reset_hold()

    def reset_hold(self) -> None:
        self._active_label = None
        self._active_since = 0.0
        self._accepted_active = False

    def text_for_speech(self) -> str:
        pieces = [self.sentence.strip(), self.current_word.strip()]
        return " ".join(piece for piece in pieces if piece).strip()

    def _append_to_sentence(self, word: str) -> None:
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        self.sentence += word.strip()
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "

    def _append_sentence_space(self) -> None:
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "

    def _label_to_character(self, label: str) -> str:
        if len(label) == 1:
            return label.upper()
        if label.lower().startswith("letter_") and len(label) > 7:
            return label[7:8].upper()
        return label
