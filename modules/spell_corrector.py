from __future__ import annotations

import logging
from pathlib import Path


class SpellCorrector:
    def __init__(self, dictionary_path: Path | None = None, logger: logging.Logger | None = None) -> None:
        self.logger = logger
        self._spellchecker = None
        self._symspell = None
        self._init_spellchecker()
        self._init_symspell(dictionary_path)

    def correct(self, word: str) -> str:
        cleaned = word.strip()
        if not cleaned:
            return cleaned

        symspell = self._correct_with_symspell(cleaned)
        if symspell:
            return self._preserve_case(cleaned, symspell)

        if self._spellchecker is not None:
            candidate = self._spellchecker.correction(cleaned.lower())
            if candidate:
                return self._preserve_case(cleaned, candidate)
        return cleaned

    def suggestions(self, word: str, limit: int = 5) -> list[str]:
        cleaned = word.strip().lower()
        if not cleaned:
            return []
        suggestions: list[str] = []
        if self._symspell is not None:
            try:
                from symspellpy import Verbosity

                suggestions.extend(
                    item.term
                    for item in self._symspell.lookup(cleaned, Verbosity.CLOSEST, max_edit_distance=2)
                )
            except Exception:
                pass
        if self._spellchecker is not None:
            suggestions.extend(sorted(self._spellchecker.candidates(cleaned) or []))
        seen = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.append(suggestion)
            if len(seen) >= limit:
                break
        return seen

    def _init_spellchecker(self) -> None:
        try:
            from spellchecker import SpellChecker

            self._spellchecker = SpellChecker()
        except Exception as exc:
            if self.logger:
                self.logger.warning("PySpellChecker unavailable: %s", exc)

    def _init_symspell(self, dictionary_path: Path | None) -> None:
        try:
            from symspellpy import SymSpell

            self._symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            if dictionary_path and dictionary_path.exists():
                self._symspell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
        except Exception as exc:
            if self.logger:
                self.logger.warning("SymSpell unavailable: %s", exc)

    def _correct_with_symspell(self, word: str) -> str | None:
        if self._symspell is None:
            return None
        try:
            from symspellpy import Verbosity

            suggestions = self._symspell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
            return suggestions[0].term if suggestions else None
        except Exception:
            return None

    def _preserve_case(self, original: str, corrected: str) -> str:
        if original.isupper():
            return corrected.upper()
        if original[:1].isupper():
            return corrected.capitalize()
        return corrected
