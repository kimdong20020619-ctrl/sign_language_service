import time
from typing import List, Optional


class SentenceBuilder:
    """수화 단어 예측을 누적해 문장을 조합한다.

    같은 단어가 `hold_frames` 프레임 연속으로 인식되면 문장에 추가.
    직전 단어와 동일하면 `min_word_interval` 초 경과 후에만 다시 추가.
    """

    def __init__(self, hold_frames: int = 15, max_words: int = 30,
                 min_word_interval: float = 1.2):
        self.hold_frames = hold_frames
        self.max_words = max_words
        self.min_word_interval = min_word_interval

        self._words: List[str] = []
        self._current_word: Optional[str] = None
        self._hold_count: int = 0
        self._last_added_time: float = 0.0

    # ------------------------------------------------------------------
    def add_prediction(self, word: Optional[str]) -> bool:
        """예측 단어 입력. 문장에 추가됐으면 True 반환."""
        if word is None:
            self._current_word = None
            self._hold_count = 0
            return False

        if word == self._current_word:
            self._hold_count += 1
            if self._hold_count >= self.hold_frames:
                return self._try_append(word)
        else:
            self._current_word = word
            self._hold_count = 1

        return False

    def _try_append(self, word: str) -> bool:
        now = time.monotonic()
        last_same = self._words and self._words[-1] == word
        if last_same and (now - self._last_added_time) < self.min_word_interval:
            return False

        self._words.append(word)
        if len(self._words) > self.max_words:
            self._words.pop(0)
        self._last_added_time = now
        self._hold_count = 0
        return True

    # ------------------------------------------------------------------
    def get_sentence(self) -> str:
        return " ".join(self._words)

    def get_words(self) -> List[str]:
        return list(self._words)

    def remove_last_word(self) -> Optional[str]:
        if self._words:
            return self._words.pop()
        return None

    def clear(self) -> None:
        self._words.clear()
        self._current_word = None
        self._hold_count = 0

    @property
    def word_count(self) -> int:
        return len(self._words)

    @property
    def is_empty(self) -> bool:
        return len(self._words) == 0
