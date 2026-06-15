"""
tts/gtts_engine.py  —  gTTS + pygame 한국어 TTS 엔진

[설치]
  pip install gTTS pygame
"""

from __future__ import annotations

import os
import queue
import tempfile
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

try:
    from gtts import gTTS
    _GTTS_OK = True
except ImportError:
    _GTTS_OK = False

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False


class GTTSEngine(QThread):
    """
    gTTS + pygame 기반 비동기 TTS 엔진.

    사용 예:
        engine = GTTSEngine(language="ko", slow=False)
        engine.playback_finished.connect(lambda: print("재생 완료"))
        engine.start()

        engine.speak("안녕하세요")          # 즉시 재생 (현재 재생 중단)
        engine.speak("감사합니다", slow=True)
        engine.stop()
    """

    playback_finished: Signal = Signal()        # 한 문장 재생 완료
    playback_started:  Signal = Signal(str)     # 재생 시작 (텍스트 전달)
    error_occurred:    Signal = Signal(str)     # 오류 메시지

    # 내부 sentinel: 스레드 종료 신호
    _STOP_SENTINEL = object()

    def __init__(
        self,
        language: str = "ko",
        slow: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._language  = language
        self._slow      = slow
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running   = False
        self._mixer_ok  = False
        self._tmp_files: list[Path] = []

    # ── 공개 API ─────────────────────────────────────────────────────

    def speak(self, text: str, slow: Optional[bool] = None) -> None:
        """
        텍스트를 재생 큐에 넣는다.
        현재 재생 중인 항목이 있으면 중단하고 새 항목을 재생한다.
        """
        if not text or not text.strip():
            return
        # 기존 항목 제거 후 새 항목 삽입 (interrupt 동작)
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put((text.strip(), slow if slow is not None else self._slow))
        self._stop_current_playback()

    def stop(self) -> None:
        """엔진 완전 종료."""
        self._running = False
        self._stop_current_playback()
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(self._STOP_SENTINEL)
        self.wait(3000)
        self._cleanup_tmp()

    # ── QThread.run ───────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        self._init_mixer()

        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is self._STOP_SENTINEL:
                break

            text, slow = item
            self._play(text, slow)

        self._cleanup_tmp()
        if self._mixer_ok:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # ── 내부 메서드 ───────────────────────────────────────────────────

    def _init_mixer(self) -> None:
        if not _PYGAME_OK:
            self.error_occurred.emit("pygame 미설치 — pip install pygame")
            return
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._mixer_ok = True
        except Exception as e:
            self.error_occurred.emit(f"pygame.mixer 초기화 실패: {e}")

    def _play(self, text: str, slow: bool) -> None:
        if not _GTTS_OK:
            self.error_occurred.emit("gTTS 미설치 — pip install gTTS")
            return
        if not self._mixer_ok:
            self.error_occurred.emit("pygame.mixer 초기화 안 됨")
            return

        tmp = Path(tempfile.mktemp(suffix=".mp3"))
        self._tmp_files.append(tmp)

        try:
            # 1. gTTS 변환
            tts = gTTS(text=text, lang=self._language, slow=slow)
            tts.save(str(tmp))

            # 2. pygame 재생
            self.playback_started.emit(text)
            pygame.mixer.music.load(str(tmp))
            pygame.mixer.music.play()

            # 3. 재생 완료 대기 (0.1초 폴링, 중단 체크)
            while pygame.mixer.music.get_busy() and self._running:
                time.sleep(0.05)
                # 큐에 새 항목이 있으면 즉시 중단
                if not self._queue.empty():
                    pygame.mixer.music.stop()
                    break

            self.playback_finished.emit()

        except Exception as e:
            self.error_occurred.emit(f"TTS 재생 오류: {e}")
        finally:
            # pygame이 파일을 잡고 있으므로 잠시 후 삭제
            pygame.mixer.music.unload()
            self._safe_remove(tmp)

    def _stop_current_playback(self) -> None:
        if self._mixer_ok:
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except Exception:
                pass

    def _safe_remove(self, path: Path) -> None:
        for _ in range(5):
            try:
                if path.exists():
                    path.unlink()
                return
            except PermissionError:
                time.sleep(0.1)

    def _cleanup_tmp(self) -> None:
        for p in list(self._tmp_files):
            self._safe_remove(p)
        self._tmp_files.clear()

    # ── 설정 ─────────────────────────────────────────────────────────

    def set_language(self, lang: str) -> None:
        self._language = lang

    def set_slow(self, slow: bool) -> None:
        self._slow = slow
