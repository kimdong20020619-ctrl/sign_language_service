import os
import tempfile
from PySide6.QtCore import QThread, Signal


class _TTSWorker(QThread):
    finished = Signal()
    error = Signal(str)

    def __init__(self, text: str, rate: int, volume: float):
        super().__init__()
        self.text = text
        self.rate = rate
        self.volume = volume

    def run(self) -> None:
        if self._try_pyttsx3():
            return
        if self._try_gtts():
            return
        self.error.emit("TTS 엔진을 사용할 수 없습니다.")

    # ── pyttsx3 (오프라인, 1순위) ──────────────────────────────────
    def _try_pyttsx3(self) -> bool:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._set_korean_voice(engine)
            engine.say(self.text)
            engine.runAndWait()
            engine.stop()
            self.finished.emit()
            return True
        except Exception as e:
            print(f"[TTS] pyttsx3 실패: {e}")
            return False

    def _set_korean_voice(self, engine) -> None:
        voices = engine.getProperty("voices")
        for v in voices:
            name_lower = (v.name or "").lower()
            id_lower = (v.id or "").lower()
            if "korean" in name_lower or "ko" in id_lower or "heami" in name_lower:
                engine.setProperty("voice", v.id)
                return

    # ── gTTS (온라인, 2순위) ───────────────────────────────────────
    def _try_gtts(self) -> bool:
        tmp_path = None
        try:
            from gtts import gTTS
            tts = gTTS(text=self.text, lang="ko", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            self._play_mp3(tmp_path)
            self.finished.emit()
            return True
        except Exception as e:
            print(f"[TTS] gTTS 실패: {e}")
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _play_mp3(self, path: str) -> None:
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                self.msleep(50)
            pygame.mixer.music.unload()
        except Exception:
            # pygame 없을 때 Windows 기본 플레이어로 fallback
            import subprocess
            subprocess.run(["start", "/wait", path], shell=True, check=False)


class TTSEngine:
    """비동기 TTS 엔진. pyttsx3 → gTTS 순으로 시도."""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._enabled = True
        self._worker: _TTSWorker | None = None

    def speak(self, text: str) -> None:
        if not self._enabled or not text.strip():
            return
        self.stop()
        self._worker = _TTSWorker(text, self.rate, self.volume)
        self._worker.error.connect(lambda msg: print(f"[TTS] {msg}"))
        self._worker.start()

    def stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)

    def set_rate(self, rate: int) -> None:
        self.rate = rate

    def set_volume(self, volume: float) -> None:
        self.volume = volume

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.stop()
