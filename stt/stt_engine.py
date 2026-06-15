from typing import Callable, Optional
from PySide6.QtCore import QThread, Signal


class _STTWorker(QThread):
    text_recognized = Signal(str)
    listening_started = Signal()
    error = Signal(str)

    def __init__(self, language: str = "ko-KR",
                 timeout: int = 5, phrase_limit: int = 10):
        super().__init__()
        self.language = language
        self.timeout = timeout
        self.phrase_limit = phrase_limit

    def run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            self.error.emit("SpeechRecognition 라이브러리가 설치되지 않았습니다.")
            return

        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.4)
                self.listening_started.emit()
                audio = r.listen(source,
                                 timeout=self.timeout,
                                 phrase_time_limit=self.phrase_limit)
        except sr.WaitTimeoutError:
            self.error.emit("음성 입력 시간 초과")
            return
        except OSError as e:
            self.error.emit(f"마이크 오류: {e}")
            return

        try:
            text = r.recognize_google(audio, language=self.language)
            self.text_recognized.emit(text)
        except sr.UnknownValueError:
            self.error.emit("음성을 인식하지 못했습니다.")
        except sr.RequestError as e:
            self.error.emit(f"STT 서비스 오류: {e}")


class STTEngine:
    """비동기 STT 엔진 (Google Speech Recognition 기반)."""

    def __init__(self, language: str = "ko-KR"):
        self.language = language
        self._worker: Optional[_STTWorker] = None

    def start_listening(
        self,
        on_recognized: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_listening: Optional[Callable[[], None]] = None,
        timeout: int = 5,
        phrase_limit: int = 10,
    ) -> None:
        if self._worker and self._worker.isRunning():
            return  # 이미 청취 중

        self._worker = _STTWorker(self.language, timeout, phrase_limit)
        if on_recognized:
            self._worker.text_recognized.connect(on_recognized)
        if on_error:
            self._worker.error.connect(on_error)
        if on_listening:
            self._worker.listening_started.connect(on_listening)
        self._worker.start()

    def stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)

    @property
    def is_listening(self) -> bool:
        return self._worker is not None and self._worker.isRunning()
