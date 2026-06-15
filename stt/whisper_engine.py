"""
stt/whisper_engine.py  —  Whisper 로컬 한국어 STT 엔진

[설치]
  pip install openai-whisper sounddevice scipy

[모델 캐시]
  첫 실행 시 ~/.cache/whisper/ 에 자동 다운로드.
  이후 재시작부터는 로컬 캐시 사용 (재다운로드 없음).

[지원 모델] tiny / base / small / medium / large
  config.json → stt.model = "base" 권장 (속도/정확도 균형)
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

try:
    import whisper as _whisper_lib
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False

# ── 오디오 파라미터 ───────────────────────────────────────────────────
SAMPLE_RATE    = 16000   # Whisper 요구 샘플레이트
CHANNELS       = 1
CHUNK_SEC      = 3       # 한 번에 처리할 오디오 길이 (초)
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SEC

# ── VAD 파라미터 ──────────────────────────────────────────────────────
VAD_ENERGY_THR = 0.01    # RMS 에너지 임계값 (이 이상이어야 음성으로 판단)
VAD_SILENCE_DB = -40     # dBFS 기준 무음 판단
MIN_SPEECH_SEC = 0.3     # 최소 음성 지속 시간 (초) — 짧은 소음 필터


class WhisperEngine(QThread):
    """
    마이크 → sounddevice 캡처 → Whisper 인식 → Signal 전송.

    사용 예:
        engine = WhisperEngine(model_size="base", language="ko")
        engine.text_recognized.connect(chat_panel.set_hearing_text)
        engine.error_occurred.connect(lambda e: print(e))
        engine.start()

        engine.set_active(True)   # 인식 시작
        engine.set_active(False)  # 일시 정지
        engine.stop()             # 완전 종료
    """

    text_recognized:  Signal = Signal(str)   # 인식된 텍스트
    listening_state:  Signal = Signal(bool)  # 마이크 활성 상태 변경
    error_occurred:   Signal = Signal(str)   # 오류 메시지

    def __init__(
        self,
        model_size: str = "base",
        language:   str = "ko",
        enabled:    bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model_size = model_size
        self._language   = language
        self._enabled    = enabled
        self._active     = False       # 실시간 인식 활성 여부
        self._running    = False

        self._model: Optional[object] = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

        # 모델 로드는 run() 안에서 수행 (QThread 시작 후)
        self._model_ready = threading.Event()

    # ── 공개 API ─────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """인식 활성/비활성 토글. 마이크 스트림은 유지."""
        self._active = active
        self.listening_state.emit(active)

    def stop(self) -> None:
        """엔진 완전 종료."""
        self._running = False
        self._active  = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self.wait(5000)

    # ── QThread.run ───────────────────────────────────────────────────

    def run(self) -> None:
        if not self._enabled:
            return

        # 의존 패키지 확인
        if not _SD_OK:
            self.error_occurred.emit("sounddevice 미설치 — pip install sounddevice")
            return
        if not _WHISPER_OK:
            self.error_occurred.emit("openai-whisper 미설치 — pip install openai-whisper")
            return

        # Whisper 모델 로드 (캐시 우선)
        try:
            self._model = _whisper_lib.load_model(self._model_size)
            self._model_ready.set()
        except Exception as e:
            self.error_occurred.emit(f"Whisper 모델 로드 실패: {e}")
            return

        # 마이크 스트림 시작
        self._running = True
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=int(SAMPLE_RATE * 0.1),  # 100ms 블록
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self.error_occurred.emit(f"마이크 스트림 오류: {e}")
            return

        # 인식 루프
        self._recognition_loop()

    # ── 오디오 콜백 (sounddevice 내부 스레드) ────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            pass  # overflow 등 무시
        if self._active and self._running:
            self._audio_queue.put(indata.copy().flatten())

    # ── 인식 루프 ─────────────────────────────────────────────────────

    def _recognition_loop(self) -> None:
        buffer = np.array([], dtype=np.float32)

        while self._running:
            # 큐에서 오디오 청크 수집
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, chunk])
            except queue.Empty:
                if not self._active:
                    buffer = np.array([], dtype=np.float32)
                continue

            # CHUNK_SAMPLES 쌓이면 인식 실행
            if len(buffer) < CHUNK_SAMPLES:
                continue

            audio_segment = buffer[:CHUNK_SAMPLES]
            buffer        = buffer[CHUNK_SAMPLES:]

            # VAD: 무음이면 스킵
            if not self._has_speech(audio_segment):
                continue

            # Whisper 인식
            result = self._transcribe(audio_segment)
            if result:
                self.text_recognized.emit(result)

    # ── VAD (에너지 기반) ────────────────────────────────────────────

    def _has_speech(self, audio: np.ndarray) -> bool:
        """간단한 에너지 기반 VAD. 음성이 포함된 경우 True."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < VAD_ENERGY_THR:
            return False

        # 음성 구간 비율 체크 (MIN_SPEECH_SEC 이상 음성이어야)
        frame_size   = int(SAMPLE_RATE * 0.02)   # 20ms 프레임
        frames       = len(audio) // frame_size
        speech_count = 0

        for i in range(frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            if np.sqrt(np.mean(frame ** 2)) >= VAD_ENERGY_THR:
                speech_count += 1

        speech_ratio = speech_count / max(frames, 1)
        min_ratio    = MIN_SPEECH_SEC / CHUNK_SEC

        return speech_ratio >= min_ratio

    # ── Whisper 인식 ──────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            return ""
        try:
            result = self._model.transcribe(
                audio,
                language=self._language,
                fp16=False,            # CPU 환경 호환
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            text = result.get("text", "").strip()
            return self._filter_noise_text(text)
        except Exception as e:
            self.error_occurred.emit(f"Whisper 인식 오류: {e}")
            return ""

    # ── 노이즈 텍스트 필터 ────────────────────────────────────────────

    _NOISE_PATTERNS = {
        "MBC", "KBS", "SBS",        # 방송사 오인식
        "시청해 주셔서 감사합니다",
        "구독과 좋아요",
        "Thank you",
        "thanks for watching",
    }

    def _filter_noise_text(self, text: str) -> str:
        """Whisper가 무음에서 종종 뱉는 노이즈 텍스트 제거."""
        if not text:
            return ""
        t_lower = text.lower()
        for pat in self._NOISE_PATTERNS:
            if pat.lower() in t_lower:
                return ""
        # 1글자 이하 무시
        if len(text.replace(" ", "")) <= 1:
            return ""
        return text

    # ── 설정 변경 ─────────────────────────────────────────────────────

    def set_language(self, language: str) -> None:
        self._language = language

    def set_model_size(self, size: str) -> None:
        """모델 크기 변경. 엔진 재시작 필요."""
        self._model_size = size
        self._model      = None
        self._model_ready.clear()
