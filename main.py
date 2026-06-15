"""
main.py — 🤟 수화 소통 서비스 진입점

초기화 순서:
  ① config.json 로드
  ② Ollama 서버 연결 확인  (미연결 → 대기 다이얼로그)
  ③ 수화 분류 모델 로드     (없음  → 데모 모드 안내)
  ④ Whisper STT 로드       (백그라운드 QThread)
  ⑤ 카메라 연결
  ⑥ MediaPipe 초기화
  ⑦ gTTS 엔진 초기화
  ⑧ SQLite DB 초기화
  ⑨ PySide6 메인 윈도우 실행

데이터 흐름:
  카메라 프레임 → MediaPipe → 특징 추출 → 정적/동적 분류기
  → 단어 확정(15프레임) → 문장 조합(3초 정지)
  → Ollama 스트리밍 교정 → ChatPanel 실시간 표시
  → gTTS 음성 출력 → SQLite 저장
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PySide6.QtGui import QPixmap, QFont

# ══════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_config(path: str = "config.json") -> dict:
    full = os.path.join(BASE_DIR, path)
    if not os.path.exists(full):
        QMessageBox.critical(None, "설정 파일 없음",
                             f"config.json 을 찾을 수 없습니다.\n경로: {full}")
        sys.exit(1)
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def _err_box(title: str, msg: str) -> None:
    QMessageBox.warning(None, title, msg)


# ── ② Ollama 연결 대기 다이얼로그 ─────────────────────────────────────

def _wait_for_ollama(bridge) -> tuple[bool, str]:
    """
    Ollama 연결을 최대 3회 재시도.
    사용자가 '없이 시작'을 누르면 (False, reason) 반환.
    """
    for attempt in range(1, 4):
        try:
            ok, msg = bridge.test_connection()
            if ok:
                return True, msg
        except Exception as e:
            ok, msg = False, str(e)

        dlg = QMessageBox()
        dlg.setWindowTitle("🤟 수화 소통 서비스")
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText(
            f"<b>Ollama 서버에 연결할 수 없습니다.</b><br><br>"
            f"터미널에서 다음 명령을 실행하세요:<br>"
            f"<code>&nbsp;&nbsp;ollama serve</code><br><br>"
            f"상태: {msg}<br>"
            f"(시도 {attempt}/3)"
        )
        retry_btn = dlg.addButton("재시도", QMessageBox.AcceptRole)
        skip_btn  = dlg.addButton("Ollama 없이 시작", QMessageBox.RejectRole)
        dlg.exec()
        if dlg.clickedButton() is skip_btn:
            return False, "수동 스킵"

    return False, msg


# ── ③ 모델 없음 안내 ────────────────────────────────────────────────

def _show_model_missing_info() -> None:
    QMessageBox.information(
        None, "수화 인식 모델 없음",
        "<b>학습된 수화 인식 모델이 없습니다.</b><br><br>"
        "모델 준비 순서:<br>"
        "1. <code>python data_collection/aihub_downloader.py</code><br>"
        "2. <code>python data_collection/collect_data.py</code><br>"
        "3. <code>python data_collection/train_model.py</code><br><br>"
        "<i>지금은 데모 모드로 실행됩니다.</i><br>"
        "(임의 단어가 표시될 수 있습니다)"
    )


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("🤟 수화 소통 서비스")
    app.setOrganizationName("SignLanguageAI")

    # 작업 디렉터리를 프로젝트 루트로 고정
    os.chdir(BASE_DIR)

    # ─────────────────────────────────────────────────────────────────
    # ① config.json 로드
    # ─────────────────────────────────────────────────────────────────
    config       = _load_config()
    default_mode = config.get("app", {}).get("default_mode", "카페")

    # ─────────────────────────────────────────────────────────────────
    # 모듈 임포트 (개별 try-catch: 패키지 미설치여도 앱 실행)
    # ─────────────────────────────────────────────────────────────────
    _import_errors: list[str] = []

    try:
        from core.camera import CameraThread
    except Exception as e:
        _import_errors.append(f"CameraThread: {e}")
        CameraThread = None                          # type: ignore[assignment,misc]

    try:
        from core.mediapipe_engine import MediaPipeEngine
    except Exception as e:
        _import_errors.append(f"MediaPipeEngine: {e}")
        MediaPipeEngine = None                       # type: ignore[assignment,misc]

    try:
        from core.gesture_classifier import GestureClassifier
    except Exception as e:
        _import_errors.append(f"GestureClassifier: {e}")
        GestureClassifier = None                     # type: ignore[assignment,misc]

    try:
        from core.sentence_builder import SentenceBuilder
    except Exception as e:
        _import_errors.append(f"SentenceBuilder: {e}")
        SentenceBuilder = None                       # type: ignore[assignment,misc]

    try:
        from core.context_manager import ContextManager
    except Exception as e:
        _import_errors.append(f"ContextManager: {e}")
        ContextManager = None                        # type: ignore[assignment,misc]

    try:
        from ai.ollama_bridge import OllamaBridge
    except Exception as e:
        _import_errors.append(f"OllamaBridge: {e}")
        OllamaBridge = None                          # type: ignore[assignment,misc]

    try:
        from tts.gtts_engine import GTTSEngine
    except Exception as e:
        _import_errors.append(f"GTTSEngine: {e}")
        GTTSEngine = None                            # type: ignore[assignment,misc]

    try:
        from stt.whisper_engine import WhisperEngine
    except Exception as e:
        _import_errors.append(f"WhisperEngine: {e}")
        WhisperEngine = None                         # type: ignore[assignment,misc]

    try:
        from db.history_db import HistoryDB
    except Exception as e:
        _import_errors.append(f"HistoryDB: {e}")
        HistoryDB = None                             # type: ignore[assignment,misc]

    from ui.main_window import MainWindow            # UI는 반드시 성공해야 함

    # ─────────────────────────────────────────────────────────────────
    # ② Ollama 연결 확인
    # ─────────────────────────────────────────────────────────────────
    ollama_bridge   = None
    ollama_ok       = False
    ollama_msg      = "OllamaBridge 로드 실패"

    if OllamaBridge is not None:
        try:
            ollama_bridge = OllamaBridge(config)
            ollama_ok, ollama_msg = _wait_for_ollama(ollama_bridge)
        except Exception as e:
            ollama_msg = str(e)
            _import_errors.append(f"Ollama 초기화: {e}")
    else:
        ollama_msg = "ollama 패키지 미설치"

    if ollama_bridge:
        ollama_bridge.set_mode(default_mode)

    # ─────────────────────────────────────────────────────────────────
    # ③ 수화 분류 모델 로드
    # ─────────────────────────────────────────────────────────────────
    gesture_classifier = None
    model_loaded       = False
    model_label        = "제스처 모델"

    g_cfg = config.get("gesture", {})

    if GestureClassifier is not None:
        try:
            gesture_classifier = GestureClassifier(
                model_path=g_cfg.get("model_path", "models/gesture_model.pkl"),
                confidence_threshold=g_cfg.get("confidence_threshold", 0.7),
            )
            model_loaded = gesture_classifier.is_loaded
            model_label  = (g_cfg.get("model_path", "")
                            .split("/")[-1].replace(".pkl", "") or "제스처 모델")
            if not model_loaded:
                _show_model_missing_info()
        except Exception as e:
            _import_errors.append(f"GestureClassifier 초기화: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ④ Whisper STT 로드 (백그라운드)
    # ─────────────────────────────────────────────────────────────────
    stt_engine = None
    stt_cfg    = config.get("stt", {})

    if WhisperEngine is not None and stt_cfg.get("enabled", True):
        try:
            stt_engine = WhisperEngine(
                model_size=stt_cfg.get("model", "base"),
                language=stt_cfg.get("language", "ko"),
                enabled=stt_cfg.get("enabled", True),
            )
            stt_engine.start()          # 백그라운드에서 Whisper 모델 로드
        except Exception as e:
            _import_errors.append(f"WhisperEngine: {e}")
            stt_engine = None

    # ─────────────────────────────────────────────────────────────────
    # ⑤ 카메라 연결
    # ─────────────────────────────────────────────────────────────────
    camera_thread = None
    cam_cfg       = config.get("camera", {})

    if CameraThread is not None:
        try:
            camera_thread = CameraThread(
                camera_index=cam_cfg.get("index", 0),
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps", 30),
            )
        except Exception as e:
            _import_errors.append(f"CameraThread: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ⑥ MediaPipe 초기화
    # ─────────────────────────────────────────────────────────────────
    mediapipe_engine = None
    mediapipe_worker = None
    mp_cfg           = config.get("mediapipe", {})

    if MediaPipeEngine is not None:
        try:
            mediapipe_engine = MediaPipeEngine(
                max_hands=mp_cfg.get("max_hands", 2),
                detection_confidence=mp_cfg.get("detection_confidence", 0.7),
                tracking_confidence=mp_cfg.get("tracking_confidence", 0.5),
            )
            from core.mediapipe_worker import MediaPipeWorker
            mediapipe_worker = MediaPipeWorker(mediapipe_engine)
            mediapipe_worker.start()
        except Exception as e:
            _import_errors.append(f"MediaPipeEngine: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ⑦ gTTS 엔진 초기화
    # ─────────────────────────────────────────────────────────────────
    tts_engine = None
    tts_cfg    = config.get("tts", {})

    if GTTSEngine is not None:
        try:
            tts_engine = GTTSEngine(
                language=tts_cfg.get("language", "ko"),
                slow=tts_cfg.get("slow", False),
            )
            tts_engine.start()          # pygame mixer 초기화 (백그라운드)
            tts_engine.error_occurred.connect(
                lambda msg: print(f"[TTS] {msg}")
            )
        except Exception as e:
            _import_errors.append(f"GTTSEngine: {e}")

    # ─────────────────────────────────────────────────────────────────
    # SentenceBuilder / ContextManager
    # ─────────────────────────────────────────────────────────────────
    sentence_builder = None
    if SentenceBuilder is not None:
        try:
            sentence_builder = SentenceBuilder(
                hold_frames=g_cfg.get("hold_frames", 15),
            )
        except Exception as e:
            _import_errors.append(f"SentenceBuilder: {e}")

    context_manager = None
    if ContextManager is not None:
        try:
            context_manager = ContextManager()
            context_manager.set_mode(default_mode)
        except Exception as e:
            _import_errors.append(f"ContextManager: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ⑧ SQLite DB 초기화
    # ─────────────────────────────────────────────────────────────────
    history_db = None
    if HistoryDB is not None:
        try:
            db_path    = config.get("db", {}).get(
                "path", "db/conversation_history.db"
            )
            history_db = HistoryDB(db_path=db_path)
        except Exception as e:
            _import_errors.append(f"HistoryDB: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ⑨ PySide6 메인 윈도우 실행
    # ─────────────────────────────────────────────────────────────────
    window = MainWindow(config)
    window.set_modules(
        camera_thread      = camera_thread,
        mediapipe_engine   = mediapipe_engine,
        gesture_classifier = gesture_classifier,
        sentence_builder   = sentence_builder,
        ollama_bridge      = ollama_bridge,
        tts_engine         = tts_engine,
        db                 = history_db,
        context_manager    = context_manager,
        stt_engine         = stt_engine,
        mediapipe_worker   = mediapipe_worker,
    )
    window.show()

    # ─────────────────────────────────────────────────────────────────
    # 카메라 시작 (윈도우 표시 후)
    # ─────────────────────────────────────────────────────────────────
    if camera_thread:
        camera_thread.start()

    # ─────────────────────────────────────────────────────────────────
    # 상태바 초기 업데이트 (1 s 뒤 — UI 렌더링 완료 후)
    # ─────────────────────────────────────────────────────────────────
    def _update_status_bar() -> None:
        window.set_ollama_status(ollama_ok, ollama_msg)
        window.set_model_status(model_loaded, model_label)
        if _import_errors:
            brief = _import_errors[0][:60]
            window.statusBar().showMessage(
                f"⚠ 일부 모듈 로드 실패: {brief}", 8000
            )
            print("[main] 모듈 로드 오류 목록:")
            for err in _import_errors:
                print(f"  • {err}")

    QTimer.singleShot(1000, _update_status_bar)

    # ─────────────────────────────────────────────────────────────────
    # 종료 시 정리
    # ─────────────────────────────────────────────────────────────────
    def _on_quit() -> None:
        if camera_thread and camera_thread.isRunning():
            camera_thread.stop()
        if mediapipe_worker and mediapipe_worker.isRunning():
            mediapipe_worker.stop()
        if stt_engine and stt_engine.isRunning():
            stt_engine.stop()
        if tts_engine and tts_engine.isRunning():
            tts_engine.stop()
        if mediapipe_engine:
            try:
                mediapipe_engine.release()
            except Exception:
                pass

    app.aboutToQuit.connect(_on_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
