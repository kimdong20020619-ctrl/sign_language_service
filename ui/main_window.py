import time

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QComboBox, QPushButton, QLabel, QApplication, QMenuBar,
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont, QAction

from ui.camera_widget import CameraWidget
from ui.chat_panel import ChatPanel
from ui.quick_buttons import QuickButtonsPanel


class MainWindow(QMainWindow):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

        # 외부 모듈 참조 (set_modules로 주입)
        self.camera_thread      = None
        self.mediapipe_engine   = None
        self.gesture_classifier = None
        self.sentence_builder   = None
        self.ollama_bridge      = None
        self.stt_engine         = None
        self.tts_engine         = None
        self.db                 = None
        self.context_manager    = None
        self._last_non_manual: dict = {}   # latest non-manual signals from classifier
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        self._active_workers: list = []    # Ollama _StreamWorker 참조 유지 (GC 방지)

        self.setWindowTitle("🤟 수화 소통 서비스")
        self.setMinimumSize(1280, 800)
        self._setup_ui()
        self._apply_theme()

    # ──────────────────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        self._build_menubar()

        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_topbar())

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(12, 10, 12, 10)
        content_layout.setSpacing(12)

        content_layout.addWidget(self._build_left_panel(), stretch=1)
        right = self._build_right_panel()
        right.setFixedWidth(460)
        content_layout.addWidget(right)

        outer.addWidget(content, stretch=1)

        self.statusBar().showMessage("준비 완료  |  카메라 시작 중...")
        self.statusBar().setStyleSheet(
            "background: #12121E; color: #607090; font-size: 11px;"
            "border-top: 1px solid #2A2A3E;"
        )
        self._build_status_indicators()

        # 시그널
        self.chat_panel.message_sent.connect(self._on_hearing_message)
        self.chat_panel.sentence_confirmed.connect(self._on_sentence_confirmed)
        self.quick_buttons.phrase_selected.connect(self._on_quick_phrase)

    # ── 메뉴바 ────────────────────────────────────────────────────────
    def _build_menubar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("파일(&F)")
        act_txt = QAction("대화 내보내기 (TXT)", self)
        act_pdf = QAction("대화 내보내기 (PDF)", self)
        act_clear = QAction("대화 기록 지우기", self)
        act_quit = QAction("종료", self)

        act_txt.triggered.connect(lambda: self._export_history("txt"))
        act_pdf.triggered.connect(lambda: self._export_history("pdf"))
        act_clear.triggered.connect(self._clear_history)
        act_quit.triggered.connect(QApplication.quit)

        file_menu.addAction(act_txt)
        file_menu.addAction(act_pdf)
        file_menu.addSeparator()
        file_menu.addAction(act_clear)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        settings_menu = mb.addMenu("설정(&S)")
        act_settings = QAction("환경 설정...", self)
        act_settings.triggered.connect(self._open_settings)
        settings_menu.addAction(act_settings)

    # ── 상단 바 ───────────────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(54)
        bar.setStyleSheet(
            "background-color: #161b22; border-bottom: 1px solid #1e2a3a;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)

        title = QLabel("🤟  수화 소통 서비스")
        title.setFont(QFont("맑은 고딕", 13, QFont.Bold))
        title.setStyleSheet("color: #00c8ff; background: transparent;")

        mode_label = QLabel("환경 모드:")
        mode_label.setStyleSheet("color: #7090b0; background: transparent;")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["카페", "병원", "은행", "학교", "마트", "직장"])
        self.mode_combo.setCurrentText(
            self.config.get("app", {}).get("default_mode", "카페")
        )
        self.mode_combo.setFixedWidth(110)
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e2a4a; color: #c0d4f0;
                border: 1px solid #2a3a6a; border-radius: 6px;
                padding: 4px 10px; font-size: 13px;
            }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox QAbstractItemView {
                background-color: #1e2a4a; color: #c0d4f0;
                selection-background-color: #2a4a8a;
            }
        """)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        settings_btn = QPushButton("⚙ 설정")
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e2a4a; color: #c0d0e8;
                border: 1px solid #2a3a6a; border-radius: 6px;
                padding: 5px 12px;
            }
            QPushButton:hover { background-color: #263460; }
        """)
        settings_btn.clicked.connect(self._open_settings)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_combo)
        layout.addSpacing(8)
        layout.addWidget(settings_btn)
        return bar

    # ── 왼쪽 패널 (카메라) ───────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        self.camera_widget = CameraWidget()
        layout.addWidget(self.camera_widget, stretch=1)

        ctrl = QHBoxLayout()
        self.toggle_cam_btn = QPushButton("카메라 중지")
        self.toggle_cam_btn.setStyleSheet(self._btn("#8b2222"))
        self.toggle_cam_btn.clicked.connect(self._toggle_camera)

        self.capture_btn = QPushButton("현재 문장 전송")
        self.capture_btn.setStyleSheet(self._btn("#1a6a3a"))
        self.capture_btn.clicked.connect(self._capture_sentence)

        self.undo_btn = QPushButton("마지막 단어 취소")
        self.undo_btn.setStyleSheet(self._btn("#4a2a7a"))
        self.undo_btn.clicked.connect(self._undo_last_word)

        ctrl.addWidget(self.toggle_cam_btn)
        ctrl.addWidget(self.capture_btn)
        ctrl.addWidget(self.undo_btn)
        layout.addLayout(ctrl)
        return w

    # ── 오른쪽 패널 (채팅 + 퀵버튼) ─────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chat_panel = ChatPanel()
        self.quick_buttons = QuickButtonsPanel()
        self.quick_buttons.setMaximumHeight(250)

        layout.addWidget(self.chat_panel, stretch=1)
        layout.addWidget(self.quick_buttons)
        return w

    # ──────────────────────────────────────────────────────────────────
    def set_modules(self, camera_thread, mediapipe_engine, gesture_classifier,
                    sentence_builder, ollama_bridge, tts_engine, db,
                    context_manager, stt_engine=None,
                    mediapipe_worker=None) -> None:
        self.camera_thread       = camera_thread
        self.mediapipe_engine    = mediapipe_engine
        self.gesture_classifier  = gesture_classifier
        self.sentence_builder    = sentence_builder
        self.ollama_bridge       = ollama_bridge
        self.tts_engine          = tts_engine
        self.db                  = db
        self.context_manager     = context_manager
        self.stt_engine          = stt_engine

        if camera_thread:
            camera_thread.error_occurred.connect(self._on_camera_error)
            if mediapipe_worker:
                # MediaPipe는 별도 스레드: 카메라 → worker → 처리된 프레임 → UI
                camera_thread.frame_ready.connect(mediapipe_worker.submit_frame)
                mediapipe_worker.frame_processed.connect(self._on_worker_frame)
                camera_thread.frame_ready.connect(self._on_raw_frame_fps)
            else:
                camera_thread.frame_ready.connect(self._on_raw_frame)

        # TTS 요청 연결 (🔊 버튼)
        self.chat_panel.tts_requested.connect(
            lambda text: tts_engine.speak(text) if tts_engine else None
        )

        # STT 토글 연결 (🎤 버튼)
        self.chat_panel.stt_toggle.connect(self._on_stt_toggle)

        # Whisper 인식 결과 → 입력창
        if stt_engine:
            stt_engine.text_recognized.connect(self.chat_panel.set_input_text)
            stt_engine.listening_state.connect(self.chat_panel.set_stt_active)
            stt_engine.error_occurred.connect(
                lambda msg: self.statusBar().showMessage(f"STT: {msg}", 4000)
            )

    # ──────────────────────────────────────────────────────────────────
    # ── 하단 상태바 영구 인디케이터 ──────────────────────────────────────
    def _build_status_indicators(self) -> None:
        sb = self.statusBar()

        def _perm(text: str, color: str = "#607090") -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color:{color}; font-size:11px; padding:0 12px 0 4px;"
                "background:transparent;"
            )
            return lbl

        sep_style = "color:#2A3A5A; font-size:11px; background:transparent;"

        self._ollama_lbl = _perm("● Ollama: 확인 중...", "#f0a020")
        sep1 = QLabel("│"); sep1.setStyleSheet(sep_style)
        self._fps_lbl_sb = _perm("📷 FPS: --")
        sep2 = QLabel("│"); sep2.setStyleSheet(sep_style)
        self._model_lbl  = _perm("🧠 모델: 확인 중...", "#f0a020")

        sb.addPermanentWidget(self._ollama_lbl)
        sb.addPermanentWidget(sep1)
        sb.addPermanentWidget(self._fps_lbl_sb)
        sb.addPermanentWidget(sep2)
        sb.addPermanentWidget(self._model_lbl)

    def set_ollama_status(self, connected: bool, msg: str = "") -> None:
        text  = f"● Ollama: {msg or '연결됨'}" if connected else f"● Ollama: {msg or '미연결'}"
        color = "#30c060" if connected else "#e05050"
        self._ollama_lbl.setText(text)
        self._ollama_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; padding:0 12px 0 4px; background:transparent;"
        )

    def set_claude_status(self, connected: bool) -> None:
        self.set_ollama_status(connected)

    def set_model_status(self, loaded: bool, name: str = "") -> None:
        label = name or "제스처 모델"
        text  = f"🧠 {label}: 로드완료" if loaded else f"🧠 {label}: 미로드"
        color = "#30c060" if loaded else "#f0a020"
        self._model_lbl.setText(text)
        self._model_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; padding:0 12px 0 4px; background:transparent;"
        )

    def update_fps(self, fps: float) -> None:
        self._fps_lbl_sb.setText(f"📷 {fps:.0f} fps")

    # ──────────────────────────────────────────────────────────────────
    @Slot(np.ndarray)
    def _on_raw_frame_fps(self, frame: np.ndarray) -> None:
        """카메라 원본 프레임 — FPS 측정 전용 (MediaPipe worker 사용 시)."""
        self._fps_count += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            self.update_fps(self._fps_count / (now - self._fps_t0))
            self._fps_count = 0
            self._fps_t0 = now

    @Slot(np.ndarray, object)
    def _on_worker_frame(self, annotated: np.ndarray, frame_data) -> None:
        """MediaPipe worker 스레드에서 처리 완료된 프레임 수신."""
        self._handle_frame_result(annotated, frame_data)

    @Slot(np.ndarray)
    def _on_raw_frame(self, frame: np.ndarray) -> None:
        """worker 없을 때 — 메인 스레드에서 직접 MediaPipe 처리."""
        self._fps_count += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            self.update_fps(self._fps_count / (now - self._fps_t0))
            self._fps_count = 0
            self._fps_t0 = now

        if self.mediapipe_engine:
            annotated, frame_data = self.mediapipe_engine.process_frame(frame)
        else:
            annotated, frame_data = frame, None
        self._handle_frame_result(annotated, frame_data)

    def _handle_frame_result(self, annotated: np.ndarray, frame_data) -> None:
        """공통 처리: 제스처 분류 → 오버레이 → UI 업데이트."""
        word, conf = "", 0.0
        has_right = False
        has_left  = False

        if self.gesture_classifier:
            result       = self.gesture_classifier.classify(frame_data)
            word         = result.get("word") or ""
            conf         = result.get("confidence", 0.0)
            sentence_end = result.get("sentence_end", False)
            non_manual   = result.get("non_manual", {})
            if non_manual:
                self._last_non_manual = non_manual

            if frame_data:
                m = frame_data.get("manual", {})
                has_right = m.get("right_hand") is not None
                has_left  = m.get("left_hand")  is not None

            if self.sentence_builder:
                # 신뢰도 임계값 이상 확정 단어만 문장에 추가
                confirmed_word = word if result.get("is_confirmed") else None
                added = self.sentence_builder.add_prediction(confirmed_word)
                if added:
                    self.chat_panel.update_raw_words(
                        self.sentence_builder.get_sentence()
                    )

            if (sentence_end and self.sentence_builder
                    and not self.sentence_builder.is_empty):
                sentence = self.sentence_builder.get_sentence()
                self.chat_panel.update_recognized_sentence(sentence)
                self._on_sentence_confirmed(sentence)

        if word and self.mediapipe_engine:
            self.mediapipe_engine.draw_word_overlay(annotated, word, conf)

        self.camera_widget.set_recognition_result(word or "", conf)
        self.camera_widget.set_hand_status(has_right, has_left)

        if self._last_non_manual:
            nm    = self._last_non_manual
            stype = nm.get("sentence_type", "neutral")
            expr  = (stype if stype in ("question", "negative", "emphasis")
                     else nm.get("emotion", "neutral"))
            self.camera_widget.set_expression(expr)

        self.camera_widget.update_frame(annotated)

    def _on_mode_changed(self, mode: str) -> None:
        self.quick_buttons.set_mode(mode)
        if self.ollama_bridge:
            self.ollama_bridge.set_mode(mode)
        if self.context_manager:
            self.context_manager.set_mode(mode)
        self.statusBar().showMessage(f"환경 모드 변경: {mode}")

    def _on_hearing_message(self, text: str) -> None:
        self.chat_panel.add_bubble(f"{text}", is_deaf=False)
        if self.context_manager:
            self.context_manager.add_hearing_message(text)
        if self.db and self.context_manager:
            self.db.save_message("hearing", text, self.context_manager.current_mode)

    def _on_sentence_confirmed(self, sentence: str) -> None:
        """수화 문장 확정 → Ollama 스트리밍 교정 → ChatPanel 실시간 표시 → TTS → DB."""
        # 단어 리스트 확보 (OllamaBridge는 list[str] 입력 요구)
        raw_words = (
            self.sentence_builder.get_words()
            if self.sentence_builder and not self.sentence_builder.is_empty
            else sentence.split()
        )
        raw_sentence = sentence
        nm = self._last_non_manual or {}
        self._last_non_manual = {}
        self.chat_panel.clear_recognized()

        if self.ollama_bridge and raw_words:
            # 스트리밍 버블 시작
            self.chat_panel.begin_streaming_bubble()
            try:
                mode = (self.context_manager.current_mode
                        if self.context_manager else None)
                worker = self.ollama_bridge.correct_sentence_stream(
                    words=raw_words, mode=mode, non_manual=nm
                )
                self._active_workers.append(worker)

                worker.text_chunk.connect(self.chat_panel.append_token)
                worker.sentence_complete.connect(
                    lambda full: self._on_llm_complete(full, raw_sentence)
                )
                worker.error_occurred.connect(
                    lambda e: self.statusBar().showMessage(f"Ollama: {e}", 5000)
                )
                worker.finished.connect(
                    lambda: self._active_workers.remove(worker)
                    if worker in self._active_workers else None
                )
                worker.start()
            except Exception as e:
                self.statusBar().showMessage(f"Ollama 오류: {e}", 5000)
                self.chat_panel.finalize_bubble(sentence)
                self._finalize_sentence(sentence, raw_sentence)
        else:
            # Ollama 없음 → 원문 그대로
            self.chat_panel.add_bubble(sentence, is_deaf=True)
            self._finalize_sentence(sentence, raw_sentence)

        if self.sentence_builder:
            self.sentence_builder.clear()

    def _on_llm_complete(self, corrected: str, raw: str) -> None:
        """Ollama 스트리밍 완료 → TTS + DB 저장."""
        self.chat_panel.finalize_bubble(corrected)
        self._finalize_sentence(corrected, raw)

    def _finalize_sentence(self, sentence: str, raw: str) -> None:
        """TTS 재생 + 컨텍스트·DB 저장."""
        try:
            if self.tts_engine:
                self.tts_engine.speak(sentence)
        except Exception as e:
            self.statusBar().showMessage(f"TTS 오류: {e}", 3000)
        try:
            if self.context_manager:
                self.context_manager.add_deaf_message(sentence, raw_words=raw)
            if self.db and self.context_manager:
                self.db.save_message(
                    "deaf", sentence,
                    self.context_manager.current_mode,
                    raw_sentence=raw,
                )
        except Exception as e:
            self.statusBar().showMessage(f"DB 저장 오류: {e}", 3000)

    def _on_stt_toggle(self) -> None:
        """🎤 버튼 → WhisperEngine 활성/비활성 토글."""
        if not self.stt_engine:
            self.statusBar().showMessage("STT 엔진이 초기화되지 않았습니다.", 3000)
            return
        new_state = not self.stt_engine._active
        self.stt_engine.set_active(new_state)
        state_txt = "시작" if new_state else "중지"
        self.statusBar().showMessage(f"음성 입력 {state_txt}", 2000)

    def _on_quick_phrase(self, phrase: str) -> None:
        self._on_hearing_message(phrase)

    def _toggle_camera(self) -> None:
        if not self.camera_thread:
            return
        if self.camera_thread.is_active:
            self.camera_thread.stop()
            self.toggle_cam_btn.setText("카메라 시작")
        else:
            self.camera_thread.start()
            self.toggle_cam_btn.setText("카메라 중지")

    def _capture_sentence(self) -> None:
        if self.sentence_builder and not self.sentence_builder.is_empty:
            raw = self.sentence_builder.get_sentence()
            self.chat_panel.update_recognized_sentence(raw)
            self._on_sentence_confirmed(raw)

    def _undo_last_word(self) -> None:
        if self.sentence_builder:
            self.sentence_builder.remove_last_word()
            self.chat_panel.update_raw_words(self.sentence_builder.get_sentence())

    def _export_history(self, fmt: str) -> None:
        if self.db:
            path = self.db.export(fmt)
            self.statusBar().showMessage(f"내보내기 완료: {path}")

    def _clear_history(self) -> None:
        if self.db:
            self.db.clear_all()
        if self.context_manager:
            self.context_manager.clear()
        self.statusBar().showMessage("대화 기록이 삭제되었습니다.")

    def _open_settings(self) -> None:
        from ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self.config, self)
        dlg.settings_saved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self, new_cfg: dict) -> None:
        self.config.update(new_cfg)
        tts_cfg = new_cfg.get("tts", {})
        if self.tts_engine:
            if hasattr(self.tts_engine, "set_rate"):
                self.tts_engine.set_rate(tts_cfg.get("rate", 150))
            if hasattr(self.tts_engine, "set_volume"):
                self.tts_engine.set_volume(tts_cfg.get("volume", 1.0))
            if hasattr(self.tts_engine, "set_language"):
                self.tts_engine.set_language(tts_cfg.get("language", "ko"))
            if hasattr(self.tts_engine, "set_slow"):
                self.tts_engine.set_slow(tts_cfg.get("slow", False))
        stt_cfg = new_cfg.get("stt", {})
        if self.stt_engine:
            if hasattr(self.stt_engine, "set_language"):
                self.stt_engine.set_language(stt_cfg.get("language", "ko"))
        if self.ollama_bridge:
            self.ollama_bridge.set_mode(
                new_cfg.get("ui", {}).get("default_mode", "카페")
            )
        self.statusBar().showMessage("설정이 저장되었습니다.")

    def _on_camera_error(self, msg: str) -> None:
        self.camera_widget.show_no_camera()
        self.statusBar().showMessage(f"카메라 오류: {msg}")


    # ──────────────────────────────────────────────────────────────────
    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E2E;
                color: #c8d4e0;
                font-family: '맑은 고딕', sans-serif;
            }
            QMenuBar {
                background-color: #161b22;
                color: #c0cce0;
                border-bottom: 1px solid #1e2a3a;
            }
            QMenuBar::item:selected { background-color: #1e2a4a; }
            QMenu {
                background-color: #161b22;
                color: #c0cce0;
                border: 1px solid #1e2a3a;
            }
            QMenu::item:selected { background-color: #1e2a4a; }
            QStatusBar { background-color: #161b22; color: #607080; }
        """)

    @staticmethod
    def _btn(color: str) -> str:
        return (
            f"QPushButton {{ background-color: {color}; color: #e0e8f0; "
            f"border: none; border-radius: 7px; padding: 7px 14px; "
            f"font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {color}cc; }}"
        )

    def closeEvent(self, event) -> None:
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        if self.mediapipe_engine:
            self.mediapipe_engine.release()
        if self.stt_engine and self.stt_engine.isRunning():
            self.stt_engine.stop()
        if self.tts_engine:
            self.tts_engine.stop()
        # 진행 중인 Ollama worker 정리
        for w in list(self._active_workers):
            if w.isRunning():
                w.terminate()
                w.wait(1000)
        event.accept()
