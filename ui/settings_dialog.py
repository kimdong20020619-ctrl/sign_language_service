"""ui/settings_dialog.py — 환경 설정 다이얼로그 (탭 4개)"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

_DIALOG_STYLE = """
    QDialog, QWidget, QTabWidget::pane {{
        background:#1E1E2E; color:#FFFFFF;
    }}
    QTabBar::tab {{
        background:#2A2A3E; color:#8888AA;
        padding:7px 18px; border-radius:5px 5px 0 0;
        margin-right:2px;
    }}
    QTabBar::tab:selected {{ background:#6C63FF; color:#FFFFFF; }}
    QGroupBox {{
        border:1px solid #3A3A5E; border-radius:8px;
        margin-top:10px; padding:12px 8px 8px 8px;
        color:#8888AA; font-weight:bold;
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:10px; }}
    QLabel           {{ background:transparent; color:#C0C0D8; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background:#14141E; color:#FFFFFF;
        border:1px solid #3A3A5E; border-radius:6px;
        padding:5px 8px; min-width:160px;
    }}
    QComboBox QAbstractItemView {{
        background:#1E1E2E; color:#FFFFFF;
        selection-background-color:#6C63FF;
    }}
    QCheckBox {{ color:#C0C0D8; }}
    QCheckBox::indicator {{
        width:16px; height:16px; border-radius:3px;
        border:1px solid #3A3A5E; background:#14141E;
    }}
    QCheckBox::indicator:checked {{ background:#6C63FF; border-color:#6C63FF; }}
    QPushButton {{
        background:#6C63FF; color:#FFFFFF;
        border:none; border-radius:7px;
        padding:8px 22px; font-weight:bold;
    }}
    QPushButton:hover   {{ background:#7C73FF; }}
    QPushButton:pressed {{ background:#5C53EF; }}
    QPushButton#cancel  {{ background:#2A2A3E; color:#8888AA; }}
    QPushButton#cancel:hover {{ background:#3A3A5E; color:#FFFFFF; }}
"""


class SettingsDialog(QDialog):
    """
    탭 구성:
      ① AI 모델   — Ollama URL, 모델명, 타임아웃
      ② 카메라    — 인덱스, 해상도, FPS
      ③ 음성      — TTS(slow/fast), STT(모델, 활성화)
      ④ 화면      — 기본 모드, 폰트 크기, 테마
    """

    settings_saved: Signal = Signal(dict)

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        # 딥 카피
        self._cfg = json.loads(json.dumps(config))
        self.setWindowTitle("⚙  환경 설정")
        self.setMinimumWidth(460)
        self.setStyleSheet(_DIALOG_STYLE)
        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._tab_ai(),     "🤖  AI 모델")
        self._tabs.addTab(self._tab_camera(), "📷  카메라")
        self._tabs.addTab(self._tab_voice(),  "🔊  음성")
        self._tabs.addTab(self._tab_ui(),     "🖥️  화면")
        root.addWidget(self._tabs)
        root.addLayout(self._build_btn_row())

    # ── 탭 ①: AI 모델 ────────────────────────────────────────────────

    def _tab_ai(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        g = QGroupBox("Ollama (로컬 LLM)")
        form = QFormLayout(g)

        oc = self._cfg.get("ollama", {})

        self._ollama_url = QLineEdit(oc.get("base_url", "http://localhost:11434"))
        self._ollama_url.setPlaceholderText("http://localhost:11434")

        self._ollama_model = QLineEdit(oc.get("model", "exaone3.5"))
        self._ollama_model.setPlaceholderText("exaone3.5 / llama3.2")

        self._ollama_timeout = QSpinBox()
        self._ollama_timeout.setRange(5, 120)
        self._ollama_timeout.setValue(oc.get("timeout", 30))
        self._ollama_timeout.setSuffix(" 초")

        form.addRow("서버 URL:", self._ollama_url)
        form.addRow("모델명:",   self._ollama_model)
        form.addRow("타임아웃:", self._ollama_timeout)

        g2 = QGroupBox("수화 인식 모델")
        form2 = QFormLayout(g2)
        mc = self._cfg.get("model", {})

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.3, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(mc.get("confidence_threshold", 0.75))

        self._confirm_frames = QSpinBox()
        self._confirm_frames.setRange(5, 60)
        self._confirm_frames.setValue(mc.get("confirm_frames", 15))
        self._confirm_frames.setSuffix(" 프레임")

        self._word_gap = QDoubleSpinBox()
        self._word_gap.setRange(0.5, 5.0)
        self._word_gap.setSingleStep(0.1)
        self._word_gap.setDecimals(1)
        self._word_gap.setValue(mc.get("word_gap_seconds", 1.5))
        self._word_gap.setSuffix(" 초")

        form2.addRow("신뢰도 임계값:", self._conf_spin)
        form2.addRow("확정 프레임 수:", self._confirm_frames)
        form2.addRow("단어 간격:",     self._word_gap)

        vbox.addWidget(g)
        vbox.addWidget(g2)
        vbox.addStretch()
        return w

    # ── 탭 ②: 카메라 ────────────────────────────────────────────────

    def _tab_camera(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        g = QGroupBox("카메라 설정")
        form = QFormLayout(g)
        cc = self._cfg.get("camera", {})

        self._cam_idx = QSpinBox()
        self._cam_idx.setRange(0, 9)
        self._cam_idx.setValue(cc.get("index", 0))

        self._cam_w = QSpinBox()
        self._cam_w.setRange(320, 1920)
        self._cam_w.setSingleStep(160)
        self._cam_w.setValue(cc.get("width", 1280))
        self._cam_w.setSuffix(" px")

        self._cam_h = QSpinBox()
        self._cam_h.setRange(240, 1080)
        self._cam_h.setSingleStep(120)
        self._cam_h.setValue(cc.get("height", 720))
        self._cam_h.setSuffix(" px")

        self._cam_fps = QSpinBox()
        self._cam_fps.setRange(10, 60)
        self._cam_fps.setValue(cc.get("fps", 30))
        self._cam_fps.setSuffix(" fps")

        form.addRow("카메라 인덱스:", self._cam_idx)
        form.addRow("가로 해상도:",   self._cam_w)
        form.addRow("세로 해상도:",   self._cam_h)
        form.addRow("FPS:",          self._cam_fps)

        g2 = QGroupBox("MediaPipe")
        form2 = QFormLayout(g2)
        mp = self._cfg.get("mediapipe", {}).get("hands", {})

        self._max_hands = QSpinBox()
        self._max_hands.setRange(1, 2)
        self._max_hands.setValue(mp.get("max_num_hands", 2))

        self._mp_det_conf = QDoubleSpinBox()
        self._mp_det_conf.setRange(0.3, 1.0)
        self._mp_det_conf.setSingleStep(0.05)
        self._mp_det_conf.setDecimals(2)
        self._mp_det_conf.setValue(mp.get("min_detection_confidence", 0.7))

        form2.addRow("최대 손 수:",    self._max_hands)
        form2.addRow("감지 신뢰도:",   self._mp_det_conf)

        vbox.addWidget(g)
        vbox.addWidget(g2)
        vbox.addStretch()
        return w

    # ── 탭 ③: 음성 ──────────────────────────────────────────────────

    def _tab_voice(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        # TTS
        g_tts = QGroupBox("TTS — gTTS (음성 출력)")
        form_tts = QFormLayout(g_tts)
        tc = self._cfg.get("tts", {})

        self._tts_lang = QComboBox()
        self._tts_lang.addItems(["ko", "en", "ja", "zh"])
        self._tts_lang.setCurrentText(tc.get("language", "ko"))

        self._tts_slow = QCheckBox("천천히 읽기")
        self._tts_slow.setChecked(tc.get("slow", False))

        form_tts.addRow("언어:",     self._tts_lang)
        form_tts.addRow("",         self._tts_slow)

        # STT
        g_stt = QGroupBox("STT — Whisper (음성 입력)")
        form_stt = QFormLayout(g_stt)
        sc = self._cfg.get("stt", {})

        self._stt_model = QComboBox()
        self._stt_model.addItems(["tiny", "base", "small", "medium", "large"])
        self._stt_model.setCurrentText(sc.get("model", "base"))

        self._stt_lang = QComboBox()
        self._stt_lang.addItems(["ko", "en", "ja", "zh", "auto"])
        self._stt_lang.setCurrentText(sc.get("language", "ko"))

        self._stt_enabled = QCheckBox("STT 활성화")
        self._stt_enabled.setChecked(sc.get("enabled", True))

        form_stt.addRow("Whisper 모델:", self._stt_model)
        form_stt.addRow("인식 언어:",    self._stt_lang)
        form_stt.addRow("",             self._stt_enabled)

        vbox.addWidget(g_tts)
        vbox.addWidget(g_stt)
        vbox.addStretch()
        return w

    # ── 탭 ④: 화면 ──────────────────────────────────────────────────

    def _tab_ui(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        g = QGroupBox("UI 설정")
        form = QFormLayout(g)
        uc = self._cfg.get("ui", {})

        self._default_mode = QComboBox()
        self._default_mode.addItems(["카페", "병원", "은행", "학교", "마트", "직장"])
        self._default_mode.setCurrentText(uc.get("default_mode", "카페"))

        self._font_result = QSpinBox()
        self._font_result.setRange(16, 48)
        self._font_result.setValue(uc.get("font_size_result", 28))
        self._font_result.setSuffix(" px")

        self._font_chat = QSpinBox()
        self._font_chat.setRange(10, 24)
        self._font_chat.setValue(uc.get("font_size_chat", 16))
        self._font_chat.setSuffix(" px")

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "high_contrast"])
        self._theme_combo.setCurrentText(uc.get("theme", "dark"))

        form.addRow("기본 환경 모드:", self._default_mode)
        form.addRow("결과 폰트 크기:", self._font_result)
        form.addRow("채팅 폰트 크기:", self._font_chat)
        form.addRow("테마:",           self._theme_combo)

        vbox.addWidget(g)
        vbox.addStretch()
        return w

    # ── 저장/취소 ─────────────────────────────────────────────────────

    def _build_btn_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        save_btn   = QPushButton("💾  저장")
        cancel_btn = QPushButton("취소")
        cancel_btn.setObjectName("cancel")
        save_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self.reject)
        row.addStretch()
        row.addWidget(save_btn)
        row.addWidget(cancel_btn)
        return row

    def _on_save(self) -> None:
        self._cfg.setdefault("ollama", {}).update({
            "base_url": self._ollama_url.text().strip(),
            "model":    self._ollama_model.text().strip(),
            "timeout":  self._ollama_timeout.value(),
        })
        self._cfg.setdefault("camera", {}).update({
            "index":  self._cam_idx.value(),
            "width":  self._cam_w.value(),
            "height": self._cam_h.value(),
            "fps":    self._cam_fps.value(),
        })
        self._cfg.setdefault("mediapipe", {}).setdefault("hands", {}).update({
            "max_num_hands":           self._max_hands.value(),
            "min_detection_confidence": self._mp_det_conf.value(),
        })
        self._cfg.setdefault("model", {}).update({
            "confidence_threshold": self._conf_spin.value(),
            "confirm_frames":       self._confirm_frames.value(),
            "word_gap_seconds":     self._word_gap.value(),
        })
        self._cfg.setdefault("tts", {}).update({
            "language": self._tts_lang.currentText(),
            "slow":     self._tts_slow.isChecked(),
        })
        self._cfg.setdefault("stt", {}).update({
            "model":    self._stt_model.currentText(),
            "language": self._stt_lang.currentText(),
            "enabled":  self._stt_enabled.isChecked(),
        })
        self._cfg.setdefault("ui", {}).update({
            "default_mode":    self._default_mode.currentText(),
            "font_size_result": self._font_result.value(),
            "font_size_chat":   self._font_chat.value(),
            "theme":            self._theme_combo.currentText(),
        })

        # config.json 저장
        cfg_path = Path(__file__).resolve().parent.parent / "config.json"
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self._cfg, f, ensure_ascii=False, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "저장 실패", str(e))
            return

        self.settings_saved.emit(self._cfg)
        self.accept()
