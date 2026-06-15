"""ui/chat_panel.py — 수화 변환 결과 표시 + 대화 말풍선 + 직원 입력 영역"""

from __future__ import annotations

import pyperclip  # pip install pyperclip  (선택; 없으면 복사 버튼 비활성)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QTextEdit, QVBoxLayout,
    QWidget,
)

_BG     = "#1E1E2E"
_CARD   = "#2A2A3E"
_ACCENT = "#6C63FF"
_RESULT = "#00FF88"
_TEXT   = "#FFFFFF"
_DIM    = "#8888AA"

# 말풍선 색
_DEAF_BG   = "#1A2E1A"   # 왼쪽 (청각장애인)
_DEAF_FG   = "#00FF88"
_HEAR_BG   = "#1E1A2E"   # 오른쪽 (직원/상대방)
_HEAR_FG   = "#B0A0FF"


def _btn(bg: str, hover: str = "") -> str:
    hv = hover or bg + "BB"
    return (
        f"QPushButton {{ background:{bg}; color:#FFFFFF; border:none;"
        f" border-radius:7px; padding:6px 14px; font-size:12px;"
        f" font-weight:bold; }}"
        f"QPushButton:hover {{ background:{hv}; }}"
        f"QPushButton:pressed {{ background:{bg}88; }}"
    )


# ══════════════════════════════════════════════════════════════════════
# 말풍선 위젯
# ══════════════════════════════════════════════════════════════════════

class _Bubble(QFrame):
    def __init__(self, text: str, is_deaf: bool,
                 font_size: int = 13, high_contrast: bool = False) -> None:
        super().__init__()
        self._is_deaf = is_deaf
        self._lbl     = QLabel(text)
        self._lbl.setWordWrap(True)
        self._lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if is_deaf:
            layout.addWidget(self._lbl)
            layout.addStretch()
        else:
            layout.addStretch()
            layout.addWidget(self._lbl)

        self.apply_style(font_size, high_contrast)

    def apply_style(self, font_size: int, high_contrast: bool) -> None:
        if high_contrast:
            if self._is_deaf:
                bg, fg = "#000000", "#FFFF00"
            else:
                bg, fg = "#000033", "#FFFFFF"
        else:
            if self._is_deaf:
                bg, fg = _DEAF_BG, _DEAF_FG
            else:
                bg, fg = _HEAR_BG, _HEAR_FG

        self._lbl.setStyleSheet(
            f"QLabel {{ background:{bg}; color:{fg}; border-radius:10px;"
            f" padding:10px 14px; font-size:{font_size}px;"
            f" font-family:'맑은 고딕'; max-width:320px; }}"
        )
        self._lbl.setFont(QFont("맑은 고딕", font_size))


# ══════════════════════════════════════════════════════════════════════
# ChatPanel
# ══════════════════════════════════════════════════════════════════════

class ChatPanel(QWidget):
    """
    오른쪽 패널 전체:
      ① 변환 문장 표시 (대형, 녹색)
      ② [🔊 음성재생] [📋 복사]
      ③ 대화 말풍선 스크롤 영역
      ④ 직원 답변 입력 + [🎤 음성입력] [📤 전송]
      ⑤ 폰트 크기 슬라이더 + 고대비 모드

    시그널:
      message_sent(str)       — 직원이 텍스트 전송
      sentence_confirmed(str) — 수화 문장 확정 (TTS+저장)
      tts_requested(str)      — 🔊 버튼 클릭
      stt_toggle()            — 🎤 버튼 클릭 (STT 토글)
    """

    message_sent:       Signal = Signal(str)
    sentence_confirmed: Signal = Signal(str)
    tts_requested:      Signal = Signal(str)
    stt_toggle:         Signal = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._font_size     = 16
        self._high_contrast = False
        self._last_sentence = ""
        self._streaming_bubble: _Bubble | None = None
        self._bubbles: list[_Bubble] = []
        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        root.addWidget(self._build_result_area())
        root.addWidget(self._build_conv_area(), stretch=1)
        root.addWidget(self._build_input_area())
        root.addWidget(self._build_toolbar())

    # ── ① 변환 문장 표시 ─────────────────────────────────────────────

    def _build_result_area(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{_CARD}; border-radius:10px; padding:2px;"
        )
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(12, 8, 12, 8)
        vbox.setSpacing(6)

        # 제목
        title = QLabel("💬  변환된 문장")
        title.setStyleSheet(f"color:{_DIM}; font-size:11px; background:transparent;")
        vbox.addWidget(title)

        # 결과 라벨 (대형)
        self._result_lbl = QLabel("수화를 시작하면 여기에 표시됩니다")
        self._result_lbl.setFont(QFont("맑은 고딕", 20, QFont.Bold))
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setAlignment(Qt.AlignCenter)
        self._result_lbl.setMinimumHeight(80)
        self._result_lbl.setStyleSheet(
            f"color:#3A4A5A; padding:8px; background:transparent;"
            f" border:1px solid #2A2A3E; border-radius:8px;"
        )
        vbox.addWidget(self._result_lbl)

        # 누적 단어
        self._raw_lbl = QLabel("")
        self._raw_lbl.setStyleSheet(
            f"color:{_DIM}; font-size:11px; background:transparent;"
        )
        vbox.addWidget(self._raw_lbl)

        # 버튼 행
        btn_row = QHBoxLayout()
        self._tts_btn = QPushButton("🔊  음성 재생")
        self._tts_btn.setStyleSheet(_btn("#2E4A2E", "#3A6A3A"))
        self._tts_btn.clicked.connect(self._on_tts)

        self._copy_btn = QPushButton("📋  복사")
        self._copy_btn.setStyleSheet(_btn("#2A2A4E", "#3A3A6E"))
        self._copy_btn.clicked.connect(self._on_copy)

        self._confirm_btn = QPushButton("✅  전송")
        self._confirm_btn.setStyleSheet(_btn("#1E3E1E", "#2A5A2A"))
        self._confirm_btn.clicked.connect(self._on_confirm)

        btn_row.addWidget(self._tts_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._confirm_btn)
        vbox.addLayout(btn_row)
        return frame

    # ── ② 대화 말풍선 스크롤 ────────────────────────────────────────

    def _build_conv_area(self) -> QScrollArea:
        self._conv_inner  = QWidget()
        self._conv_layout = QVBoxLayout(self._conv_inner)
        self._conv_layout.setAlignment(Qt.AlignTop)
        self._conv_layout.setSpacing(6)
        self._conv_layout.setContentsMargins(6, 6, 6, 6)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._conv_inner)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"""
            QScrollArea         {{ background:#14141E; border-radius:8px;
                                   border:1px solid {_CARD}; }}
            QScrollBar:vertical {{ background:#14141E; width:6px; border:none; }}
            QScrollBar::handle:vertical
                                {{ background:#3A3A5E; border-radius:3px; }}
        """)
        return self._scroll

    # ── ③ 직원 입력 영역 ────────────────────────────────────────────

    def _build_input_area(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{_CARD}; border-radius:10px; padding:2px;"
        )
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(10, 6, 10, 6)
        vbox.setSpacing(5)

        title = QLabel("⌨️  직원 답변 입력")
        title.setStyleSheet(f"color:{_DIM}; font-size:11px; background:transparent;")
        vbox.addWidget(title)

        row = QHBoxLayout()
        self._input_edit = QTextEdit()
        self._input_edit.setMaximumHeight(70)
        self._input_edit.setPlaceholderText("답변을 입력하세요...")
        self._input_edit.setStyleSheet(f"""
            QTextEdit {{
                background:#14141E; color:{_TEXT};
                border:1px solid #3A3A5E; border-radius:8px;
                padding:8px; font-size:13px; font-family:'맑은 고딕';
            }}
        """)
        # Ctrl+Enter → 전송
        self._input_edit.installEventFilter(self)

        btn_col = QVBoxLayout()
        self._stt_btn = QPushButton("🎤")
        self._stt_btn.setFixedSize(42, 30)
        self._stt_btn.setToolTip("음성 입력 (STT 토글)")
        self._stt_btn.setStyleSheet(_btn("#3A1A4A", "#5A2A6A"))
        self._stt_btn.clicked.connect(self.stt_toggle)

        self._send_btn = QPushButton("📤")
        self._send_btn.setFixedSize(42, 30)
        self._send_btn.setToolTip("전송 (Ctrl+Enter)")
        self._send_btn.setStyleSheet(_btn("#1A3A6A", "#2A4A8A"))
        self._send_btn.clicked.connect(self._on_send)

        btn_col.addWidget(self._stt_btn)
        btn_col.addWidget(self._send_btn)

        row.addWidget(self._input_edit)
        row.addLayout(btn_col)
        vbox.addLayout(row)
        return frame

    # ── ④ 툴바: 폰트 슬라이더 + 고대비 ─────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(f"background:{_CARD}; border-radius:6px;")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(8)

        fnt_lbl = QLabel("가")
        fnt_lbl.setStyleSheet(f"color:{_DIM}; font-size:11px;")
        row.addWidget(fnt_lbl)

        self._font_slider = QSlider(Qt.Horizontal)
        self._font_slider.setRange(12, 28)
        self._font_slider.setValue(self._font_size)
        self._font_slider.setFixedWidth(100)
        self._font_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background:#14141E; height:4px; border-radius:2px;
            }}
            QSlider::handle:horizontal {{
                background:{_ACCENT}; width:12px; height:12px;
                border-radius:6px; margin:-4px 0;
            }}
            QSlider::sub-page:horizontal {{
                background:{_ACCENT}; border-radius:2px;
            }}
        """)
        self._font_slider.valueChanged.connect(self._on_font_size)
        row.addWidget(self._font_slider)

        big_lbl = QLabel("가")
        big_lbl.setStyleSheet(f"color:{_DIM}; font-size:16px;")
        row.addWidget(big_lbl)

        row.addStretch()

        self._contrast_btn = QPushButton("◑  고대비")
        self._contrast_btn.setCheckable(True)
        self._contrast_btn.setFixedHeight(24)
        self._contrast_btn.setStyleSheet(f"""
            QPushButton           {{ background:#22223A; color:{_DIM};
                                    border:1px solid #3A3A5E; border-radius:5px;
                                    padding:2px 10px; font-size:11px; }}
            QPushButton:checked   {{ background:{_ACCENT}; color:#FFFFFF;
                                    border-color:{_ACCENT}; }}
            QPushButton:hover     {{ border-color:{_ACCENT}; }}
        """)
        self._contrast_btn.toggled.connect(self._on_contrast)
        row.addWidget(self._contrast_btn)
        return bar

    # ── 공개 API ─────────────────────────────────────────────────────

    def add_bubble(self, text: str, is_deaf: bool) -> None:
        bubble = _Bubble(text, is_deaf, self._font_size, self._high_contrast)
        self._bubbles.append(bubble)
        self._conv_layout.addWidget(bubble)
        self._scroll_bottom()

    def update_raw_words(self, words: str) -> None:
        self._raw_lbl.setText(f"누적: {words}" if words else "")

    def update_recognized_sentence(self, text: str) -> None:
        self._last_sentence = text
        if text:
            self._result_lbl.setText(text)
            self._result_lbl.setStyleSheet(
                f"color:{_RESULT}; padding:8px; background:transparent;"
                f" border:1px solid #1A4A2A; border-radius:8px;"
                f" font-weight:bold;"
            )
        else:
            self.clear_recognized()

    def clear_recognized(self) -> None:
        self._last_sentence = ""
        self._result_lbl.setText("수화를 시작하면 여기에 표시됩니다")
        self._result_lbl.setStyleSheet(
            f"color:#3A4A5A; padding:8px; background:transparent;"
            f" border:1px solid {_CARD}; border-radius:8px;"
        )
        self._raw_lbl.setText("")

    # 스트리밍 지원 ────────────────────────────────────────────────────

    def begin_streaming_bubble(self) -> None:
        """스트리밍 시작 — 빈 청각장애인 버블을 미리 추가."""
        self._streaming_bubble = _Bubble("", True, self._font_size, self._high_contrast)
        self._bubbles.append(self._streaming_bubble)
        self._conv_layout.addWidget(self._streaming_bubble)

    def append_token(self, token: str) -> None:
        """토큰 1개씩 스트리밍 버블에 추가 (타이핑 효과)."""
        if self._streaming_bubble is None:
            self.begin_streaming_bubble()
        current = self._streaming_bubble._lbl.text()
        self._streaming_bubble._lbl.setText(current + token)
        self.update_recognized_sentence(current + token)
        self._scroll_bottom()

    def finalize_bubble(self, full_text: str) -> None:
        """스트리밍 완료 — 버블 내용 확정."""
        if self._streaming_bubble:
            self._streaming_bubble._lbl.setText(full_text)
            self._streaming_bubble = None
        self.update_recognized_sentence(full_text)

    def set_stt_active(self, active: bool) -> None:
        """STT 상태에 따라 🎤 버튼 색상 변경."""
        style = _btn("#7A0A0A", "#9A1A1A") if active else _btn("#3A1A4A", "#5A2A6A")
        self._stt_btn.setStyleSheet(style)
        self._stt_btn.setToolTip("음성 입력 중 (클릭하면 중지)" if active else "음성 입력 시작")

    def set_input_text(self, text: str) -> None:
        """STT 인식 텍스트를 입력창에 표시."""
        self._input_edit.setPlainText(text)
        self._input_edit.moveCursor(QTextCursor.End)

    # ── 이벤트 핸들러 ────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if (obj is self._input_edit
                and event.type() == QEvent.KeyPress):
            ke: QKeyEvent = event
            if (ke.key() == Qt.Key_Return
                    and ke.modifiers() & Qt.ControlModifier):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def _on_tts(self) -> None:
        if self._last_sentence:
            self.tts_requested.emit(self._last_sentence)

    def _on_copy(self) -> None:
        if self._last_sentence:
            try:
                pyperclip.copy(self._last_sentence)
            except Exception:
                QApplication.clipboard().setText(self._last_sentence)

    def _on_confirm(self) -> None:
        if self._last_sentence:
            self.sentence_confirmed.emit(self._last_sentence)

    def _on_send(self) -> None:
        text = self._input_edit.toPlainText().strip()
        if text:
            self.message_sent.emit(text)
            self._input_edit.clear()

    def _on_font_size(self, size: int) -> None:
        self._font_size = size
        for bubble in self._bubbles:
            bubble.apply_style(size, self._high_contrast)

    def _on_contrast(self, checked: bool) -> None:
        self._high_contrast = checked
        for bubble in self._bubbles:
            bubble.apply_style(self._font_size, checked)
        bg = "#000000" if checked else "#14141E"
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background:{bg}; border-radius:8px; "
            f"border:1px solid {_CARD}; }}"
        )

    def _scroll_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, lambda: sb.setValue(sb.maximum()))
