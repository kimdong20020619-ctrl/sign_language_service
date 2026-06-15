"""ui/camera_widget.py — 카메라 뷰 + AR 오버레이 + 상태 인디케이터"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar,
    QSizePolicy, QVBoxLayout, QWidget,
)

_BG    = "#1E1E2E"
_CARD  = "#2A2A3E"
_GREEN = "#00FF88"
_BLUE  = "#4D9EFF"
_GRAY  = "#555577"
_TEXT  = "#FFFFFF"
_DIM   = "#8888AA"
_ACCENT = "#6C63FF"


class CameraWidget(QWidget):
    """
    카메라 프레임 표시 + 손/표정 상태 인디케이터 + 인식 결과 오버레이.

    update_frame(frame)              매 프레임 호출 (BGR numpy array)
    set_recognition_result(word, conf)
    set_hand_status(right, left)
    set_expression(expr_str)
    show_no_camera()
    set_high_contrast(bool)
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._high_contrast = False
        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # 카메라 프레임 라벨
        self._frame_lbl = QLabel(alignment=Qt.AlignCenter)
        self._frame_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame_lbl.setStyleSheet(
            "background:#0A0A18; border-radius:10px;"
            f"border:1px solid {_CARD}; color:{_GRAY}; font-size:14px;"
        )
        self._frame_lbl.setText("📷  카메라 초기화 중...")
        self._frame_lbl.setFont(QFont("맑은 고딕", 13))
        root.addWidget(self._frame_lbl, stretch=1)

        # 하단 상태 패널
        root.addWidget(self._build_status_panel())

    def _build_status_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedHeight(86)
        panel.setStyleSheet(
            f"background:{_CARD}; border-radius:8px; padding:0px;"
        )
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(12, 6, 12, 6)
        vbox.setSpacing(5)

        # 행 1: 손 상태 + 표정
        row1 = QHBoxLayout()
        self._rhand_lbl = _DotLabel("✋  오른손", _GREEN)
        self._lhand_lbl = _DotLabel("🤚  왼손",  _BLUE)
        self._expr_lbl  = QLabel("😐  표정: 중립")
        self._expr_lbl.setStyleSheet(f"color:{_DIM}; font-size:12px;")

        row1.addWidget(self._rhand_lbl)
        row1.addSpacing(14)
        row1.addWidget(self._lhand_lbl)
        row1.addStretch()
        row1.addWidget(self._expr_lbl)
        vbox.addLayout(row1)

        # 행 2: 인식 단어 + 신뢰도 바
        row2 = QHBoxLayout()
        self._word_lbl = QLabel("인식 중: —")
        self._word_lbl.setStyleSheet(
            f"color:{_GREEN}; font-size:14px; font-weight:bold;"
        )

        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setFixedWidth(110)
        self._conf_bar.setFixedHeight(10)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setStyleSheet(f"""
            QProgressBar           {{ background:#14142A; border-radius:5px; }}
            QProgressBar::chunk    {{ background:{_ACCENT}; border-radius:5px; }}
        """)
        self._conf_pct = QLabel("0%")
        self._conf_pct.setStyleSheet(f"color:{_DIM}; font-size:11px;")
        self._conf_pct.setFixedWidth(34)

        row2.addWidget(self._word_lbl)
        row2.addStretch()
        conf_row = QHBoxLayout()
        conf_row.setSpacing(4)
        conf_lbl = QLabel("신뢰도")
        conf_lbl.setStyleSheet(f"color:{_DIM}; font-size:11px;")
        conf_row.addWidget(conf_lbl)
        conf_row.addWidget(self._conf_bar)
        conf_row.addWidget(self._conf_pct)
        row2.addLayout(conf_row)
        vbox.addLayout(row2)

        return panel

    # ── 공개 API ─────────────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        # BGR → RGB
        rgb = frame[..., ::-1].copy() if frame.ndim == 3 else frame
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._frame_lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._frame_lbl.setPixmap(pix)

    def set_recognition_result(self, word: str, conf: float) -> None:
        self._word_lbl.setText(f'인식 중: "{word}"' if word else "인식 중: —")
        pct = int(conf * 100)
        self._conf_bar.setValue(pct)
        self._conf_pct.setText(f"{pct}%")

    def set_hand_status(self, right: bool, left: bool) -> None:
        self._rhand_lbl.set_active(right)
        self._lhand_lbl.set_active(left)

    def set_expression(self, expr: str) -> None:
        icons  = {
            "question":  "🤔", "questioning": "🤔",
            "negative":  "😟",
            "emphasis":  "😮",
            "command":   "👆",
            "happy":     "😊",
            "neutral":   "😐",
        }
        names  = {
            "question":  "의문",  "questioning": "의문",
            "negative":  "부정",
            "emphasis":  "강조",
            "command":   "명령",
            "happy":     "기쁨",
            "neutral":   "중립",
        }
        colors = {
            "question":  "#FFD700", "questioning": "#FFD700",
            "negative":  "#FF6B6B",
            "emphasis":  "#FF9900",
            "command":   "#CC88FF",
            "happy":     _GREEN,
            "neutral":   _DIM,
        }
        icon  = icons.get(expr, "😐")
        name  = names.get(expr, expr)
        color = colors.get(expr, _DIM)
        self._expr_lbl.setText(f"{icon}  표정: {name}")
        self._expr_lbl.setStyleSheet(f"color:{color}; font-size:12px;")

    def show_no_camera(self) -> None:
        self._frame_lbl.setPixmap(QPixmap())
        self._frame_lbl.setText("📷  카메라를 찾을 수 없습니다\n설정에서 카메라 인덱스를 확인해주세요")
        self._frame_lbl.setStyleSheet(
            "background:#1A0010; border-radius:10px;"
            "border:1px solid #FF4466; color:#FF6688; font-size:14px;"
        )

    def set_high_contrast(self, on: bool) -> None:
        self._high_contrast = on
        color = "#FFFF00" if on else _GREEN
        self._word_lbl.setStyleSheet(
            f"color:{color}; font-size:14px; font-weight:bold;"
        )


# ── 내부 헬퍼 위젯 ────────────────────────────────────────────────────

class _DotLabel(QLabel):
    """● / ○ 상태 표시 라벨."""

    def __init__(self, text: str, color: str, parent=None) -> None:
        super().__init__(parent)
        self._text  = text
        self._color = color
        self._active = False
        self._refresh()

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self._refresh()

    def _refresh(self) -> None:
        dot   = "●" if self._active else "○"
        color = self._color if self._active else _GRAY
        self.setText(f"{dot}  {self._text}")
        self.setStyleSheet(f"color:{color}; font-size:12px;")
