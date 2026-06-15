"""ui/quick_buttons.py — 환경별 자주 쓰는 문장 퀵버튼 패널"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

_BG    = "#1E1E2E"
_CARD  = "#2A2A3E"
_ACCENT = "#6C63FF"
_DIM   = "#8888AA"

# ── 환경별 퀵 문구 ────────────────────────────────────────────────────
QUICK_PHRASES: dict[str, list[str]] = {
    "카페": [
        "영수증", "카드", "현금", "포장", "먹고갈게요",
        "물", "테이크아웃", "봉투", "할인", "영수증필요없어요",
    ],
    "병원": [
        "아파요", "열나요", "언제부터", "약주세요",
        "입원", "검사", "예약", "응급", "처방전", "진료기록",
    ],
    "은행": [
        "통장개설", "이체", "출금", "잔액조회",
        "카드신청", "서류", "비밀번호", "확인", "대출상담",
    ],
    "학교": [
        "질문있어요", "이해못했어요", "다시설명", "화장실",
        "숙제", "시험", "상담", "결석", "지각", "조퇴",
    ],
    "마트": [
        "이거얼마예요", "할인되나요", "교환", "환불",
        "영수증", "포인트", "봉투", "카드결제", "위치",
    ],
    "직장": [
        "회의", "보고서", "메일", "전화", "부탁",
        "확인", "마감", "휴가", "점심", "퇴근",
    ],
}

_BTN_STYLE = """
    QPushButton {{
        background: {bg};
        color: {fg};
        border: 1px solid {border};
        border-radius: 7px;
        padding: 6px 4px;
        font-size: 12px;
        font-family: '맑은 고딕';
    }}
    QPushButton:hover  {{ background: {hover};  border-color: {accent}; }}
    QPushButton:pressed {{ background: #14142A; }}
"""


class QuickButtonsPanel(QWidget):
    """
    환경 모드에 맞는 퀵 문구 버튼을 그리드로 표시.

    phrase_selected(str) 시그널로 선택된 문구를 전달.
    set_mode(mode)로 환경 전환.
    """

    phrase_selected: Signal = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode          = "카페"
        self._high_contrast = False
        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # 헤더
        header = QLabel("🏷️  자주 쓰는 문장")
        header.setStyleSheet(
            f"color:{_DIM}; font-size:11px; font-weight:bold; padding:2px 0;"
        )
        root.addWidget(header)

        # 스크롤 영역
        self._inner = QWidget()
        self._grid  = QGridLayout(self._inner)
        self._grid.setSpacing(5)
        self._grid.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._inner)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(145)
        scroll.setStyleSheet(f"""
            QScrollArea          {{ background:{_BG}; border:none; }}
            QScrollBar:vertical  {{ background:{_BG}; width:5px; }}
            QScrollBar::handle:vertical {{ background:#3A3A5E; border-radius:2px; }}
        """)
        root.addWidget(scroll)

        self._populate()

    # ── 공개 API ─────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            self._populate()

    def set_high_contrast(self, on: bool) -> None:
        self._high_contrast = on
        self._populate()

    # ── 내부 ─────────────────────────────────────────────────────────

    def _populate(self) -> None:
        # 기존 버튼 제거
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        phrases = QUICK_PHRASES.get(self._mode, [])
        cols    = 3

        if self._high_contrast:
            bg, fg, border, hover, accent = (
                "#000000", "#FFFF00", "#FFFF00", "#222200", "#FFFF00"
            )
        else:
            bg, fg, border, hover, accent = (
                "#22223A", "#C0BDFF", "#3A3A5E", "#2E2E52", _ACCENT
            )

        style = _BTN_STYLE.format(
            bg=bg, fg=fg, border=border, hover=hover, accent=accent
        )

        for i, phrase in enumerate(phrases):
            btn = QPushButton(phrase)
            btn.setFixedHeight(38)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setStyleSheet(style)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda _checked, p=phrase: self.phrase_selected.emit(p)
            )
            self._grid.addWidget(btn, i // cols, i % cols)

        # 빈 열 stretch
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
