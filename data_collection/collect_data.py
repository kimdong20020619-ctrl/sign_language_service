#!/usr/bin/env python3
"""
수화 데이터 수집 도구
MediaPipe Hands + FaceMesh + Pose 동시 오버레이
수지 신호(손 모양/위치/방향/움직임) + 비수지 신호(표정/입/눈썹/고개/몸기울기) 수집
"""

import sys
import os
import json
import math
import time
import numpy as np
import cv2
from enum import Enum, auto
from datetime import datetime
from typing import Optional, List, Dict, Any

from PIL import Image, ImageDraw, ImageFont

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QProgressBar, QGroupBox, QListWidget,
    QListWidgetItem, QRadioButton, QButtonGroup, QFrame,
    QSizePolicy, QMessageBox, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor


# ══════════════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════════════

WORDS = [
    "안녕하세요", "감사합니다", "죄송합니다", "네", "아니오",
    "모르겠어요", "주세요", "얼마예요", "하나", "둘", "셋",
    "도와주세요", "잠깐만요", "괜찮아요", "화장실", "출구",
    "어디", "이름", "전화", "없어요",
]

TARGET_SAMPLES    = 50    # 단어당 목표 샘플 수
COLLECT_FRAMES    = 60    # 수집 프레임 수 (2초 × 30fps)
COUNTDOWN_SECS    = 3     # 카운트다운 초
STATIC_THRESHOLD  = 0.04  # 정적/동적 구분 손목 누적 이동 거리 임계값 (정규화 좌표)
EDGE_MARGIN       = 0.06  # 화면 가장자리 경계 (화면 밖 경고 기준)

# ── FaceMesh 입술 인덱스 (20개) ───────────────────────────────────────
LIP_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,    # 위 입술 외곽
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # 아래 입술 외곽
]

# ── FaceMesh 눈썹 인덱스 (각 5개) ────────────────────────────────────
EYEBROW_LEFT_IDX  = [70, 63, 105, 66, 107]
EYEBROW_RIGHT_IDX = [336, 296, 334, 293, 300]

# ── 헤드 포즈 solvePnP용 3D 참조 좌표 (mm) ───────────────────────────
FACE_3D_REF = np.array([
    [ 0.0,    0.0,    0.0],   # 코끝 (idx 4)
    [ 0.0,  -63.6,  -12.5],  # 턱 (idx 152)
    [-43.3,  32.7,  -26.0],  # 왼쪽 눈 외각 (idx 263)
    [ 43.3,  32.7,  -26.0],  # 오른쪽 눈 외각 (idx 33)
    [-28.9, -28.9,  -24.1],  # 왼쪽 입꼬리 (idx 287)
    [ 28.9, -28.9,  -24.1],  # 오른쪽 입꼬리 (idx 57)
], dtype=np.float64)

FACE_POSE_INDICES = [4, 152, 263, 33, 287, 57]

# ── Pose 상체 랜드마크 인덱스 ────────────────────────────────────────
UPPER_BODY_INDICES = frozenset(range(17))   # 0~16 (얼굴+어깨+팔꿈치+손목)

# ── 프로젝트 루트 / 데이터 경로 ─────────────────────────────────────
_KO_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
try:
    _FONT_OSD  = ImageFont.truetype(_KO_FONT_PATH, 22)
    _FONT_WARN = ImageFont.truetype(_KO_FONT_PATH, 16)
    _FONT_CD   = ImageFont.truetype(_KO_FONT_PATH, 48)
except Exception:
    _FONT_OSD = _FONT_WARN = _FONT_CD = ImageFont.load_default()


def _put_ko(frame, text, pos, font, color):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, fill=color, font=font)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")


# ══════════════════════════════════════════════════════════════════════
# 수집 상태 머신
# ══════════════════════════════════════════════════════════════════════

class CollectState(Enum):
    IDLE       = auto()
    COUNTDOWN  = auto()
    COLLECTING = auto()
    SAVING     = auto()


# ══════════════════════════════════════════════════════════════════════
# MediaPipe 통합 처리기 (Tasks API 기반 — MediaPipeEngine 래퍼)
# ══════════════════════════════════════════════════════════════════════

sys.path.insert(0, PROJECT_ROOT)
from core.mediapipe_engine import MediaPipeEngine


class MediaPipeProcessor:
    """MediaPipeEngine을 collect_data.py 인터페이스에 맞게 래핑."""

    def __init__(self):
        self._engine = MediaPipeEngine(
            max_hands=2,
            detection_confidence=0.65,
            tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> tuple:
        """Returns: (AR 오버레이 프레임, 추출 데이터 dict)"""
        annotated, frame_data = self._engine.process_frame(frame)
        if frame_data is None:
            frame_data = {
                "manual": {
                    "right_hand": None, "left_hand": None,
                    "right_hand_norm": None, "left_hand_norm": None,
                    "palm_direction": [0.0, 0.0, 0.0],
                    "hand_orientation": 0.0,
                    "_dominant_wrist": None,
                },
                "non_manual": {
                    "face_landmarks": None, "lip_shape": None,
                    "eyebrow_left": None, "eyebrow_right": None,
                    "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    "body_lean": {"shoulder_angle": 0.0, "torso_direction": 0.0},
                },
                "signing_space": {
                    "hand_relative_to_face":     [0.0, 0.0, 0.0],
                    "hand_relative_to_shoulder": [0.0, 0.0, 0.0],
                },
            }
        else:
            # non_manual에 body_lean 키 보정 (engine은 body_lean을 non_manual 안에 둠)
            nm = frame_data.get("non_manual", {})
            if "body_lean" not in nm:
                nm["body_lean"] = {"shoulder_angle": 0.0, "torso_direction": 0.0}
            # _dominant_wrist 내부 키 추가 (서명 공간 계산용)
            m = frame_data.get("manual", {})
            rh = m.get("right_hand")
            lh = m.get("left_hand")
            wrist = rh[0] if rh else (lh[0] if lh else None)
            m["_dominant_wrist"] = np.array(wrist) if wrist else None
        return annotated, frame_data

    def release(self):
        self._engine.release()


# ══════════════════════════════════════════════════════════════════════
# 카메라 스레드
# ══════════════════════════════════════════════════════════════════════

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    error       = Signal(str)

    def __init__(self, index: int = 0):
        super().__init__()
        self.index    = index
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error.emit(f"카메라 {self.index}를 열 수 없습니다.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        while self._running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)   # 기본 거울 모드
                self.frame_ready.emit(frame)
            self.msleep(33)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)


# ══════════════════════════════════════════════════════════════════════
# 메인 애플리케이션
# ══════════════════════════════════════════════════════════════════════

class CollectionApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("수화 데이터 수집 도구")
        self.setMinimumSize(1150, 740)

        self.processor     = MediaPipeProcessor()
        self.camera_thread = CameraThread()

        # 수집 상태
        self.state          = CollectState.IDLE
        self.current_word   = WORDS[0]
        self.collected      = []         # 현재 수집 중인 프레임 데이터
        self.wrist_traj     = []         # 손목 궤적 (정적/동적 판별용)
        self.prev_wrist     = None       # 직전 프레임 손목 위치

        # 품질 플래그
        self.hand_detected  = False
        self.face_detected  = False
        self.hand_in_frame  = True

        # 카운트다운 타이머
        self.countdown_val  = COUNTDOWN_SECS
        self.countdown_timer = QTimer()
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._tick_countdown)

        # 단어별 기존 샘플 수
        self.sample_counts  = {w: self._count_existing(w) for w in WORDS}

        self._setup_ui()
        self._apply_theme()

        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.error.connect(
            lambda msg: self.statusBar().showMessage(f"카메라 오류: {msg}")
        )
        self.camera_thread.start()

    # ──────────────────────────────────────────────────────────────────
    # 파일 경로 유틸
    # ──────────────────────────────────────────────────────────────────
    def _sample_dir(self, word: str) -> str:
        return os.path.join(DATA_ROOT, word)

    def _count_existing(self, word: str) -> int:
        d = self._sample_dir(word)
        if not os.path.isdir(d):
            return 0
        return sum(1 for f in os.listdir(d) if f.endswith(".json"))

    def _next_sample_num(self, word: str) -> int:
        return self.sample_counts.get(word, 0) + 1

    # ──────────────────────────────────────────────────────────────────
    # UI 구성
    # ──────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_camera_panel(), stretch=3)
        root.addWidget(self._build_control_panel(), stretch=2)

        sb = self.statusBar()
        sb.showMessage("준비 완료  |  수집할 단어를 선택하고 '수집 시작'을 누르세요")

    # ── 왼쪽: 카메라 패널 ─────────────────────────────────────────────
    def _build_camera_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("카메라 초기화 중...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(560, 420)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: #0d1117; color: #555; font-size: 14px;
                border-radius: 10px; border: 2px solid #1e2a3a;
            }
        """)
        layout.addWidget(self.camera_label, stretch=1)

        # 품질 인디케이터 바
        qi = QHBoxLayout()
        self.qi_hand  = self._qi_badge("손 감지", False)
        self.qi_face  = self._qi_badge("얼굴 감지", False)
        self.qi_edge  = self._qi_badge("화면 내", True)
        self.qi_type  = QLabel("수화 유형: ─")
        self.qi_type.setStyleSheet("color: #8090a0; font-size: 11px;")

        qi.addWidget(self.qi_hand)
        qi.addWidget(self.qi_face)
        qi.addWidget(self.qi_edge)
        qi.addStretch()
        qi.addWidget(self.qi_type)
        layout.addLayout(qi)
        return w

    def _qi_badge(self, text: str, ok: bool) -> QLabel:
        lbl = QLabel(f"● {text}")
        lbl.setStyleSheet(
            f"color: {'#30c060' if ok else '#808090'}; font-size: 11px;"
        )
        return lbl

    def _update_qi(self, hand: bool, face: bool, in_frame: bool,
                   sign_type: Optional[str]):
        def style(ok): return f"color: {'#30c060' if ok else '#c04040'}; font-size: 11px;"
        self.qi_hand.setStyleSheet(style(hand))
        self.qi_hand.setText(f"● 손 감지")
        self.qi_face.setStyleSheet(style(face))
        self.qi_face.setText(f"● 얼굴 감지")
        self.qi_edge.setStyleSheet(style(in_frame))
        self.qi_edge.setText(f"● {'화면 내' if in_frame else '화면 밖 !'}")
        if sign_type:
            self.qi_type.setText(f"수화 유형: {sign_type}")

    # ── 오른쪽: 컨트롤 패널 ───────────────────────────────────────────
    def _build_control_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(380)
        layout = QVBoxLayout(w)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_settings_group())
        layout.addWidget(self._build_word_group(), stretch=1)
        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_progress_group())
        layout.addWidget(self._build_buttons_group())
        return w

    def _build_settings_group(self) -> QGroupBox:
        g = QGroupBox("주동손 설정")
        layout = QHBoxLayout(g)

        self.right_rb = QRadioButton("오른손잡이")
        self.left_rb  = QRadioButton("왼손잡이")
        self.right_rb.setChecked(True)

        self._hand_group = QButtonGroup()
        self._hand_group.addButton(self.right_rb)
        self._hand_group.addButton(self.left_rb)

        note = QLabel("(왼손잡이: 자동 좌우 반전)")
        note.setStyleSheet("color: #6080a0; font-size: 10px;")

        layout.addWidget(self.right_rb)
        layout.addWidget(self.left_rb)
        layout.addStretch()
        layout.addWidget(note)
        return g

    def _build_word_group(self) -> QGroupBox:
        g = QGroupBox("수집할 단어 목록 (클릭으로 선택)")
        layout = QVBoxLayout(g)

        self.word_list = QListWidget()
        self.word_list.setStyleSheet("""
            QListWidget {
                background: #0d1117; border: 1px solid #1e2a3a;
                border-radius: 6px; font-size: 12px;
            }
            QListWidget::item { padding: 5px 8px; border-bottom: 1px solid #151f2e; }
            QListWidget::item:selected { background: #1a3a6a; color: #80d0ff; }
            QListWidget::item:hover { background: #0f2040; }
        """)
        self.word_list.currentRowChanged.connect(self._on_word_selected)
        self._populate_word_list()
        layout.addWidget(self.word_list)

        # 전체 진행률
        self.total_progress = QProgressBar()
        self.total_progress.setRange(0, len(WORDS) * TARGET_SAMPLES)
        self.total_progress.setFormat("전체: %v / %m 샘플")
        self.total_progress.setStyleSheet(self._progress_style("#2a7a4a"))
        layout.addWidget(self.total_progress)
        self._refresh_total_progress()
        return g

    def _build_status_group(self) -> QGroupBox:
        g = QGroupBox("수집 상태")
        layout = QVBoxLayout(g)
        layout.setSpacing(6)

        # 현재 단어 표시
        self.cur_word_label = QLabel(self.current_word)
        self.cur_word_label.setFont(QFont("맑은 고딕", 18, QFont.Bold))
        self.cur_word_label.setAlignment(Qt.AlignCenter)
        self.cur_word_label.setStyleSheet("color: #00c8ff;")

        # 상태 메시지
        self.status_label = QLabel("단어를 선택하고 수집을 시작하세요")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #8090a0; font-size: 12px;")

        # 현재 단어 진행
        self.word_progress = QProgressBar()
        self.word_progress.setRange(0, TARGET_SAMPLES)
        self.word_progress.setFormat(f"%v / {TARGET_SAMPLES} 샘플")
        self.word_progress.setStyleSheet(self._progress_style("#1a5a9a"))

        # 수집 프레임 프로그레스
        self.frame_progress = QProgressBar()
        self.frame_progress.setRange(0, COLLECT_FRAMES)
        self.frame_progress.setFormat("프레임: %v / %m")
        self.frame_progress.setValue(0)
        self.frame_progress.setStyleSheet(self._progress_style("#7a3a1a"))

        layout.addWidget(self.cur_word_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.word_progress)
        layout.addWidget(self.frame_progress)
        self._refresh_word_progress()
        return g

    def _build_progress_group(self) -> QGroupBox:
        g = QGroupBox("단어별 완료 현황")
        layout = QVBoxLayout(g)

        self.completion_label = QLabel()
        self.completion_label.setWordWrap(True)
        self.completion_label.setStyleSheet("color: #8090a0; font-size: 10px;")
        layout.addWidget(self.completion_label)
        self._refresh_completion_label()
        return g

    def _build_buttons_group(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        self.start_btn = QPushButton("▶  수집 시작  (3초 후 시작)")
        self.start_btn.setMinimumHeight(44)
        self.start_btn.setStyleSheet(self._btn("#1a6a3a"))
        self.start_btn.clicked.connect(self._start_collection)

        self.stop_btn = QPushButton("■  수집 중지")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self._btn("#7a2020"))
        self.stop_btn.clicked.connect(self._stop_collection)

        self.delete_btn = QPushButton("✕  마지막 샘플 삭제")
        self.delete_btn.setMinimumHeight(32)
        self.delete_btn.setStyleSheet(self._btn("#4a3020"))
        self.delete_btn.clicked.connect(self._delete_last_sample)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.delete_btn)
        return w

    # ──────────────────────────────────────────────────────────────────
    # 프레임 처리 (카메라 스레드 → 메인 스레드)
    # ──────────────────────────────────────────────────────────────────
    @Slot(np.ndarray)
    def _on_frame(self, frame: np.ndarray):
        # 왼손잡이: 추가 좌우 반전 (카메라 스레드가 이미 한 번 flip)
        # → 원본 복원 후 다시 flip 하면 두 번 반전이므로,
        #   카메라 스레드의 flip을 취소하고 반전 여부를 여기서 결정
        if self.left_rb.isChecked():
            frame = cv2.flip(frame, 1)   # 왼손잡이: 추가 반전 없음 (이미 거울)
        # 오른손잡이는 카메라 스레드의 기본 거울 모드 유지

        annotated, data = self.processor.process(frame)
        h_f, w_f = annotated.shape[:2]

        manual = data["manual"]

        # ── 품질 체크 ─────────────────────────────────────────────────
        self.hand_detected = (manual["right_hand"] is not None or
                              manual["left_hand"] is not None)
        self.face_detected = data["non_manual"]["face_landmarks"] is not None
        self.hand_in_frame = self._check_hand_in_frame(manual)

        # ── 수집 처리 ─────────────────────────────────────────────────
        if self.state == CollectState.COLLECTING:
            self._accumulate_frame(data)

        # ── 오버레이 ──────────────────────────────────────────────────
        self._draw_overlays(annotated, w_f, h_f)

        # ── Qt 이미지 업데이트 ────────────────────────────────────────
        self._display_frame(annotated)

        # ── 품질 인디케이터 UI 업데이트 ──────────────────────────────
        sign_type = None
        if self.state == CollectState.IDLE and len(self.wrist_traj) > 1:
            sign_type = "정적" if self._is_static() else "동적"
        self._update_qi(self.hand_detected, self.face_detected,
                        self.hand_in_frame, sign_type)

    def _check_hand_in_frame(self, manual: dict) -> bool:
        """손이 화면 가장자리에서 벗어나지 않는지 확인."""
        hand = manual.get("right_hand") or manual.get("left_hand")
        if hand is None:
            return False
        for pt in hand:
            if pt[0] < EDGE_MARGIN or pt[0] > 1-EDGE_MARGIN \
               or pt[1] < EDGE_MARGIN or pt[1] > 1-EDGE_MARGIN:
                return False
        return True

    def _draw_overlays(self, frame, w, h):
        """카운트다운 / 수집 진행 / 경고 오버레이."""
        if self.state == CollectState.COUNTDOWN:
            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            # 카운트다운 숫자
            txt = str(self.countdown_val)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 6, 14)
            cv2.putText(frame, txt,
                        ((w - tw) // 2, (h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 6.0,
                        (0, 230, 230), 14, cv2.LINE_AA)
            # 단어 표시
            _put_ko(frame, self.current_word,
                    (w // 2 - 60, h // 2 + 60),
                    _FONT_CD, (255, 255, 255))

        elif self.state == CollectState.COLLECTING:
            # 수집 진행 바
            n = len(self.collected)
            bw = int(w * n / COLLECT_FRAMES)
            cv2.rectangle(frame, (0, h-10), (bw, h), (0, 200, 80), -1)
            cv2.rectangle(frame, (0, h-10), (w, h), (40, 40, 40), 1)
            # REC 표시
            cv2.circle(frame, (20, 20), 8, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (34, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 60, 60), 2)
            # 프레임 수
            cv2.putText(frame, f"{n}/{COLLECT_FRAMES}",
                        (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2)

        # 품질 경고
        if not self.hand_in_frame and self.hand_detected:
            cv2.rectangle(frame, (0, 0), (w, 4), (0, 0, 240), -1)
            _put_ko(frame, "경고: 손이 화면 가장자리에 있습니다",
                    (10, h - 30), _FONT_WARN, (100, 100, 255))

        if not self.hand_detected and self.state == CollectState.COLLECTING:
            _put_ko(frame, "손이 감지되지 않습니다",
                    (10, 42), _FONT_OSD, (100, 130, 255))

    def _display_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qt_img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pix)

    # ──────────────────────────────────────────────────────────────────
    # 수집 로직
    # ──────────────────────────────────────────────────────────────────
    def _start_collection(self):
        if self.state != CollectState.IDLE:
            return
        n = self.sample_counts.get(self.current_word, 0)
        if n >= TARGET_SAMPLES:
            QMessageBox.information(
                self, "수집 완료",
                f"'{self.current_word}'의 수집이 이미 완료되었습니다.\n"
                f"다른 단어를 선택해주세요."
            )
            return

        self.state = CollectState.COUNTDOWN
        self.countdown_val = COUNTDOWN_SECS
        self.collected.clear()
        self.wrist_traj.clear()
        self.prev_wrist = None

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"준비하세요...  {self.countdown_val}")
        self.countdown_timer.start()

    def _stop_collection(self):
        """수집 강제 중지."""
        self.countdown_timer.stop()
        self.state = CollectState.IDLE
        self.collected.clear()
        self.wrist_traj.clear()
        self.prev_wrist = None
        self.frame_progress.setValue(0)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("수집이 취소되었습니다.")
        self.statusBar().showMessage("수집 취소됨")

    def _tick_countdown(self):
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.status_label.setText(f"준비하세요...  {self.countdown_val}")
        else:
            self.countdown_timer.stop()
            self.state = CollectState.COLLECTING
            self.status_label.setText("수집 중...")
            self.statusBar().showMessage(
                f"'{self.current_word}' 수화를 보여주세요  ({COLLECT_FRAMES}프레임 수집 중)"
            )

    def _accumulate_frame(self, data: dict):
        """프레임 데이터를 버퍼에 추가하고 완료 시 저장."""
        manual = data["manual"]
        dom_wrist = manual.get("_dominant_wrist")

        # 손목 궤적 및 속도
        if dom_wrist is not None:
            wrist_pos = dom_wrist.tolist()
        else:
            wrist_pos = [0.0, 0.0, 0.0]

        velocity = [0.0, 0.0, 0.0]
        if self.prev_wrist is not None and dom_wrist is not None:
            velocity = (dom_wrist - np.array(self.prev_wrist)).tolist()
        self.prev_wrist = wrist_pos
        self.wrist_traj.append(wrist_pos)

        # 내부 키 제거
        clean_manual = {k: v for k, v in manual.items() if not k.startswith("_")}

        # 누적 궤적 (이 프레임까지의 모든 손목 위치)
        cumulative = list(self.wrist_traj)

        frame_record = {
            "frame_idx": len(self.collected),
            "manual":       clean_manual,
            "non_manual":   data["non_manual"],
            "signing_space": data["signing_space"],
            "movement": {
                "velocity":   velocity,
                "trajectory": cumulative,
                "is_static":  False,   # 수집 완료 후 결정
            },
        }
        self.collected.append(frame_record)
        self.frame_progress.setValue(len(self.collected))

        if len(self.collected) >= COLLECT_FRAMES:
            self._finish_collection()

    def _finish_collection(self):
        self.state = CollectState.SAVING
        self.countdown_timer.stop()
        self.stop_btn.setEnabled(False)

        is_static = self._is_static()

        # 모든 프레임에 is_static 적용
        for fr in self.collected:
            fr["movement"]["is_static"] = is_static

        # 샘플 JSON 구성
        word        = self.current_word
        sample_num  = self._next_sample_num(word)
        handedness  = "right" if self.right_rb.isChecked() else "left"
        sign_type   = "static" if is_static else "dynamic"

        sample = {
            "word":         word,
            "handedness":   handedness,
            "sign_type":    sign_type,
            "sample_num":   sample_num,
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "total_frames": len(self.collected),
            "frames":       self.collected,
        }

        # 저장
        save_dir = self._sample_dir(word)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"sample_{sample_num:03d}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, separators=(",", ":"),
                      default=_json_default)

        # UI 갱신
        self.sample_counts[word] = sample_num
        self._refresh_word_progress()
        self._refresh_total_progress()
        self._refresh_completion_label()
        self._populate_word_list()

        result_msg = (
            f"✓  '{word}' 샘플 {sample_num}/{TARGET_SAMPLES} 저장  "
            f"[{sign_type}]"
        )
        self.status_label.setText(result_msg)
        self.statusBar().showMessage(f"저장 완료: {filepath}")

        # 리셋
        self.collected.clear()
        self.wrist_traj.clear()
        self.prev_wrist = None
        self.frame_progress.setValue(0)
        self.state = CollectState.IDLE
        self.start_btn.setEnabled(True)

        # 목표 도달 시 알림
        if sample_num >= TARGET_SAMPLES:
            self.status_label.setText(f"🎉 '{word}' 수집 완료! 다음 단어를 선택하세요.")

    def _is_static(self) -> bool:
        """손목 총 이동 거리로 정적/동적 판별."""
        if len(self.wrist_traj) < 2:
            return True
        arr  = np.array(self.wrist_traj)
        dist = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
        return dist < STATIC_THRESHOLD

    # ──────────────────────────────────────────────────────────────────
    # 단어 선택
    # ──────────────────────────────────────────────────────────────────
    def _on_word_selected(self, row: int):
        if 0 <= row < len(WORDS):
            self.current_word = WORDS[row]
            if not hasattr(self, 'cur_word_label'):
                return
            self.cur_word_label.setText(self.current_word)
            self._refresh_word_progress()
            n = self.sample_counts.get(self.current_word, 0)
            self.status_label.setText(
                f"'{self.current_word}'  현재 {n}/{TARGET_SAMPLES} 수집됨"
            )

    # ──────────────────────────────────────────────────────────────────
    # 샘플 삭제
    # ──────────────────────────────────────────────────────────────────
    def _delete_last_sample(self):
        word    = self.current_word
        n       = self.sample_counts.get(word, 0)
        if n == 0:
            QMessageBox.information(self, "삭제", f"'{word}'의 삭제할 샘플이 없습니다.")
            return
        filepath = os.path.join(self._sample_dir(word), f"sample_{n:03d}.json")
        if os.path.exists(filepath):
            reply = QMessageBox.question(
                self, "샘플 삭제 확인",
                f"'{word}' 샘플 {n}번을 삭제하시겠습니까?\n\n{filepath}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                os.remove(filepath)
                self.sample_counts[word] = n - 1
                self._refresh_word_progress()
                self._refresh_total_progress()
                self._refresh_completion_label()
                self._populate_word_list()
                self.status_label.setText(f"삭제 완료: {word} 샘플 {n}번")
                self.statusBar().showMessage(f"삭제: {filepath}")
        else:
            QMessageBox.warning(self, "오류", f"파일을 찾을 수 없습니다:\n{filepath}")

    # ──────────────────────────────────────────────────────────────────
    # UI 갱신 헬퍼
    # ──────────────────────────────────────────────────────────────────
    def _populate_word_list(self):
        self.word_list.blockSignals(True)
        cur_row = self.word_list.currentRow()
        self.word_list.clear()

        for word in WORDS:
            n    = self.sample_counts.get(word, 0)
            done = n >= TARGET_SAMPLES
            item = QListWidgetItem(
                f"{'✓ ' if done else '  '}{word}   "
                f"({n}/{TARGET_SAMPLES})"
            )
            if done:
                item.setForeground(QColor("#30c060"))
            elif n > 0:
                item.setForeground(QColor("#d0a020"))
            else:
                item.setForeground(QColor("#9090b0"))
            self.word_list.addItem(item)

        self.word_list.blockSignals(False)
        self.word_list.setCurrentRow(cur_row if cur_row >= 0 else 0)

    def _refresh_word_progress(self):
        n = self.sample_counts.get(self.current_word, 0)
        self.word_progress.setValue(n)
        self.word_progress.setFormat(
            f"'{self.current_word}': %v / {TARGET_SAMPLES} 샘플"
        )

    def _refresh_total_progress(self):
        total = sum(min(c, TARGET_SAMPLES) for c in self.sample_counts.values())
        self.total_progress.setValue(total)

    def _refresh_completion_label(self):
        done  = sum(1 for n in self.sample_counts.values() if n >= TARGET_SAMPLES)
        total = sum(min(n, TARGET_SAMPLES) for n in self.sample_counts.values())
        self.completion_label.setText(
            f"완료된 단어: {done} / {len(WORDS)}  |  "
            f"총 수집 샘플: {total} / {len(WORDS) * TARGET_SAMPLES}"
        )

    # ──────────────────────────────────────────────────────────────────
    # 스타일
    # ──────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117; color: #c8d4e0;
                font-family: '맑은 고딕', sans-serif;
            }
            QGroupBox {
                border: 1px solid #1e2a3a; border-radius: 8px;
                margin-top: 10px; padding: 10px 6px 6px 6px;
                color: #6080a0; font-weight: bold; font-size: 11px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QRadioButton { color: #a0b0c0; }
            QStatusBar { background: #161b22; color: #5a6a7a; font-size: 11px; }
        """)
        self._populate_word_list()

    @staticmethod
    def _progress_style(color: str) -> str:
        return (
            f"QProgressBar {{ background: #161b22; border: 1px solid #1e2a3a; "
            f"border-radius: 5px; text-align: center; color: #c0d0e0; "
            f"font-size: 11px; height: 18px; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
        )

    @staticmethod
    def _btn(color: str) -> str:
        return (
            f"QPushButton {{ background: {color}; color: #d0e0f0; "
            f"border: none; border-radius: 7px; font-size: 12px; "
            f"font-weight: bold; }}"
            f"QPushButton:hover {{ background: {color}cc; }}"
            f"QPushButton:disabled {{ background: #1a2030; color: #404050; }}"
        )

    # ──────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.countdown_timer.stop()
        self.camera_thread.stop()
        self.processor.release()
        event.accept()


# ══════════════════════════════════════════════════════════════════════
# JSON 직렬화 도우미
# ══════════════════════════════════════════════════════════════════════

def _json_default(obj):
    """numpy 타입을 JSON 직렬화 가능 타입으로 변환."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    raise TypeError(f"직렬화 불가 타입: {type(obj)}")


# ══════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════

def main():
    os.chdir(PROJECT_ROOT)  # 작업 디렉터리를 프로젝트 루트로 고정
    app = QApplication(sys.argv)
    app.setApplicationName("수화 데이터 수집 도구")
    window = CollectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
