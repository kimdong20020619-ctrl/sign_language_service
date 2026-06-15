#!/usr/bin/env python3
"""
수화 모델 학습 파이프라인
- 정적 수화: RandomForest (단일 프레임 특징벡터 평균)
- 동적 수화: LSTM 시퀀스 분류기
- 좌우 반전 데이터 증강 (×2)
- 단어별 혼동 행렬 출력
"""

import sys
import os
import json
import math
import copy
import re
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ── TensorFlow 선택적 임포트 ─────────────────────────────────────────
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                          BatchNormalization, Masking)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── matplotlib 선택적 임포트 ─────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── ANSI 컬러 (Windows Terminal / CI 공통 지원) ──────────────────────
_GREEN = "\033[32m"
_RED   = "\033[31m"
_CYAN  = "\033[36m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

# ══════════════════════════════════════════════════════════════════════
# 경로
# ══════════════════════════════════════════════════════════════════════

_THIS_DIR    = Path(__file__).parent
PROJECT_ROOT = _THIS_DIR.parent
DATA_ROOT    = PROJECT_ROOT / "data" / "raw"
DATA_AIHUB   = PROJECT_ROOT / "data" / "aihub"          # 단어 JSON
DATA_SENT    = PROJECT_ROOT / "data" / "aihub_sentences" # 문장 JSON
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════════════

SEQ_LEN = 60        # LSTM 시퀀스 길이 (프레임 수)
TEST_RATIO  = 0.20  # 테스트 비율
RANDOM_SEED = 42

# 입술 20개 포인트 중 5개 키포인트 인덱스 (10D 추출용)
LIP_KEY_IDX = [0, 3, 5, 9, 14]

# FaceMesh 눈 외각 랜드마크 (눈썹 올림 기준점)
EYE_L_CORNER = 362
EYE_R_CORNER = 133

# MediaPipe Hands 랜드마크 인덱스
MID_MCP_IDX = 9    # 중지 MCP
MID_TIP_IDX = 12   # 중지 끝


# ══════════════════════════════════════════════════════════════════════
# 특징 추출
# ══════════════════════════════════════════════════════════════════════

def _scale_hand(hand_norm: Optional[List]) -> np.ndarray:
    """
    손목 기준 정규화 좌표 → 중지 길이(MCP→TIP)로 스케일 정규화.
    없으면 63차원 영벡터 반환.
    """
    if hand_norm is None:
        return np.zeros(63, dtype=np.float32)

    arr = np.array(hand_norm, dtype=np.float32)          # (21, 3)
    if arr.shape != (21, 3):
        return np.zeros(63, dtype=np.float32)

    mid_len = float(np.linalg.norm(arr[MID_TIP_IDX] - arr[MID_MCP_IDX]))
    if mid_len > 1e-6:
        arr = arr / mid_len
    return arr.flatten()                                  # 63D


def extract_manual(frame: dict) -> np.ndarray:
    """
    수지 특징 추출.
    오른손 정규화(63D) + 왼손 정규화(63D, 없으면 0) + 손바닥방향(3D) + 회전각(1D)
    = 130D
    """
    m  = frame.get("manual", {})
    rh = _scale_hand(m.get("right_hand_norm"))            # 63D
    lh = _scale_hand(m.get("left_hand_norm"))             # 63D (패딩 포함)
    pd = np.array(m.get("palm_direction", [0.0, 0.0, 0.0]), dtype=np.float32)  # 3D
    ori = np.array([m.get("hand_orientation", 0.0)], dtype=np.float32)          # 1D
    return np.concatenate([rh, lh, pd, ori])              # 130D


def extract_non_manual(frame: dict) -> np.ndarray:
    """
    비수지 특징 추출.
    눈썹올림(2D) + 입개방도(1D) + 입모양키포인트(10D)
    + 고개자세(3D) + 어깨기울기(1D) + 몸통방향(1D) = 18D
    """
    nm   = frame.get("non_manual", {})
    fl   = nm.get("face_landmarks")
    eb_l = nm.get("eyebrow_left")
    eb_r = nm.get("eyebrow_right")
    lips = nm.get("lip_shape")
    hp   = nm.get("head_pose", {})
    bl   = nm.get("body_lean", {})

    feats: List[float] = []

    # ── 눈썹 올림 (2D) ────────────────────────────────────────────
    if fl and eb_l and eb_r:
        # 얼굴 포맷에 따라 눈 기준점 인덱스 선택
        if len(fl) >= 362:      # MediaPipe FaceMesh
            eye_l_y = fl[EYE_L_CORNER][1]
            eye_r_y = fl[EYE_R_CORNER][1]
        elif len(fl) >= 46:     # OpenPose 70점: 45=왼눈외곽, 36=오른눈외곽
            eye_l_y = fl[45][1]
            eye_r_y = fl[36][1]
        else:
            eye_l_y = eye_r_y = None
        if eye_l_y is not None:
            left_raise  = float(eye_l_y - np.mean([p[1] for p in eb_l]))
            right_raise = float(eye_r_y - np.mean([p[1] for p in eb_r]))
        else:
            left_raise = right_raise = 0.0
    else:
        left_raise = right_raise = 0.0
    feats += [left_raise, right_raise]

    # ── 입 개방도 (1D) ────────────────────────────────────────────
    if lips and len(lips) >= 20:
        # index 5 = landmark 0 (상순 중심), index 14 = landmark 17 (하순 중심)
        mouth_open = float(lips[14][1] - lips[5][1])
    else:
        mouth_open = 0.0
    feats.append(mouth_open)

    # ── 입 모양 키포인트 (10D: 5포인트 × x,y) ────────────────────
    if lips and len(lips) >= 20:
        lps_arr  = np.array(lips, dtype=np.float32)
        key_pts  = lps_arr[LIP_KEY_IDX, :2].flatten()    # 10D
    else:
        key_pts = np.zeros(10, dtype=np.float32)
    feats += key_pts.tolist()

    # ── 고개 자세 pitch/yaw/roll (3D) ────────────────────────────
    feats += [hp.get("pitch", 0.0),
              hp.get("yaw",   0.0),
              hp.get("roll",  0.0)]

    # ── 어깨 기울기 (1D) + 몸통 방향 (1D) ───────────────────────
    feats.append(bl.get("shoulder_angle",  0.0))
    feats.append(bl.get("torso_direction", 0.0))

    return np.array(feats, dtype=np.float32)              # 18D


def extract_spatial(frame: dict) -> np.ndarray:
    """수화 공간 특징 (6D): 얼굴 기준 손 위치(3D) + 어깨 기준 손 위치(3D)."""
    ss  = frame.get("signing_space", {})
    rf  = ss.get("hand_relative_to_face",     [0.0, 0.0, 0.0])
    rs  = ss.get("hand_relative_to_shoulder", [0.0, 0.0, 0.0])
    return np.array(rf + rs, dtype=np.float32)            # 6D


def extract_frame_vector(frame: dict,
                          velocity:   Optional[List] = None,
                          hand_change: float = 0.0) -> np.ndarray:
    """
    단일 프레임 완전 특징벡터.
    정적: 154D  /  동적(velocity+변화량 포함): 158D
    """
    base = np.concatenate([
        extract_manual(frame),       # 130D
        extract_non_manual(frame),   # 18D
        extract_spatial(frame),      #  6D
    ])                               # = 154D

    if velocity is not None:
        vel = np.array(velocity, dtype=np.float32)        # 3D
        chg = np.array([hand_change], dtype=np.float32)   # 1D
        base = np.concatenate([base, vel, chg])            # 158D

    return base.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# AIHub JSON 파서 (키포인트 + 비수지 주석)
# ══════════════════════════════════════════════════════════════════════

# 비수지 라벨 클래스
NM_LABELS = ["neutral", "question", "negative", "emphasis", "command"]

# AIHub 비수지 sentence_type → 내부 라벨 매핑
_NM_TYPE_MAP: Dict[str, str] = {
    "의문문": "question",  "question": "question",
    "부정문": "negative",  "negative": "negative",
    "명령문": "command",   "command":  "command",
    "강조":   "emphasis",  "emphasis": "emphasis",
    "평서문": "neutral",   "statement": "neutral", "neutral": "neutral",
}


def _parse_aihub_frame(raw: dict, face_lms: Optional[list] = None) -> dict:
    """
    AIHub 키포인트 JSON의 단일 프레임 → 내부 frame_data 포맷 변환.

    AIHub 예상 구조 (두 가지 변형 모두 지원):
      변형 A: {"right_hand": [[x,y,z]×21], "left_hand": [...], "pose": [...], "face": [...]}
      변형 B: {"keypoints": {"right_hand": ..., "left_hand": ..., ...}}
    """
    kp = raw.get("keypoints", raw)  # 변형 B → A 통일

    rh_raw  = kp.get("right_hand")  or kp.get("rightHand")
    lh_raw  = kp.get("left_hand")   or kp.get("leftHand")
    pose_raw= kp.get("pose")
    face_raw= kp.get("face")        or face_lms

    # 손목 기준 정규화
    def _norm(pts):
        if not pts or len(pts) < 21:
            return None
        w = pts[0]
        return [[p[0]-w[0], p[1]-w[1], p[2]-w[2]] for p in pts]

    rh_norm = _norm(rh_raw)
    lh_norm = _norm(lh_raw)

    # 손바닥 방향 (오른손 기준)
    palm_dir = [0.0, 0.0, 0.0]
    hand_ori = 0.0
    if rh_raw and len(rh_raw) >= 21:
        import numpy as _np
        p0 = _np.array(rh_raw[0])
        v1 = _np.array(rh_raw[5])  - p0
        v2 = _np.array(rh_raw[17]) - p0
        n  = _np.cross(v1, v2)
        mag = float(_np.linalg.norm(n))
        palm_dir = (n / mag).tolist() if mag > 1e-6 else palm_dir
        mcp = _np.array(rh_raw[9]); tip = _np.array(rh_raw[12])
        d   = tip - mcp
        hand_ori = float(_np.arctan2(float(d[1]), float(d[0])))

    # 얼굴 랜드마크 → 눈썹/입/고개
    eb_left = eb_right = lip = None
    head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    nose_tip  = [0.0, 0.0, 0.0]

    if face_raw and len(face_raw) >= 48:
        if len(face_raw) >= 468:
            # MediaPipe FaceMesh 468점
            EYEBROW_L = list(range(46, 56))
            EYEBROW_R = list(range(336, 346))
            LIP_IDX   = list(range(61, 81))
            nose_tip  = face_raw[1] if len(face_raw) > 1 else [0.0, 0.0, 0.0]
        else:
            # OpenPose 70점: 17-21=오른눈썹, 22-26=왼눈썹, 48-67=입, 30=코끝
            EYEBROW_L = [22, 23, 24, 25, 26]
            EYEBROW_R = [17, 18, 19, 20, 21]
            LIP_IDX   = list(range(48, 68))
            nose_tip  = face_raw[30] if len(face_raw) > 30 else [0.0, 0.0, 0.0]
        eb_left  = [face_raw[i] for i in EYEBROW_L if i < len(face_raw)]
        eb_right = [face_raw[i] for i in EYEBROW_R if i < len(face_raw)]
        lip      = [face_raw[i] for i in LIP_IDX if i < len(face_raw)]

    # 포즈 → 어깨/몸통
    sh_angle = torso_dir = 0.0
    right_sh  = left_sh  = None
    if pose_raw and len(pose_raw) >= 25:
        import numpy as _np2
        ls = _np2.array(pose_raw[11][:2])
        rs = _np2.array(pose_raw[12][:2])
        lh_p = _np2.array(pose_raw[23][:2])
        rh_p = _np2.array(pose_raw[24][:2])
        diff    = rs - ls
        t_vec   = (lh_p + rh_p) / 2.0 - (ls + rs) / 2.0
        sh_angle  = float(_np2.arctan2(diff[1], diff[0]))
        torso_dir = float(_np2.arctan2(t_vec[0], t_vec[1]))
        left_sh  = pose_raw[11]
        right_sh = pose_raw[12]

    # 수화 공간
    wrist   = rh_raw[0]  if rh_raw  else [0.0, 0.0, 0.0]
    rsh     = right_sh   or [0.0, 0.0, 0.0]
    rel_face = [wrist[i] - nose_tip[i] for i in range(3)]
    rel_sh   = [wrist[i] - rsh[i]      for i in range(3)]

    return {
        "manual": {
            "right_hand":       rh_raw,    "left_hand":        lh_raw,
            "right_hand_norm":  rh_norm,   "left_hand_norm":   lh_norm,
            "palm_direction":   palm_dir,  "hand_orientation": hand_ori,
        },
        "non_manual": {
            "face_landmarks": face_raw,
            "eyebrow_left":   eb_left,    "eyebrow_right":  eb_right,
            "lip_shape":      lip,
            "head_pose":      head_pose,
            "body_lean":      {"shoulder_angle": sh_angle,
                               "torso_direction": torso_dir},
        },
        "signing_space": {
            "hand_relative_to_face":     rel_face,
            "hand_relative_to_shoulder": rel_sh,
        },
        "movement": {"velocity": [0.0, 0.0, 0.0]},
    }


def _nm_label_from_annotation(ann: dict) -> Optional[str]:
    """AIHub 비수지 주석 dict → 내부 라벨 문자열."""
    nm  = ann.get("non_manual", ann)
    raw = (nm.get("sentence_type") or nm.get("sentenceType")
           or nm.get("type") or nm.get("label", ""))
    return _NM_TYPE_MAP.get(str(raw).strip(), None)


def _nm_label_heuristic(frames: List[dict]) -> str:
    """
    프레임 특징으로 비수지 라벨을 추정 (주석 없을 때 폴백).
    - 눈썹 올림 평균 > 0.02 → question
    - 고개 yaw 표준편차 > 5 + 반전 3회 이상 → negative
    - 나머지 → neutral
    """
    eyebrow_raises, yaws = [], []

    for fr in frames:
        nm   = fr.get("non_manual", {})
        fl   = nm.get("face_landmarks")
        eb_l = nm.get("eyebrow_left")
        eb_r = nm.get("eyebrow_right")
        hp   = nm.get("head_pose", {})

        if fl and eb_l and eb_r:
            if len(fl) >= 362:
                ely, ery = fl[362][1], fl[133][1]
            elif len(fl) >= 46:
                ely, ery = fl[45][1], fl[36][1]
            else:
                ely = ery = None
            if ely is not None:
                lr = ely - sum(p[1] for p in eb_l) / len(eb_l)
                rr = ery - sum(p[1] for p in eb_r) / len(eb_r)
                eyebrow_raises.append((lr + rr) / 2)

        yaws.append(hp.get("yaw", 0.0))

    if eyebrow_raises and sum(eyebrow_raises) / len(eyebrow_raises) > 0.02:
        return "question"

    if len(yaws) >= 10:
        arr = np.array(yaws)
        if np.std(arr) > 5.0:
            chg = int(np.sum(np.diff(np.sign(np.diff(arr))) != 0))
            if chg >= 3:
                return "negative"

    return "neutral"


# ══════════════════════════════════════════════════════════════════════
# AIHub .part0 ZIP 직접 로더 (OpenPose 포맷)
# ══════════════════════════════════════════════════════════════════════

def _flat_to_pts(flat: list, n: int) -> Optional[List]:
    """OpenPose 플랫 배열 [x,y,c, x,y,c,...] → [[x,y,0.0], ...] (n개)."""
    if not flat or len(flat) < n * 3:
        return None
    return [[flat[i*3], flat[i*3+1], 0.0] for i in range(n)]


def _parse_openpose_frame(people: dict) -> dict:
    """AIHub OpenPose people dict → 내부 frame_data 포맷."""
    rh_pts   = _flat_to_pts(people.get("hand_right_keypoints_2d", []), 21)
    lh_pts   = _flat_to_pts(people.get("hand_left_keypoints_2d",  []), 21)
    pose_pts = _flat_to_pts(people.get("pose_keypoints_2d",        []), 25)
    face_pts = _flat_to_pts(people.get("face_keypoints_2d",        []), 70)
    return _parse_aihub_frame({
        "right_hand": rh_pts, "left_hand": lh_pts,
        "pose": pose_pts,     "face": face_pts,
    })


_WORD_ID_RE = re.compile(r'(?:WORD|SEN)(\d+)', re.IGNORECASE)


def _extract_word_id(name: str) -> Optional[str]:
    """파일명에서 단어 ID를 추출. 'NIA_SL_WORD0001_...' → '0001'."""
    m = _WORD_ID_RE.search(Path(name).name)
    return m.group(1) if m else None


def _build_label_map(morpheme_zip_paths: List[Path]) -> Dict[str, str]:
    """
    morpheme ZIP 파일들을 스캔하여 {word_id: label} 딕셔너리 생성.
    각 word_id당 첫 번째 morpheme 파일만 읽어 빠르게 처리.
    """
    label_map: Dict[str, str] = {}
    for zip_path in morpheme_zip_paths:
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                seen_ids: set = set()
                for name in zf.namelist():
                    if not name.endswith("_morpheme.json"):
                        continue
                    wid = _extract_word_id(name)
                    if not wid or wid in seen_ids:
                        continue
                    seen_ids.add(wid)
                    try:
                        raw = json.loads(zf.read(name).decode("utf-8"))
                        attrs = (raw.get("data") or [{}])[0].get("attributes", [])
                        label = attrs[0].get("name", "").strip() if attrs else ""
                        if label:
                            label_map.setdefault(wid, label)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [경고] morpheme ZIP 읽기 실패: {zip_path.name}: {e}")
    print(f"  단어 라벨 맵: {len(label_map)}개 단어 ID")
    return label_map


def load_aihub_from_zips(
    label_map: Dict[str, str],
    keypoint_zip_paths: List[Path],
    target_words: Optional[set] = None,
    max_clips_per_word: int = 50,
) -> List[dict]:
    """
    keypoint .part0 ZIP 파일들에서 클립을 읽어 내부 샘플 포맷으로 변환.
    label_map: {word_id → label}
    target_words: 포함할 단어 집합 (None 이면 전체)
    """
    samples: List[dict] = []
    word_counts: Dict[str, int] = defaultdict(int)

    for zip_path in keypoint_zip_paths:
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                all_names = zf.namelist()
                json_names = [n for n in all_names
                              if n.endswith("_keypoints.json")]

                # 클립 디렉터리별로 그룹화
                clip_groups: Dict[str, List[str]] = defaultdict(list)
                for name in json_names:
                    clip_groups[str(Path(name).parent)].append(name)

                for clip_dir, frame_files in sorted(clip_groups.items()):
                    clip_name = Path(clip_dir).name
                    wid = _extract_word_id(clip_name)
                    if not wid:
                        continue

                    label = label_map.get(wid, "")
                    if not label:
                        continue
                    if target_words and label not in target_words:
                        continue
                    if word_counts[label] >= max_clips_per_word:
                        continue

                    frames = []
                    for ff in sorted(frame_files):
                        try:
                            data  = json.loads(zf.read(ff).decode("utf-8"))
                            ppl   = data.get("people", {})
                            frame = _parse_openpose_frame(ppl)
                            frames.append(frame)
                        except Exception:
                            pass

                    if frames:
                        samples.append({
                            "word":      label,
                            "frames":    frames,
                            "sign_type": "static",   # RF로 시퀀스 평균 특징 학습 (TF 없이)
                            "nm_label":  "neutral",
                            "source":    "aihub",
                        })
                        word_counts[label] += 1

        except Exception as e:
            print(f"  [경고] keypoint ZIP 읽기 실패: {zip_path.name}: {e}")

    found = len(word_counts)
    total = sum(word_counts.values())
    print(f"  ZIP 로드 완료: {found}개 단어, {total}개 클립, {len(samples)}개 샘플")
    return samples


def _find_part0_zips(base_dir: Path) -> List[Path]:
    """하위 디렉터리에서 .part0 파일 전부 반환."""
    return sorted(base_dir.rglob("*.part0")) if base_dir.exists() else []


def load_aihub_samples(data_dir: Path,
                       sign_type: str = "static") -> List[dict]:
    """
    AIHub 다운로드 폴더에서 JSON 키포인트 파일을 읽어
    내부 샘플 포맷으로 변환.

    폴더 구조: data_dir/{category}/{label}/*.json
    """
    samples: List[dict] = []
    if not data_dir.exists():
        return samples

    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for label_dir in sorted(cat_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for jf in sorted(label_dir.glob("*.json")):
                try:
                    with open(jf, encoding="utf-8") as f:
                        raw = json.load(f)

                    # AIHub 변형 A: {"frames": [...]}
                    # AIHub 변형 B: {"annotations": [{"keypoints": {...}}]}
                    raw_frames = (raw.get("frames")
                                  or [a for a in raw.get("annotations", [])])
                    if not raw_frames:
                        continue

                    frames = [_parse_aihub_frame(fr) for fr in raw_frames]
                    if not frames:
                        continue

                    # 비수지 라벨 (주석 우선, 없으면 휴리스틱)
                    nm_label = _nm_label_from_annotation(raw) or \
                               _nm_label_heuristic(frames)

                    samples.append({
                        "word":      label,
                        "frames":    frames,
                        "sign_type": raw.get("sign_type", sign_type),
                        "nm_label":  nm_label,
                        "source":    "aihub",
                    })
                except Exception as e:
                    print(f"  [경고] AIHub JSON 파싱 실패 {jf.name}: {e}")

    print(f"AIHub 로드: {data_dir.name}  {len(samples)}개 샘플")
    return samples


def load_all_samples() -> Tuple[List[dict], List[dict]]:
    """
    data/raw + data/aihub + data/aihub_sentences 전부 로드.
    Returns: (static_samples, dynamic_samples)
    """
    static_list: List[dict] = []
    dynamic_list: List[dict] = []

    # ── data/raw (collect_data.py 수집 결과) ──────────────────────────
    if DATA_ROOT.exists():
        for word_dir in sorted(DATA_ROOT.iterdir()):
            if not word_dir.is_dir():
                continue
            word = word_dir.name
            for jf in sorted(word_dir.glob("sample_*.json")):
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                    frames = data.get("frames", [])
                    if not frames:
                        continue
                    item = {
                        "word":      word,
                        "frames":    frames,
                        "sign_type": data.get("sign_type", "static"),
                        "nm_label":  data.get("nm_label", "neutral"),
                        "source":    "raw",
                    }
                    (dynamic_list if item["sign_type"] == "dynamic"
                     else static_list).append(item)
                except Exception as e:
                    print(f"  [경고] {jf.name} 로드 실패: {e}")
    else:
        print(f"[정보] data/raw 폴더 없음 — AIHub 데이터만 사용")

    # ── AIHub data/aihub + data/aihub_sentences (.part0 ZIP 직접 로드) ──
    word_part0   = _find_part0_zips(DATA_AIHUB)
    sent_part0   = _find_part0_zips(DATA_SENT)
    all_part0    = word_part0 + sent_part0

    if all_part0:
        # morpheme 파일에서 word_id → 라벨 매핑 구축
        morpheme_paths = [p for p in all_part0
                          if "morpheme" in p.name.lower()]
        keypoint_paths = [p for p in all_part0
                          if "keypoint" in p.name.lower()]

        print(f"\n[AIHub] morpheme ZIP: {len(morpheme_paths)}개 "
              f"/ keypoint ZIP: {len(keypoint_paths)}개")

        label_map = _build_label_map(morpheme_paths)

        # 목표 어휘 (downloader의 WORD_CATEGORIES + SENTENCE_LIST)
        try:
            _dl_path = str(PROJECT_ROOT)
            if _dl_path not in sys.path:
                sys.path.insert(0, _dl_path)
            from data_collection.aihub_downloader import ALL_WORDS, SENTENCE_LIST as _SENT_LIST
            target_words = set(ALL_WORDS.keys())
            SENTENCE_LIST = _SENT_LIST
        except ImportError:
            target_words = None   # 전체 로드
            SENTENCE_LIST = {}    # type: ignore
        if target_words is not None:
            for sents in SENTENCE_LIST.values():
                target_words.update(sents)

        found_in_map = {lbl for lbl in label_map.values()
                        if lbl in target_words}
        print(f"  목표 어휘 {len(target_words)}개 중 AIHub에 {len(found_in_map)}개 존재")

        aihub_samples = load_aihub_from_zips(
            label_map, keypoint_paths,
            target_words=target_words,
            max_clips_per_word=50,
        )
        for s in aihub_samples:
            (dynamic_list if s["sign_type"] == "dynamic"
             else static_list).append(s)
    else:
        # 레거시: 추출된 JSON 파일 폴더 방식
        for s in load_aihub_samples(DATA_AIHUB):
            (dynamic_list if s["sign_type"] == "dynamic"
             else static_list).append(s)
        for s in load_aihub_samples(DATA_SENT):
            (dynamic_list if s["sign_type"] == "dynamic"
             else static_list).append(s)

    print(f"로드 완료: 정적 {len(static_list)}개 / 동적 {len(dynamic_list)}개")
    return static_list, dynamic_list


# ══════════════════════════════════════════════════════════════════════
# 좌우 반전 데이터 증강
# ══════════════════════════════════════════════════════════════════════

def _flip_coord3(pts: Optional[List]) -> Optional[List]:
    """[[x,y,z], ...] 에서 x 부호 반전."""
    if pts is None:
        return None
    return [[-p[0], p[1], p[2]] for p in pts]


def _flip_frame(frame: dict) -> dict:
    """
    단일 프레임 좌우 반전:
    - 손 좌표 x 반전 + 왼손/오른손 교체
    - 손바닥 방향 x 반전
    - 손 회전 각도: π − θ
    - 수화 공간 x 반전
    - 고개 yaw 반전
    - 어깨·몸통 각도 부호 반전
    - 손목 속도 x 반전
    """
    f = copy.deepcopy(frame)
    m = f.setdefault("manual", {})

    # 손 교체 + x 반전
    rh, lh = m.get("right_hand"), m.get("left_hand")
    rn, ln = m.get("right_hand_norm"), m.get("left_hand_norm")
    m["right_hand"]      = _flip_coord3(lh)
    m["left_hand"]       = _flip_coord3(rh)
    m["right_hand_norm"] = _flip_coord3(ln)
    m["left_hand_norm"]  = _flip_coord3(rn)

    # 손바닥 방향 x 반전
    pd = m.get("palm_direction", [0.0, 0.0, 0.0])
    m["palm_direction"] = [-pd[0], pd[1], pd[2]]

    # 손 회전 각도 (atan2(y,x) → atan2(y,-x) ≈ π − θ)
    ori = m.get("hand_orientation", 0.0)
    m["hand_orientation"] = float(math.pi - ori)

    # 수화 공간 x 반전
    ss = f.setdefault("signing_space", {})
    for key in ("hand_relative_to_face", "hand_relative_to_shoulder"):
        v = ss.get(key, [0.0, 0.0, 0.0])
        ss[key] = [-v[0], v[1], v[2]]

    # 비수지: 고개 yaw, 어깨·몸통 각도
    nm = f.setdefault("non_manual", {})
    hp = nm.setdefault("head_pose", {})
    hp["yaw"] = -float(hp.get("yaw", 0.0))
    bl = nm.setdefault("body_lean", {})
    bl["shoulder_angle"]  = -float(bl.get("shoulder_angle",  0.0))
    bl["torso_direction"] = -float(bl.get("torso_direction", 0.0))

    # 손목 속도 x 반전
    mv = f.setdefault("movement", {})
    vel = mv.get("velocity", [0.0, 0.0, 0.0])
    mv["velocity"] = [-vel[0], vel[1], vel[2]]

    return f


def augment_flip(samples: List[dict]) -> List[dict]:
    """원본 + 좌우 반전본 = 2배 증강."""
    flipped = [
        {
            "word":      s["word"],
            "frames":    [_flip_frame(fr) for fr in s["frames"]],
            "sign_type": s["sign_type"],
        }
        for s in samples
    ]
    return samples + flipped


def _perturb_hand(pts: Optional[List], noise_frac: float,
                  rng: np.random.RandomState) -> Optional[List]:
    """손목 기준 정규화 좌표에 비율 기반 가우시안 노이즈 추가."""
    if pts is None:
        return None
    arr = np.array(pts, dtype=np.float32)
    # 손 크기 = 손목→중지끝 거리 (좌표 스케일 독립)
    scale = float(np.linalg.norm(arr[MID_TIP_IDX] - arr[0]))
    if scale < 1e-6:
        scale = 1.0
    arr += rng.normal(0, noise_frac * scale, arr.shape).astype(np.float32)
    return arr.tolist()


def _rotate_hand_pts(pts: Optional[List], cos_a: float,
                     sin_a: float) -> Optional[List]:
    """XY 평면 기준 2D 회전."""
    if pts is None:
        return None
    return [
        [p[0] * cos_a - p[1] * sin_a,
         p[0] * sin_a + p[1] * cos_a,
         p[2] if len(p) > 2 else 0.0]
        for p in pts
    ]


def _noise_frame(frame: dict, noise_frac: float,
                 rng: np.random.RandomState) -> dict:
    f = copy.deepcopy(frame)
    m = f.setdefault("manual", {})
    for key in ("right_hand", "left_hand", "right_hand_norm", "left_hand_norm"):
        m[key] = _perturb_hand(m.get(key), noise_frac, rng)
    return f


def _rotate_frame(frame: dict, cos_a: float, sin_a: float) -> dict:
    f = copy.deepcopy(frame)
    m = f.setdefault("manual", {})
    for key in ("right_hand", "left_hand", "right_hand_norm", "left_hand_norm"):
        m[key] = _rotate_hand_pts(m.get(key), cos_a, sin_a)
    pd = m.get("palm_direction", [0.0, 0.0, 0.0])
    m["palm_direction"] = [
        pd[0] * cos_a - pd[1] * sin_a,
        pd[0] * sin_a + pd[1] * cos_a,
        pd[2],
    ]
    ori = m.get("hand_orientation", 0.0)
    m["hand_orientation"] = ori + math.atan2(sin_a, cos_a)
    # 수화 공간 회전
    ss = f.setdefault("signing_space", {})
    for skey in ("hand_relative_to_face", "hand_relative_to_shoulder"):
        v = ss.get(skey, [0.0, 0.0, 0.0])
        ss[skey] = [v[0]*cos_a - v[1]*sin_a,
                    v[0]*sin_a + v[1]*cos_a,
                    v[2]]
    return f


def augment_advanced(
    samples: List[dict],
    n_noise: int = 4,
    noise_frac: float = 0.04,
    angles_deg: List[float] = None,
) -> List[dict]:
    """
    노이즈(×n_noise) + 회전(×len(angles))으로 증강.
    flip은 별도로 augment_flip에서 처리.
    원본 포함 최대 (1 + n_noise + len(angles))배 증강.
    """
    if angles_deg is None:
        angles_deg = [-12.0, -6.0, 6.0, 12.0]

    rng = np.random.RandomState(42)
    result = list(samples)  # 원본 포함

    # 회전 각도별 sin/cos 미리 계산
    rot_params = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
                  for a in angles_deg]

    for s in samples:
        base_frames = s["frames"]

        # 노이즈 증강
        for _ in range(n_noise):
            result.append({
                "word":      s["word"],
                "sign_type": s.get("sign_type", "static"),
                "frames":    [_noise_frame(fr, noise_frac, rng)
                               for fr in base_frames],
            })

        # 회전 증강
        for cos_a, sin_a in rot_params:
            result.append({
                "word":      s["word"],
                "sign_type": s.get("sign_type", "static"),
                "frames":    [_rotate_frame(fr, cos_a, sin_a)
                               for fr in base_frames],
            })

    return result


# ══════════════════════════════════════════════════════════════════════
# 정적 데이터셋 준비
# ══════════════════════════════════════════════════════════════════════

def prepare_static_dataset(
    samples: List[dict],
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    각 샘플의 모든 프레임 특징을 평균 + 표준편차로 집계 (308D).
    평균만 쓰면 정보 손실이 크므로 mean||std 연결.
    """
    X, y = [], []
    for s in samples:
        vecs = np.array([extract_frame_vector(fr) for fr in s["frames"]],
                        dtype=np.float32)   # (n_frames, 154)
        feat = np.concatenate([vecs.mean(axis=0), vecs.std(axis=0)])  # 308D
        X.append(feat)
        y.append(s["word"])

    X_arr = np.array(X, dtype=np.float32)
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"  정적 특징 차원: {X_arr.shape[1]}D  (프레임 평균+표준편차 연결)")
    return X_arr, y_enc, le


# ══════════════════════════════════════════════════════════════════════
# 동적 데이터셋 준비
# ══════════════════════════════════════════════════════════════════════

def _adjust_seq(seq: np.ndarray, target: int) -> np.ndarray:
    """시퀀스 길이를 target으로 조정 (균등 샘플링 / 제로 패딩)."""
    n, d = seq.shape
    if n == target:
        return seq
    if n > target:
        idx = np.round(np.linspace(0, n - 1, target)).astype(int)
        return seq[idx]
    pad = np.zeros((target - n, d), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


def prepare_dynamic_dataset(
    samples: List[dict],
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    각 샘플 → (seq_len, 158D) 시퀀스 텐서.
    158D = 154D 정적 특징 + 3D 속도 + 1D 손모양 변화량.
    """
    X, y = [], []
    for s in samples:
        frames = s["frames"]
        seq    = []
        prev_rn: Optional[np.ndarray] = None

        for fr in frames:
            vel = fr.get("movement", {}).get("velocity", [0.0, 0.0, 0.0])
            rn_raw = fr.get("manual", {}).get("right_hand_norm")
            if rn_raw and prev_rn is not None:
                hand_chg = float(np.linalg.norm(
                    np.array(rn_raw, dtype=np.float32) - prev_rn
                ))
            else:
                hand_chg = 0.0
            prev_rn = np.array(rn_raw, dtype=np.float32) if rn_raw else None

            seq.append(extract_frame_vector(fr, velocity=vel,
                                             hand_change=hand_chg))

        seq_arr = np.array(seq, dtype=np.float32)       # (n, 158)
        seq_adj = _adjust_seq(seq_arr, seq_len)          # (seq_len, 158)
        X.append(seq_adj)
        y.append(s["word"])

    X_arr = np.array(X, dtype=np.float32)               # (N, seq_len, 158)
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"  동적 특징 차원: {X_arr.shape[2]}D × {seq_len}프레임")
    return X_arr, y_enc, le


# ══════════════════════════════════════════════════════════════════════
# 정적 모델: RandomForest
# ══════════════════════════════════════════════════════════════════════

def train_static_model(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> Tuple[RandomForestClassifier, StandardScaler]:

    print(f"\n{_BOLD}{'═'*58}{_RESET}")
    print(f"{_BOLD}  정적 수화 분류기  (RandomForest  n_estimators=200){_RESET}")
    print(f"{'═'*58}")
    print(f"  샘플: {X.shape[0]}개  |  특징: {X.shape[1]}D  |  클래스: {len(le.classes_)}개")

    # 표준화
    scaler  = StandardScaler()
    X_scale = scaler.fit_transform(X)

    # 데이터 충분 여부 확인
    min_class = np.bincount(y).min()
    n_splits  = min(5, int(min_class))

    if X.shape[0] < 10 or len(le.classes_) < 2:
        print("  [경고] 데이터 부족 — 5-Fold CV 생략, 단순 학습만 수행")
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                        n_jobs=-1, class_weight="balanced")
        model.fit(X_scale, y)
        print(f"  학습 완료 (테스트 분리 불가)")
        return model, scaler

    # 5-Fold 교차 검증
    if n_splits >= 2:
        kf     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RANDOM_SEED)
        cv_acc = cross_val_score(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                   n_jobs=-1, class_weight="balanced"),
            X_scale, y, cv=kf, scoring="accuracy",
        )
        print(f"\n  {n_splits}-Fold CV 정확도: "
              f"{cv_acc.mean():.2%} ± {cv_acc.std():.2%}")

    # 최종 Train/Test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scale, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    print(f"\n  {_GREEN}테스트 정확도: {acc:.2%}{_RESET}\n")
    print(classification_report(
        y_te, y_pred,
        target_names=[le.classes_[i] for i in sorted(set(y_te))],
        zero_division=0,
    ))
    _print_confusion(y_te, y_pred, le.classes_, "정적 수화")
    return model, scaler


# ══════════════════════════════════════════════════════════════════════
# 동적 모델: LSTM
# ══════════════════════════════════════════════════════════════════════

def _build_lstm(n_cls: int, seq_len: int, n_feat: int) -> "Sequential":
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(seq_len, n_feat)),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(n_cls, activation="softmax"),
    ], name="sign_lstm")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_dynamic_model(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> Optional["Sequential"]:

    if not HAS_TF:
        print(f"\n{_RED}[건너뜀]{_RESET} TensorFlow 미설치 → 동적 모델 학습 불가")
        print("  pip install tensorflow  후 재실행하세요.")
        return None

    print(f"\n{_BOLD}{'═'*58}{_RESET}")
    print(f"{_BOLD}  동적 수화 분류기  (LSTM  128 → 64 → softmax){_RESET}")
    print(f"{'═'*58}")
    n, seq_len, n_feat = X.shape
    n_cls = len(le.classes_)
    print(f"  샘플: {n}개  |  시퀀스: {seq_len}프레임 × {n_feat}D  |  클래스: {n_cls}개")

    if n < 10 or n_cls < 2:
        print("  [경고] 데이터 부족 — 기본 학습만 수행")
        model = _build_lstm(n_cls, seq_len, n_feat)
        model.fit(X, y, epochs=30, batch_size=8, verbose=0)
        return model

    # 특징 정규화 (샘플 × 프레임 × 특징 → mean/std)
    flat  = X.reshape(-1, n_feat)
    mu    = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-8
    X_norm = (X - mu) / sigma
    joblib.dump({"mean": mu, "std": sigma}, MODELS_DIR / "dynamic_scaler.pkl")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y,
    )

    model = _build_lstm(n_cls, seq_len, n_feat)
    model.summary()

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True,
                      monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6, verbose=0),
    ]

    model.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=150,
        batch_size=min(32, max(8, len(X_tr) // 6)),
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc    = accuracy_score(y_te, y_pred)
    print(f"\n  {_GREEN}테스트 정확도: {acc:.2%}{_RESET}\n")
    print(classification_report(
        y_te, y_pred,
        target_names=[le.classes_[i] for i in sorted(set(y_te))],
        zero_division=0,
    ))
    _print_confusion(y_te, y_pred, le.classes_, "동적 수화")
    return model


# ══════════════════════════════════════════════════════════════════════
# 혼동 행렬 출력 + 저장
# ══════════════════════════════════════════════════════════════════════

def _print_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                     classes: np.ndarray, title: str) -> None:
    present      = sorted(set(y_true.tolist() + y_pred.tolist()))
    p_classes    = [classes[i] for i in present]
    cm           = confusion_matrix(y_true, y_pred, labels=present)

    nw = max((len(c) for c in p_classes), default=4) + 1
    vw = max(4, len(str(int(cm.max()))) + 1)

    print(f"\n  ── 혼동 행렬: {title} ──────────────────")
    # 헤더
    print(" " * (nw + 2) + "".join(c[:vw-1].rjust(vw) for c in p_classes))
    print(" " * (nw + 2) + "─" * (vw * len(p_classes)))
    # 행
    for i, c in enumerate(p_classes):
        row = c.ljust(nw) + " │"
        for j in range(len(p_classes)):
            v = cm[i, j]
            s = str(v).rjust(vw)
            if i == j:
                row += f"{_GREEN}{s}{_RESET}"
            elif v > 0:
                row += f"{_RED}{s}{_RESET}"
            else:
                row += " " * vw
        print(row)

    # 상위 혼동 쌍
    errors = sorted(
        [(cm[i, j], p_classes[i], p_classes[j])
         for i in range(len(p_classes))
         for j in range(len(p_classes)) if i != j and cm[i, j] > 0],
        reverse=True,
    )
    if errors:
        print(f"\n  {_CYAN}주요 혼동 (상위 5):{_RESET}")
        for cnt, tr, pr in errors[:5]:
            print(f"    '{tr}' → '{pr}'  {cnt}회")

    # matplotlib 저장
    if HAS_MPL and len(p_classes) > 0:
        _save_confusion_plot(cm, p_classes, title)


def _save_confusion_plot(cm: np.ndarray, labels: List[str], title: str) -> None:
    sz  = max(6, len(labels))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    im  = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    mx = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > mx * 0.55 else "black")
    ax.set_xlabel("예측 (Predicted)", fontsize=9)
    ax.set_ylabel("실제 (True)",       fontsize=9)
    ax.set_title(f"혼동 행렬 — {title}", fontsize=10)
    plt.tight_layout()
    slug = title.replace(" ", "_").replace("/", "_")
    out  = MODELS_DIR / f"confusion_{slug}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  혼동 행렬 이미지 저장: {out}")


# ══════════════════════════════════════════════════════════════════════
# 특징 중요도 출력 (정적 RF)
# ══════════════════════════════════════════════════════════════════════

def _print_feature_importance(model: RandomForestClassifier,
                               n_feat_per_group: int = 154) -> None:
    imp = model.feature_importances_
    # 그룹별 집계 (mean / std 결합이므로 2 × 154D = 308D)
    half = min(n_feat_per_group, len(imp) // 2)
    groups = {
        "오른손 정규화(63D)":    imp[:63].sum(),
        "왼손 정규화(63D)":     imp[63:126].sum(),
        "손바닥방향+회전(4D)":   imp[126:130].sum(),
        "비수지 신호(18D)":     imp[130:148].sum(),
        "수화 공간(6D)":        imp[148:154].sum(),
    }
    print("\n  특징 그룹별 중요도 (평균 파트):")
    total = sum(groups.values()) or 1
    for name, val in sorted(groups.items(), key=lambda x: -x[1]):
        bar = "█" * int(val / total * 30)
        print(f"    {name:22s}  {bar:30s}  {val/total:.1%}")


# ══════════════════════════════════════════════════════════════════════
# 모델 저장
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# 비수지 데이터셋 준비 + 학습
# ══════════════════════════════════════════════════════════════════════

def prepare_non_manual_dataset(
    samples: List[dict],
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    각 샘플의 비수지 특징(18D)을 프레임 평균으로 집계.
    라벨: nm_label (neutral / question / negative / emphasis / command)
    """
    X, y = [], []
    for s in samples:
        nm_label = s.get("nm_label", "neutral")
        frames   = s["frames"]
        if not frames:
            continue

        vecs = np.array(
            [extract_non_manual(fr) for fr in frames],
            dtype=np.float32,
        )  # (n_frames, 18)
        feat = np.concatenate([vecs.mean(axis=0), vecs.std(axis=0)])  # 36D
        X.append(feat)
        y.append(nm_label)

    if not X:
        return np.empty((0, 36)), np.empty(0, dtype=int), LabelEncoder()

    X_arr = np.array(X, dtype=np.float32)
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"  비수지 특징 차원: {X_arr.shape[1]}D  |  클래스: {list(le.classes_)}")
    return X_arr, y_enc, le


def train_non_manual_model(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> Tuple[Optional[object], Optional[StandardScaler]]:

    print(f"\n{_BOLD}{'═'*58}{_RESET}")
    print(f"{_BOLD}  비수지 신호 분류기  (RandomForest  n_estimators=200){_RESET}")
    print(f"{'═'*58}")

    if X.shape[0] < 6 or len(le.classes_) < 2:
        print(f"  {_RED}[건너뜀]{_RESET} 비수지 데이터 부족 "
              f"(샘플:{X.shape[0]}  클래스:{len(le.classes_)})")
        print("  → 데이터 다운로드 후 재실행하세요.")
        return None, None

    print(f"  샘플: {X.shape[0]}개  |  특징: {X.shape[1]}D  |  클래스: {len(le.classes_)}개")

    scaler  = StandardScaler()
    X_scale = scaler.fit_transform(X)

    min_cls  = int(np.bincount(y).min())
    n_splits = min(5, min_cls)

    if n_splits >= 2:
        kf     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RANDOM_SEED)
        cv_acc = cross_val_score(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                   n_jobs=-1, class_weight="balanced"),
            X_scale, y, cv=kf, scoring="accuracy",
        )
        print(f"\n  {n_splits}-Fold CV 정확도: "
              f"{cv_acc.mean():.2%} ± {cv_acc.std():.2%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scale, y, test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y if min_cls >= 2 else None,
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED,
        n_jobs=-1, class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    print(f"\n  {_GREEN}테스트 정확도: {acc:.2%}{_RESET}\n")
    print(classification_report(
        y_te, y_pred,
        target_names=[le.classes_[i] for i in sorted(set(y_te))],
        zero_division=0,
    ))
    _print_confusion(y_te, y_pred, le.classes_, "비수지 신호")
    return model, scaler


def save_all(static_model, static_le, static_scaler,
             dynamic_model, dynamic_le,
             nm_model=None, nm_le=None, nm_scaler=None) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'═'*58}")
    print("  모델 파일 저장")
    print(f"{'═'*58}")

    # ── 정적 모델 ────────────────────────────────────────────────
    if static_model is not None:
        joblib.dump(static_model,  MODELS_DIR / "static_model.pkl")
        joblib.dump(static_le,     MODELS_DIR / "static_label_encoder.pkl")
        joblib.dump(static_scaler, MODELS_DIR / "static_scaler.pkl")
        joblib.dump(static_model,  MODELS_DIR / f"static_model_{stamp}.pkl")
        print(f"  {_GREEN}✓{_RESET} {MODELS_DIR / 'static_model.pkl'}")
        print(f"  {_GREEN}✓{_RESET} {MODELS_DIR / 'static_label_encoder.pkl'}")

        # gesture_classifier.py 호환 통합 파일
        joblib.dump(
            {"model": static_model, "label_encoder": static_le},
            MODELS_DIR / "gesture_model.pkl",
        )
        print(f"  {_GREEN}✓{_RESET} {MODELS_DIR / 'gesture_model.pkl'}"
              f"  (gesture_classifier.py 호환)")

    # ── 동적 모델 ────────────────────────────────────────────────
    if dynamic_model is not None and HAS_TF:
        path = str(MODELS_DIR / "dynamic_model.h5")
        dynamic_model.save(path)
        joblib.dump(dynamic_le, MODELS_DIR / "dynamic_label_encoder.pkl")
        dynamic_model.save(str(MODELS_DIR / f"dynamic_model_{stamp}.h5"))
        print(f"  {_GREEN}✓{_RESET} {path}")
        print(f"  {_GREEN}✓{_RESET} {MODELS_DIR / 'dynamic_label_encoder.pkl'}")

    # ── 비수지 모델 ──────────────────────────────────────────────────
    if nm_model is not None:
        joblib.dump(nm_model,  MODELS_DIR / "non_manual_model.pkl")
        joblib.dump(nm_le,     MODELS_DIR / "non_manual_label_encoder.pkl")
        joblib.dump(nm_scaler, MODELS_DIR / "non_manual_scaler.pkl")
        print(f"  {_GREEN}✓{_RESET} {MODELS_DIR / 'non_manual_model.pkl'}")
        print(f"  {_GREEN}✓{_RESET} {MODELS_DIR / 'non_manual_label_encoder.pkl'}")

    if static_model is None and dynamic_model is None and nm_model is None:
        print(f"  {_RED}저장할 모델이 없습니다.{_RESET}")


# ══════════════════════════════════════════════════════════════════════
# 데이터 통계 출력
# ══════════════════════════════════════════════════════════════════════

def _print_data_stats(samples: List[dict], label: str) -> None:
    from collections import Counter
    cnt = Counter(s["word"] for s in samples)
    print(f"\n  [{label}]  총 {len(samples)}개 샘플  |  단어 {len(cnt)}개")
    for word, n in sorted(cnt.items()):
        bar = "▓" * n + "░" * max(0, 10 - n)
        print(f"    {word:12s}  {bar}  {n:3d}개")


# ══════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    os.chdir(PROJECT_ROOT)

    print(f"\n{'━'*58}")
    print(f"{_BOLD}  수화 모델 학습 파이프라인{_RESET}")
    print(f"  프로젝트 루트: {PROJECT_ROOT}")
    print(f"  TensorFlow: {'✓' if HAS_TF else '✗ (미설치)'}"
          f"  |  matplotlib: {'✓' if HAS_MPL else '✗ (미설치)'}")
    print(f"{'━'*58}\n")

    # ── 1. 데이터 로드 ───────────────────────────────────────────
    static_raw, dynamic_raw = load_all_samples()

    if not static_raw and not dynamic_raw:
        print(f"\n{_RED}[오류]{_RESET} 수집된 데이터가 없습니다.")
        print(f"  먼저  python data_collection/collect_data.py  를 실행해 데이터를 수집하세요.")
        sys.exit(1)

    _print_data_stats(static_raw,  "원본 정적")
    _print_data_stats(dynamic_raw, "원본 동적")

    # ── 2. 데이터 증강 (노이즈 + 회전 + 좌우반전) ───────────────────
    print(f"\n[데이터 증강] 노이즈(×4) + 회전(×4) + 좌우반전(×2) ...")
    static_aug  = augment_flip(augment_advanced(static_raw))
    dynamic_aug = augment_flip(augment_advanced(dynamic_raw))
    print(f"  정적:  {len(static_raw):4d} → {len(static_aug):4d}"
          f"  (×{len(static_aug)//max(len(static_raw),1)})")
    print(f"  동적:  {len(dynamic_raw):4d} → {len(dynamic_aug):4d}"
          f"  (×{len(dynamic_aug)//max(len(dynamic_raw),1)})")

    static_model  = static_le  = static_scaler = None
    dynamic_model = dynamic_le = None

    # ── 3. 정적 모델 학습 ────────────────────────────────────────
    if static_aug:
        X_s, y_s, le_s = prepare_static_dataset(static_aug)
        static_model, static_scaler = train_static_model(X_s, y_s, le_s)
        static_le = le_s
        _print_feature_importance(static_model)
    else:
        print(f"\n{_CYAN}[정적 모델]{_RESET} 데이터 없음 — 건너뜁니다.")

    # ── 4. 동적 모델 학습 ────────────────────────────────────────
    if dynamic_aug:
        X_d, y_d, le_d = prepare_dynamic_dataset(dynamic_aug)
        dynamic_model = train_dynamic_model(X_d, y_d, le_d)
        dynamic_le    = le_d
    else:
        print(f"\n{_CYAN}[동적 모델]{_RESET} 데이터 없음 — 건너뜁니다.")

    # ── 5. 비수지 모델 학습 ──────────────────────────────────────────
    nm_model = nm_le = nm_scaler = None
    all_samples = static_aug + dynamic_aug
    if all_samples:
        X_nm, y_nm, le_nm = prepare_non_manual_dataset(all_samples)
        if X_nm.shape[0] >= 6:
            nm_model, nm_scaler = train_non_manual_model(X_nm, y_nm, le_nm)
            nm_le = le_nm
    else:
        print(f"\n{_CYAN}[비수지 모델]{_RESET} 데이터 없음 — 건너뜁니다.")

    # ── 6. 저장 ──────────────────────────────────────────────────
    save_all(static_model, static_le, static_scaler,
             dynamic_model, dynamic_le,
             nm_model, nm_le, nm_scaler)

    print(f"\n{'━'*58}")
    print(f"{_BOLD}{_GREEN}  학습 완료{_RESET}")
    print(f"{'━'*58}\n")


if __name__ == "__main__":
    main()
