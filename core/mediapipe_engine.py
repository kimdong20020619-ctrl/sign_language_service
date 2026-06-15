import math
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

# 한글 폰트 (Windows 기본 내장)
_KO_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
try:
    _FONT_WORD = ImageFont.truetype(_KO_FONT_PATH, 26)
    _FONT_STAT = ImageFont.truetype(_KO_FONT_PATH, 13)
except Exception:
    _FONT_WORD = ImageFont.load_default()
    _FONT_STAT = ImageFont.load_default()

# ── Model file paths ──────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).parent.parent / "models"

_MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

# MediaPipe 내장 제스처 → 한국어 단어 매핑 (데이터 수집 불필요)
BUILTIN_GESTURE_MAP: dict[str, str] = {
    "Thumb_Up":    "좋아요",
    "Thumb_Down":  "아니오",
    "Open_Palm":   "안녕하세요",
    "Closed_Fist": "잠깐만요",
    "Victory":     "감사합니다",
    "Pointing_Up": "주세요",
    "ILoveYou":    "도와주세요",
}

_GESTURE_KO = {**BUILTIN_GESTURE_MAP}


def _ensure_model(name: str) -> str:
    path = _MODELS_DIR / name
    if not path.exists():
        url = _MODEL_URLS[name]
        print(f"[MediaPipe] 모델 다운로드: {name} ...", flush=True)
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"[MediaPipe] 완료: {path}", flush=True)
    return str(path)


# ── Static connection lists (extracted once at import) ────────────────────────
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (c.start, c.end)
    for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
]
_FACE_TESSELATION: List[Tuple[int, int]] = [
    (c.start, c.end)
    for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
]
_FACE_OVAL: List[Tuple[int, int]] = [
    (c.start, c.end)
    for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
]

# ── Landmark index constants (FaceMesh 478-pt with attention mesh) ────────────
EYEBROW_LEFT_IDX  = [70, 63, 105, 66, 107]
EYEBROW_RIGHT_IDX = [336, 296, 334, 293, 300]

LIP_INDICES = [
     61, 185,  40,  39,  37,   0, 267, 269, 270, 409,
    146,  91, 181,  84,  17, 314, 405, 321, 375, 291,
]

FACE_POSE_IDX = [4, 152, 263, 33, 287, 57]

FACE_3D_REF = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)

_POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
]
_POSE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]


def _put_ko(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
    color: Tuple[int, int, int],
) -> None:
    """PIL로 한글 텍스트를 BGR numpy frame에 그린다."""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, fill=color, font=font)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class MediaPipeEngine:
    """Full-body sign-language AR overlay and feature extraction (Tasks API)."""

    FINGERTIP_INDICES = frozenset({4, 8, 12, 16, 20})

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ) -> None:
        self.dominant_hand = "right"

        hand_path = _ensure_model("hand_landmarker.task")
        face_path = _ensure_model("face_landmarker.task")
        pose_path = _ensure_model("pose_landmarker_lite.task")

        RunningMode = mp_vision.RunningMode

        self._hand_lm = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_path),
                running_mode=RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        )
        self._face_lm = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_path),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=detection_confidence,
                min_face_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                output_face_blendshapes=True,
            )
        )
        self._pose_lm = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_path),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=detection_confidence,
                min_pose_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        )

        # MediaPipe 내장 제스처 인식기 (데이터 수집 불필요)
        self._gesture_rec = None
        self._last_builtin: dict = {"name": None, "score": 0.0}
        try:
            gest_path = _ensure_model("gesture_recognizer.task")
            self._gesture_rec = mp_vision.GestureRecognizer.create_from_options(
                mp_vision.GestureRecognizerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=gest_path),
                    running_mode=RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=detection_confidence,
                    min_hand_presence_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence,
                )
            )
            print("[MediaPipe] GestureRecognizer 로드 완료")
        except Exception as e:
            print(f"[MediaPipe] GestureRecognizer 초기화 실패 (건너뜀): {e}")

        self._t0_ms = int(time.monotonic() * 1000)
        self._frame_skip = 0   # 짝수 프레임만 face/pose 처리
        self._last_face_res = None
        self._last_pose_res = None

    # ── public API ────────────────────────────────────────────────────────────

    def set_dominant_hand(self, hand: str) -> None:
        self.dominant_hand = hand.lower()

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[dict]]:
        ts_ms = int(time.monotonic() * 1000) - self._t0_ms
        self._frame_skip += 1

        # MediaPipe 입력은 절반 해상도로 (속도 향상)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        h_res = self._hand_lm.detect_for_video(mp_img, ts_ms)

        # 내장 제스처 인식 (HandLandmarker와 독립적으로 실행)
        if self._gesture_rec is not None:
            try:
                g_res = self._gesture_rec.recognize_for_video(mp_img, ts_ms)
                if g_res.gestures:
                    top = g_res.gestures[0][0]
                    name = top.category_name if top.category_name != "None" else None
                    self._last_builtin = {"name": name, "score": float(top.score)}
                else:
                    self._last_builtin = {"name": None, "score": 0.0}
            except Exception:
                pass

        # face/pose는 2프레임마다 1번만 처리 (추적 유지)
        if self._frame_skip % 2 == 0 or self._last_face_res is None:
            f_res = self._face_lm.detect_for_video(mp_img, ts_ms)
            p_res = self._pose_lm.detect_for_video(mp_img, ts_ms)
            self._last_face_res = f_res
            self._last_pose_res = p_res
        else:
            f_res = self._last_face_res
            p_res = self._last_pose_res

        annotated = frame.copy()

        hand_data = self._process_hands(h_res)
        face_data = self._process_face(f_res, frame.shape)
        pose_data = self._process_pose(p_res)

        self._draw_pose(annotated, p_res)
        self._draw_face_mesh(annotated, f_res)
        self._draw_signing_space_box(annotated, face_data)
        self._draw_hands(annotated, h_res)
        self._draw_feature_overlay(annotated, hand_data, face_data, pose_data)

        frame_data = self._build_frame_data(hand_data, face_data, pose_data)
        return annotated, frame_data

    def draw_word_overlay(
        self, frame: np.ndarray, word: str, confidence: float
    ) -> None:
        if not word:
            return
        text = f"{word}  {confidence:.0%}"
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        bbox = draw.textbbox((14, 10), text, font=_FONT_WORD)
        draw.rectangle([6, 5, bbox[2] + 8, bbox[3] + 6], fill=(0, 0, 0))
        draw.text((14, 10), text, fill=(50, 255, 130), font=_FONT_WORD)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self._hand_lm.close()
        self._face_lm.close()
        self._pose_lm.close()
        if self._gesture_rec is not None:
            self._gesture_rec.close()

    # ── hand processing ───────────────────────────────────────────────────────

    def _process_hands(self, results: mp_vision.HandLandmarkerResult) -> dict:
        no_hand = {
            "right_hand": None, "left_hand": None,
            "right_hand_norm": None, "left_hand_norm": None,
            "palm_direction": [0.0, 0.0, 0.0],
            "hand_orientation": 0.0,
            "has_hand": False,
        }
        if not results.hand_landmarks:
            return no_hand

        right_raw = right_norm = None
        left_raw  = left_norm  = None
        palm_dir  = [0.0, 0.0, 0.0]
        hand_ori  = 0.0

        for idx, lms in enumerate(results.hand_landmarks):
            # Handedness label: "Right" or "Left"
            label = "Right"
            if results.handedness and idx < len(results.handedness):
                label = results.handedness[idx][0].display_name

            lm_list = [[lm.x, lm.y, lm.z] for lm in lms]
            wrist = lm_list[0]
            norm = [
                [lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]]
                for lm in lms
            ]

            p0 = np.array(lm_list[0])
            v1 = np.array(lm_list[5])  - p0
            v2 = np.array(lm_list[17]) - p0
            n  = np.cross(v1, v2)
            mag = float(np.linalg.norm(n))
            if mag > 1e-6:
                palm_dir = (n / mag).tolist()

            mcp = np.array(lm_list[9])
            tip = np.array(lm_list[12])
            d = tip - mcp
            hand_ori = float(np.arctan2(float(d[1]), float(d[0])))

            # Frame is horizontally flipped (mirror mode), so MediaPipe's image-based
            # handedness is inverted relative to the user: "Left" label = user's right hand.
            if label == "Left":
                right_raw, right_norm = lm_list, norm
            else:
                left_raw,  left_norm  = lm_list, norm

        if self.dominant_hand == "left":
            right_raw,  left_raw  = left_raw,  right_raw
            right_norm, left_norm = left_norm, right_norm
            palm_dir = [-palm_dir[0], palm_dir[1], palm_dir[2]]

        return {
            "right_hand": right_raw, "left_hand": left_raw,
            "right_hand_norm": right_norm, "left_hand_norm": left_norm,
            "palm_direction": palm_dir, "hand_orientation": hand_ori,
            "has_hand": True,
        }

    # ── face processing ───────────────────────────────────────────────────────

    def _process_face(
        self, results: mp_vision.FaceLandmarkerResult, shape
    ) -> dict:
        empty = {
            "face_landmarks": None, "eyebrow_left": None,
            "eyebrow_right": None, "lip_shape": None,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "nose_tip": None,
            "expression": "neutral",
        }
        if not results.face_landmarks:
            return empty

        fl_raw = results.face_landmarks[0]
        lms = [[lm.x, lm.y, lm.z] for lm in fl_raw]

        expression = "neutral"
        if results.face_blendshapes:
            expression = self._classify_expression(results.face_blendshapes[0])

        return {
            "face_landmarks": lms,
            "eyebrow_left":   [lms[i] for i in EYEBROW_LEFT_IDX],
            "eyebrow_right":  [lms[i] for i in EYEBROW_RIGHT_IDX],
            "lip_shape":      [lms[i] for i in LIP_INDICES],
            "head_pose":      self._estimate_head_pose(lms, shape),
            "nose_tip":       lms[4],
            "expression":     expression,
        }

    @staticmethod
    def _classify_expression(blendshapes) -> str:
        """52개 blendshape 계수로 수화 비수지 신호 분류."""
        bs = {b.category_name: b.score for b in blendshapes}

        brow_up   = (bs.get("browInnerUp", 0)
                     + bs.get("browOuterUpLeft", 0)
                     + bs.get("browOuterUpRight", 0)) / 3
        brow_down = (bs.get("browDownLeft", 0)
                     + bs.get("browDownRight", 0)) / 2
        mouth_open = bs.get("jawOpen", 0)
        mouth_smile = (bs.get("mouthSmileLeft", 0)
                       + bs.get("mouthSmileRight", 0)) / 2
        mouth_frown = (bs.get("mouthFrownLeft", 0)
                       + bs.get("mouthFrownRight", 0)) / 2
        eye_squint = (bs.get("eyeSquintLeft", 0)
                      + bs.get("eyeSquintRight", 0)) / 2

        # 의문문: 눈썹 올라감 + 약간 입 열림
        if brow_up > 0.35 and mouth_open > 0.15:
            return "question"
        # 강조/감탄: 눈썹 올라감 + 입 크게 열림
        if brow_up > 0.30 and mouth_open > 0.35:
            return "emphasis"
        # 부정: 눈썹 내려감 + 찡그림
        if brow_down > 0.40 and eye_squint > 0.30:
            return "negative"
        # 기쁨: 미소
        if mouth_smile > 0.45:
            return "happy"
        # 슬픔: 입 내려감
        if mouth_frown > 0.35:
            return "sad"
        # 눈썹만 올라감: 의문
        if brow_up > 0.45:
            return "question"

        return "neutral"

    def _estimate_head_pose(self, lms: list, shape) -> dict:
        h, w = shape[:2]
        img_pts = np.array(
            [[lms[i][0] * w, lms[i][1] * h] for i in FACE_POSE_IDX],
            dtype=np.float64,
        )
        fl  = float(w)
        cam = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        try:
            ok, rvec, _ = cv2.solvePnP(
                FACE_3D_REF, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                return {
                    "pitch": float(angles[0]),
                    "yaw":   float(angles[1]),
                    "roll":  float(angles[2]),
                }
        except Exception:
            pass
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    # ── pose processing ───────────────────────────────────────────────────────

    def _process_pose(self, results: mp_vision.PoseLandmarkerResult) -> dict:
        empty = {
            "shoulder_angle": 0.0, "torso_direction": 0.0,
            "left_shoulder": None, "right_shoulder": None,
        }
        if not results.pose_landmarks:
            return empty

        lm = results.pose_landmarks[0]
        ls = np.array([lm[11].x, lm[11].y])
        rs = np.array([lm[12].x, lm[12].y])
        lh = np.array([lm[23].x, lm[23].y])
        rh = np.array([lm[24].x, lm[24].y])

        sh_diff   = rs - ls
        torso_vec = (lh + rh) / 2.0 - (ls + rs) / 2.0

        return {
            "shoulder_angle":  float(np.arctan2(sh_diff[1],   sh_diff[0])),
            "torso_direction": float(np.arctan2(torso_vec[0], torso_vec[1])),
            "left_shoulder":   [lm[11].x, lm[11].y, lm[11].z],
            "right_shoulder":  [lm[12].x, lm[12].y, lm[12].z],
        }

    # ── frame data assembly ───────────────────────────────────────────────────

    def _build_frame_data(
        self, hand_data: dict, face_data: dict, pose_data: dict
    ) -> Optional[dict]:
        if not hand_data["has_hand"]:
            return None

        rh    = hand_data["right_hand"]
        wrist = rh[0] if rh else [0.0, 0.0, 0.0]
        nose  = face_data.get("nose_tip") or [0.0, 0.0, 0.0]
        rs    = pose_data.get("right_shoulder") or [0.0, 0.0, 0.0]

        rel_face = [wrist[i] - nose[i] for i in range(3)]
        rel_sh   = [wrist[i] - rs[i]   for i in range(3)]

        return {
            "builtin_gesture": dict(self._last_builtin),
            "manual": {
                "right_hand":       hand_data["right_hand"],
                "left_hand":        hand_data["left_hand"],
                "right_hand_norm":  hand_data["right_hand_norm"],
                "left_hand_norm":   hand_data["left_hand_norm"],
                "palm_direction":   hand_data["palm_direction"],
                "hand_orientation": hand_data["hand_orientation"],
            },
            "non_manual": {
                "face_landmarks": face_data["face_landmarks"],
                "eyebrow_left":   face_data["eyebrow_left"],
                "eyebrow_right":  face_data["eyebrow_right"],
                "lip_shape":      face_data["lip_shape"],
                "head_pose":      face_data["head_pose"],
                "expression":     face_data.get("expression", "neutral"),
                "emotion":        face_data.get("expression", "neutral"),
                "sentence_type":  (face_data.get("expression", "neutral")
                                   if face_data.get("expression") in
                                   ("question", "negative", "emphasis")
                                   else "neutral"),
                "body_lean": {
                    "shoulder_angle":  pose_data["shoulder_angle"],
                    "torso_direction": pose_data["torso_direction"],
                },
            },
            "signing_space": {
                "hand_relative_to_face":     rel_face,
                "hand_relative_to_shoulder": rel_sh,
            },
            "movement": {
                "velocity": [0.0, 0.0, 0.0],
            },
        }

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_hands(self, frame: np.ndarray, results: mp_vision.HandLandmarkerResult) -> None:
        if not results.hand_landmarks:
            return
        h, w = frame.shape[:2]

        for idx, lms in enumerate(results.hand_landmarks):
            label = "Right"
            if results.handedness and idx < len(results.handedness):
                label = results.handedness[idx][0].display_name

            c_joint = (0, 190, 0)    if label == "Left" else (190, 70, 0)
            c_bone  = (0, 130, 0)    if label == "Left" else (130, 40, 0)
            c_tip   = (60, 255, 60)  if label == "Left" else (60, 110, 255)

            for s_i, e_i in HAND_CONNECTIONS:
                s = lms[s_i]
                e = lms[e_i]
                cv2.line(
                    frame,
                    (int(s.x * w), int(s.y * h)),
                    (int(e.x * w), int(e.y * h)),
                    c_bone, 2, cv2.LINE_AA,
                )

            for i, lm in enumerate(lms):
                x, y = int(lm.x * w), int(lm.y * h)
                if i in self.FINGERTIP_INDICES:
                    cv2.circle(frame, (x, y), 8, c_tip, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (x, y), 4, c_joint, -1, cv2.LINE_AA)

            wrist = lms[0]
            lx = int(wrist.x * w)
            ly = max(20, int(wrist.y * h) - 22)
            label_ko = "오른손" if label == "Left" else "왼손"
            _put_ko(frame, label_ko, (lx, ly), _FONT_STAT, c_tip)

    def _draw_face_mesh(self, frame: np.ndarray, results: mp_vision.FaceLandmarkerResult) -> None:
        if not results.face_landmarks:
            return
        h, w = frame.shape[:2]
        overlay = frame.copy()
        fl = results.face_landmarks[0]

        for s_i, e_i in _FACE_TESSELATION + _FACE_OVAL:
            s = fl[s_i]
            e = fl[e_i]
            cv2.line(
                overlay,
                (int(s.x * w), int(s.y * h)),
                (int(e.x * w), int(e.y * h)),
                (140, 140, 70), 1, cv2.LINE_AA,
            )

        for i in EYEBROW_LEFT_IDX + EYEBROW_RIGHT_IDX:
            lm = fl[i]
            cv2.circle(overlay,
                       (int(lm.x * w), int(lm.y * h)),
                       3, (0, 210, 255), -1, cv2.LINE_AA)

        for i in LIP_INDICES:
            lm = fl[i]
            cv2.circle(overlay,
                       (int(lm.x * w), int(lm.y * h)),
                       2, (255, 150, 0), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    def _draw_pose(self, frame: np.ndarray, results: mp_vision.PoseLandmarkerResult) -> None:
        if not results.pose_landmarks:
            return
        h, w = frame.shape[:2]
        lm = results.pose_landmarks[0]

        for s_i, e_i in _POSE_CONNECTIONS:
            s, e = lm[s_i], lm[e_i]
            sv = getattr(s, 'visibility', 1.0) or 1.0
            ev = getattr(e, 'visibility', 1.0) or 1.0
            if sv > 0.4 and ev > 0.4:
                cv2.line(
                    frame,
                    (int(s.x * w), int(s.y * h)),
                    (int(e.x * w), int(e.y * h)),
                    (200, 200, 200), 2, cv2.LINE_AA,
                )

        for i in _POSE_JOINTS:
            pt = lm[i]
            vis = getattr(pt, 'visibility', 1.0) or 1.0
            if vis > 0.4:
                cv2.circle(frame,
                           (int(pt.x * w), int(pt.y * h)),
                           5, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_signing_space_box(
        self, frame: np.ndarray, face_data: dict
    ) -> None:
        nose = face_data.get("nose_tip")
        if nose is None:
            return
        h, w = frame.shape[:2]
        cx = int(nose[0] * w)
        cy = int(nose[1] * h)
        bw, bh = int(w * 0.38), int(h * 0.48)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - int(bh * 0.72))
        x2 = min(w - 1, cx + bw // 2)
        y2 = min(h - 1, cy + int(bh * 0.28))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 180, 255), -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 1, cv2.LINE_AA)
        _put_ko(frame, "서명 공간", (x1 + 4, y1 + 4), _FONT_STAT, (80, 180, 255))

    def _draw_feature_overlay(
        self,
        frame: np.ndarray,
        hand_data: dict,
        face_data: dict,
        pose_data: dict,
    ) -> None:
        h, w = frame.shape[:2]
        hp  = face_data.get("head_pose", {})
        deg = math.degrees

        expr = face_data.get("expression", "neutral")
        lines: List[str] = []
        bg = self._last_builtin
        if bg.get("name") and bg["name"] in _GESTURE_KO:
            lines.append(f"✋ {_GESTURE_KO[bg['name']]}  {bg['score']:.0%}")
        lines += [
            f"표정:  {expr}",
            f"pitch: {hp.get('pitch', 0.0):+.1f}",
            f"yaw:   {hp.get('yaw',   0.0):+.1f}",
            f"roll:  {hp.get('roll',  0.0):+.1f}",
            f"shld:  {deg(pose_data.get('shoulder_angle',  0.0)):+.1f}",
            f"torso: {deg(pose_data.get('torso_direction', 0.0)):+.1f}",
        ]
        if hand_data["has_hand"]:
            lines.append(f"ori:   {deg(hand_data['hand_orientation']):+.1f}")
            pd = hand_data["palm_direction"]
            lines.append(f"palm:  {pd[0]:.2f} {pd[1]:.2f} {pd[2]:.2f}")

        pw = 210
        ph = len(lines) * 17 + 10
        px = w - pw - 4
        py = 4
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([px, py, px + pw, py + ph], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((px + 5, py + 3 + i * 17), line,
                      fill=(175, 255, 175), font=_FONT_STAT)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
