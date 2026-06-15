import os
import time
import collections
from typing import Optional, List, Tuple

import numpy as np
import joblib

# ── constants ─────────────────────────────────────────────────────────────────
# MediaPipe 내장 제스처 → 한국어 매핑 (데이터 수집 불필요)
BUILTIN_GESTURE_MAP: dict = {
    "Thumb_Up":    "좋아요",
    "Thumb_Down":  "아니오",
    "Open_Palm":   "안녕하세요",
    "Closed_Fist": "잠깐만요",
    "Victory":     "감사합니다",
    "Pointing_Up": "주세요",
    "ILoveYou":    "도와주세요",
}
BUILTIN_CONF_THR  = 0.72   # 내장 제스처 최소 신뢰도
BUILTIN_HOLD_SEC  = 0.5    # 제스처 사라져도 이 시간만큼 마지막 결과 유지

STATIC_VEL_THR   = 0.04   # normalised wrist displacement → static / dynamic boundary
CONFIRM_FRAMES   = 15     # consecutive identical predictions before confirming
CONFIDENCE_THR   = 0.20   # minimum confidence to accept a prediction
SPIKE_DROP_THR   = 0.30   # ignore frame if confidence drops more than this
STILL_TIME_SEC   = 1.5    # seconds of hand stillness → word boundary signal
NO_HAND_TIME_SEC = 3.0    # seconds without hand → sentence-end signal
STATIC_BUF_LEN   = 30     # rolling buffer depth for static-model aggregation
DYNAMIC_BUF_LEN  = 60     # rolling buffer depth for LSTM sequence
DYNAMIC_STEP     = 8      # run LSTM every N frames (saves CPU)
SEQ_LEN          = 60     # expected LSTM input length

# Feature-extraction constants (mirror train_model.py)
EYE_L_IDX   = 362
EYE_R_IDX   = 133
LIP_KEY_IDX = [0, 3, 5, 9, 14]

_DUMMY_WORDS = [
    # 공통기본
    "안녕하세요", "감사합니다", "죄송합니다", "네", "아니오",
    "도와주세요", "주세요", "잠깐만요", "괜찮아요", "알겠어요",
    # 일상동사
    "가다", "오다", "먹다", "마시다", "기다리다",
    # 감정·상태
    "아프다", "피곤하다", "배고프다", "행복하다", "위험하다",
    # 장소·교통
    "화장실", "병원", "지하철", "출구", "어디",
    # 식사·쇼핑
    "커피", "계산", "얼마", "포장", "영수증",
    # 긴급
    "응급", "도움", "신고하다", "구조", "사고",
]


class NonManualClassifier:
    """
    비수지 신호 분류기 (학습 모델 기반, 규칙 기반 폴백 포함).

    non_manual_model.pkl 이 있으면 RandomForest 로 분류하고,
    없으면 기존 규칙 기반(눈썹/고개) 로직으로 대체한다.

    분류 라벨: neutral / question / negative / emphasis / command
    """

    _NM_BUF_LEN = 30     # 롤링 버퍼 (~1 초 @ 30 fps)
    _INFER_STEP = 6      # N 프레임마다 추론 (CPU 절약)
    _CONF_THR   = 0.45   # 최소 신뢰도

    def __init__(self, models_dir: str) -> None:
        self._model   = None
        self._le      = None
        self._scaler  = None
        self._buf: collections.deque = collections.deque(maxlen=self._NM_BUF_LEN)
        self._yaw_buf: collections.deque = collections.deque(maxlen=20)
        self._step    = 0
        self._last: dict = {"is_question": False, "is_negative": False,
                            "is_emphasis": False, "emotion": "neutral",
                            "sentence_type": "neutral"}
        self._load(models_dir)

    def _load(self, models_dir: str) -> None:
        mp = os.path.join(models_dir, "non_manual_model.pkl")
        lp = os.path.join(models_dir, "non_manual_label_encoder.pkl")
        sp = os.path.join(models_dir, "non_manual_scaler.pkl")
        if os.path.exists(mp):
            try:
                self._model  = joblib.load(mp)
                self._le     = joblib.load(lp) if os.path.exists(lp) else None
                self._scaler = joblib.load(sp) if os.path.exists(sp) else None
                print("[NonManualClassifier] 비수지 모델 로드 완료")
            except Exception as e:
                print(f"[NonManualClassifier] 모델 로드 실패: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── 18D 특징 추출 (GestureClassifier._extract_frame_vector와 동일 규격) ──

    @staticmethod
    def _extract_18d(frame_data: dict) -> np.ndarray:
        nm   = frame_data.get("non_manual", {})
        fl   = nm.get("face_landmarks")
        eb_l = nm.get("eyebrow_left")
        eb_r = nm.get("eyebrow_right")
        lips = nm.get("lip_shape")
        hp   = nm.get("head_pose", {})
        bl   = nm.get("body_lean", {})

        if fl and eb_l and eb_r and len(fl) > max(362, 133):
            lr = float(fl[362][1] - np.mean([p[1] for p in eb_l]))
            rr = float(fl[133][1] - np.mean([p[1] for p in eb_r]))
        else:
            lr = rr = 0.0

        if lips and len(lips) >= 20:
            mouth_open = float(lips[14][1] - lips[5][1])
            lps        = np.array(lips, dtype=np.float32)
            key_pts    = lps[[0, 3, 5, 9, 14], :2].flatten()  # 10D
        else:
            mouth_open = 0.0
            key_pts    = np.zeros(10, dtype=np.float32)

        return np.array([
            lr, rr, mouth_open, *key_pts,
            hp.get("pitch", 0.0), hp.get("yaw", 0.0), hp.get("roll", 0.0),
            bl.get("shoulder_angle", 0.0), bl.get("torso_direction", 0.0),
        ], dtype=np.float32)  # 18D

    # ── 메인 분류 ─────────────────────────────────────────────────────

    def classify(self, frame_data: dict) -> dict:
        """frame_data → {is_question, is_negative, is_emphasis, emotion, sentence_type}"""
        feat = self._extract_18d(frame_data)
        self._buf.append(feat)
        self._yaw_buf.append(frame_data.get("non_manual", {})
                             .get("head_pose", {}).get("yaw", 0.0))

        self._step += 1
        if self._step < self._INFER_STEP:
            return self._last
        self._step = 0

        label = self._infer_model() or self._infer_rules()
        self._last = self._label_to_dict(label)
        return self._last

    def _infer_model(self) -> Optional[str]:
        if self._model is None or len(self._buf) < 10:
            return None
        buf_arr = np.array(list(self._buf), dtype=np.float32)
        feat    = np.concatenate([buf_arr.mean(0), buf_arr.std(0)]).reshape(1, -1)
        if self._scaler is not None:
            feat = self._scaler.transform(feat)
        try:
            proba    = self._model.predict_proba(feat)[0]
            best_idx = int(np.argmax(proba))
            if float(proba[best_idx]) >= self._CONF_THR and self._le is not None:
                return str(self._le.inverse_transform([best_idx])[0])
        except Exception:
            pass
        return None

    def _infer_rules(self) -> str:
        """모델 없거나 신뢰도 낮을 때 규칙 기반 폴백."""
        if not self._buf:
            return "neutral"
        buf_arr = np.array(list(self._buf), dtype=np.float32)
        lr_mean = float(buf_arr[:, 0].mean())
        rr_mean = float(buf_arr[:, 1].mean())
        mo_mean = float(buf_arr[:, 2].mean())

        # 의문문: 눈썹 올림
        if lr_mean > 0.025 or rr_mean > 0.025:
            return "question"

        # 부정: 고개 좌우 진동
        if len(self._yaw_buf) >= 10:
            yaw_arr = np.array(list(self._yaw_buf))
            if (np.std(yaw_arr) > 5.0
                    and int(np.sum(np.diff(np.sign(np.diff(yaw_arr))) != 0)) >= 3):
                return "negative"

        # 강조: 큰 입 벌림 + 눈썹 올림 동반
        if mo_mean > 0.06 and (lr_mean > 0.01 or rr_mean > 0.01):
            return "emphasis"

        return "neutral"

    @staticmethod
    def _label_to_dict(label: str) -> dict:
        return {
            "is_question":   label == "question",
            "is_negative":   label == "negative",
            "is_emphasis":   label == "emphasis",
            "emotion":       label if label not in ("question", "negative",
                                                     "emphasis", "command")
                             else "neutral",
            "sentence_type": label,
        }


class GestureClassifier:
    """Real-time sign-language classifier.

    Loads static_model.pkl (RandomForest, trained on 308-D mean+std features)
    and dynamic_model.h5 (LSTM, trained on (60, 158-D) sequences).  Falls back
    to an in-process dummy RF when no model files are found.

    Call classify(frame_data) every camera frame.  Returns a rich dict:

        {
            "word":        str | None,
            "confidence":  float,
            "sign_type":   "static" | "dynamic" | None,
            "non_manual":  {"is_question": bool, "is_negative": bool,
                            "emotion": str},
            "is_confirmed": bool,       # True after CONFIRM_FRAMES in a row
            "word_boundary": bool,      # True after STILL_TIME_SEC of stillness
            "sentence_end":  bool,      # True after NO_HAND_TIME_SEC without hand
        }
    """

    def __init__(self, model_path: str, confidence_threshold: float = CONFIDENCE_THR):
        self.model_path          = model_path
        self.confidence_threshold = max(confidence_threshold, CONFIDENCE_THR)

        # Static model (RandomForest)
        self._static_model  = None
        self._static_le     = None
        self._static_scaler = None

        # Dynamic model (LSTM)
        self._dynamic_model  = None
        self._dynamic_le     = None
        self._dynamic_mu     = None
        self._dynamic_sigma  = None

        # 비수지 분류기 (학습 모델 or 규칙 기반 폴백)
        models_dir = os.path.dirname(model_path)
        self._nm_classifier = NonManualClassifier(models_dir)

        # Per-frame state
        self._prev_wrist: Optional[np.ndarray] = None
        self._vel_history: collections.deque   = collections.deque(maxlen=10)

        # Confirmation state
        self._candidate_word:  Optional[str]   = None
        self._confirm_count:   int             = 0
        self._prev_conf:       float           = 0.0

        # Timing
        self._last_hand_time:  float           = time.time()
        self._still_start_time: Optional[float] = None

        # 내장 제스처 hold 버퍼 (각도 변화로 인한 순간적 인식 끊김 방지)
        self._builtin_hold_word: Optional[str]  = None
        self._builtin_hold_time: float          = 0.0

        # Rolling buffers for model inference
        self._static_buf:  collections.deque   = collections.deque(maxlen=STATIC_BUF_LEN)
        self._dynamic_buf: collections.deque   = collections.deque(maxlen=DYNAMIC_BUF_LEN)
        self._dynamic_step_counter: int        = 0

        self._load_models()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        models_dir = os.path.dirname(self.model_path)

        # Static: prefer dedicated files, fall back to combined gesture_model.pkl
        static_path = os.path.join(models_dir, "static_model.pkl")
        if os.path.exists(static_path):
            try:
                self._static_model  = joblib.load(static_path)
                le_path = os.path.join(models_dir, "static_label_encoder.pkl")
                sc_path = os.path.join(models_dir, "static_scaler.pkl")
                self._static_le     = joblib.load(le_path) if os.path.exists(le_path) else None
                self._static_scaler = joblib.load(sc_path) if os.path.exists(sc_path) else None
                print("[GestureClassifier] 정적 모델 로드 완료")
            except Exception as e:
                print(f"[GestureClassifier] 정적 모델 로드 실패: {e}")
                self._static_model = None
        elif os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self._static_model = data["model"]
                self._static_le    = data["label_encoder"]
                print("[GestureClassifier] gesture_model.pkl 로드 (호환 모드)")
            except Exception as e:
                print(f"[GestureClassifier] gesture_model.pkl 로드 실패: {e}")

        # Dynamic: LSTM + scaler
        dynamic_path = os.path.join(models_dir, "dynamic_model.h5")
        if os.path.exists(dynamic_path):
            try:
                import tensorflow as tf
                self._dynamic_model = tf.keras.models.load_model(
                    dynamic_path, compile=False
                )
                le_path  = os.path.join(models_dir, "dynamic_label_encoder.pkl")
                sc_path  = os.path.join(models_dir, "dynamic_scaler.pkl")
                if os.path.exists(le_path):
                    self._dynamic_le = joblib.load(le_path)
                if os.path.exists(sc_path):
                    sc = joblib.load(sc_path)
                    self._dynamic_mu    = sc.get("mean")
                    self._dynamic_sigma = sc.get("std")
                print("[GestureClassifier] 동적 모델 로드 완료")
            except Exception as e:
                print(f"[GestureClassifier] 동적 모델 로드 실패: {e}")

        if self._static_model is None and self._dynamic_model is None:
            print("[GestureClassifier] 학습 모델 없음 — 더미 모델로 초기화")
            self._create_dummy_model()

    def _create_dummy_model(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        rng = np.random.default_rng(42)
        FEAT = 308  # mean+std of 154-D per-frame vector
        X, y = [], []
        for i, word in enumerate(_DUMMY_WORDS):
            for _ in range(30):
                base         = np.zeros(FEAT, dtype=np.float32)
                base[i * 5 % FEAT]         = 0.5
                base[(i * 7 + 3) % FEAT]   = 0.3
                X.append(base + rng.normal(0, 0.05, FEAT).astype(np.float32))
                y.append(word)

        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(np.array(X), y_enc)
        self._static_model = model
        self._static_le    = le
        self._static_scaler = None
        print(f"[GestureClassifier] 더미 모델 생성 ({len(_DUMMY_WORDS)}개 단어)")

    @property
    def is_loaded(self) -> bool:
        return self._static_model is not None or self._dynamic_model is not None

    # ── main entry point ──────────────────────────────────────────────────────

    def classify(self, frame_data: Optional[dict]) -> dict:
        """Classify one frame.  frame_data is the dict from MediaPipeEngine or None."""
        now = time.time()
        result = {
            "word": None, "confidence": 0.0, "sign_type": None,
            "non_manual": {"is_question": False, "is_negative": False,
                           "emotion": "neutral"},
            "is_confirmed": False, "word_boundary": False, "sentence_end": False,
        }

        # ── no hand ───────────────────────────────────────────────────────────
        if frame_data is None:
            elapsed = now - self._last_hand_time
            if elapsed >= NO_HAND_TIME_SEC:
                result["sentence_end"] = True
            # Reset wrist tracking and short-term buffers on prolonged absence
            if elapsed > 1.0:
                self._prev_wrist      = None
                self._candidate_word  = None
                self._confirm_count   = 0
                self._static_buf.clear()
            self._still_start_time = None
            return result

        self._last_hand_time = now

        # ── 내장 제스처 우선 확인 (MediaPipe pre-trained, 데이터 수집 불필요) ──
        bg      = frame_data.get("builtin_gesture", {}) if frame_data else {}
        bg_name = bg.get("name")
        bg_conf = float(bg.get("score", 0.0))

        # 신뢰도 충족 시 hold 버퍼 갱신
        if bg_name and bg_conf >= BUILTIN_CONF_THR and bg_name in BUILTIN_GESTURE_MAP:
            self._builtin_hold_word = BUILTIN_GESTURE_MAP[bg_name]
            self._builtin_hold_time = now

        # hold 시간 내에는 마지막 제스처 유지 (각도 변화로 인한 끊김 방지)
        hold_active = (self._builtin_hold_word is not None
                       and now - self._builtin_hold_time <= BUILTIN_HOLD_SEC)

        if hold_active:
            word      = self._builtin_hold_word
            use_conf  = bg_conf if bg_name and bg_conf >= BUILTIN_CONF_THR else 0.85
            if word == self._candidate_word:
                self._confirm_count += 1
            else:
                self._candidate_word = word
                self._confirm_count  = 1
            is_confirmed = self._confirm_count >= CONFIRM_FRAMES
            non_manual   = self._nm_classifier.classify(frame_data)
            result.update({
                "word":         word,
                "confidence":   use_conf,
                "sign_type":    "builtin",
                "non_manual":   non_manual,
                "is_confirmed": is_confirmed,
            })
            self._prev_conf = use_conf
            return result

        # hold 만료 시 초기화
        self._builtin_hold_word = None
        self._candidate_word    = None
        self._confirm_count     = 0

        # ── wrist velocity ────────────────────────────────────────────────────
        rh    = frame_data.get("manual", {}).get("right_hand")
        wrist = (np.array(rh[0], dtype=np.float32)
                 if rh else np.zeros(3, dtype=np.float32))

        if self._prev_wrist is not None:
            vel = wrist - self._prev_wrist
        else:
            vel = np.zeros(3, dtype=np.float32)
        self._prev_wrist = wrist.copy()

        # Write velocity back into frame_data for dynamic feature extraction
        frame_data["movement"] = {"velocity": vel.tolist()}

        vel_mag = float(np.linalg.norm(vel))
        self._vel_history.append(vel_mag)
        avg_vel = float(np.mean(self._vel_history)) if self._vel_history else 0.0

        is_dynamic = avg_vel > STATIC_VEL_THR

        # ── stillness → word-boundary ─────────────────────────────────────────
        if avg_vel < STATIC_VEL_THR * 0.5:
            if self._still_start_time is None:
                self._still_start_time = now
            elif now - self._still_start_time >= STILL_TIME_SEC:
                result["word_boundary"] = True
        else:
            self._still_start_time = None

        # ── model inference ───────────────────────────────────────────────────
        word, conf, sign_type = self._run_prediction(frame_data, is_dynamic)

        # ── 임계값 미만은 표시만, 확정·문장 추가는 하지 않음 ─────────────────────
        above_thr = (word is not None and conf >= self.confidence_threshold)

        # ── anti-spike ────────────────────────────────────────────────────────
        if above_thr and self._prev_conf > 0.0:
            if conf < self._prev_conf - SPIKE_DROP_THR:
                above_thr = False
        self._prev_conf = conf if above_thr else 0.0

        # ── 확정 카운터 (임계값 이상인 예측만 카운트) ─────────────────────────
        if above_thr and word == self._candidate_word:
            self._confirm_count += 1
        else:
            self._candidate_word = word if above_thr else None
            self._confirm_count  = 1 if above_thr else 0

        is_confirmed = (self._confirm_count >= CONFIRM_FRAMES and above_thr)

        # ── 비수지 분류 (학습 모델 or 규칙 기반 폴백) ────────────────────────
        non_manual = self._nm_classifier.classify(frame_data)

        result.update({
            "word":         word,
            "confidence":   conf,
            "sign_type":    sign_type,
            "non_manual":   non_manual,
            "is_confirmed": is_confirmed,
        })
        return result

    # ── prediction dispatch ───────────────────────────────────────────────────

    def _run_prediction(
        self, frame_data: dict, is_dynamic: bool
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """Choose static or dynamic model, run it, return (word, conf, type)."""

        if is_dynamic and self._dynamic_model is not None:
            self._dynamic_buf.append(frame_data)
            self._static_buf.clear()

            self._dynamic_step_counter += 1
            if self._dynamic_step_counter < DYNAMIC_STEP:
                return None, 0.0, "dynamic"
            self._dynamic_step_counter = 0

            if len(self._dynamic_buf) < 20:
                return None, 0.0, "dynamic"

            word, conf = self._predict_dynamic(list(self._dynamic_buf))
            return word, conf, "dynamic"

        # Static path
        if self._static_model is not None:
            self._static_buf.append(frame_data)
            self._dynamic_buf.clear()
            word, conf = self._predict_static(list(self._static_buf))
            return word, conf, "static"

        return None, 0.0, None

    # ── static inference (RandomForest) ──────────────────────────────────────

    def _predict_static(
        self, frames: List[dict]
    ) -> Tuple[Optional[str], float]:
        if not frames:
            return None, 0.0

        vecs = np.array(
            [self._extract_frame_vector(fr) for fr in frames],
            dtype=np.float32,
        )  # (n, 154)

        # Aggregate: mean ‖ std → 308-D (matches training pipeline)
        feat = np.concatenate(
            [vecs.mean(axis=0), vecs.std(axis=0)]
        ).reshape(1, -1)

        if self._static_scaler is not None:
            feat = self._static_scaler.transform(feat)

        try:
            proba     = self._static_model.predict_proba(feat)[0]
            best_idx  = int(np.argmax(proba))
            best_conf = float(proba[best_idx])
            if self._static_le is not None:
                word = self._static_le.inverse_transform([best_idx])[0]
                # 임계값 이상이면 정식 예측, 미만이면 낮은 신뢰도로 반환
                return word, best_conf
        except Exception as e:
            print(f"[GestureClassifier] 정적 예측 오류: {e}")
        return None, 0.0

    # ── dynamic inference (LSTM) ──────────────────────────────────────────────

    def _predict_dynamic(
        self, frames: List[dict]
    ) -> Tuple[Optional[str], float]:
        if not frames:
            return None, 0.0

        seq: List[np.ndarray] = []
        prev_rn: Optional[np.ndarray] = None

        for fr in frames:
            vel    = fr.get("movement", {}).get("velocity", [0.0, 0.0, 0.0])
            rn_raw = fr.get("manual", {}).get("right_hand_norm")

            if rn_raw is not None and prev_rn is not None:
                rn_flat  = np.array(rn_raw, dtype=np.float32).flatten()
                hand_chg = float(np.linalg.norm(rn_flat - prev_rn))
            else:
                hand_chg = 0.0

            if rn_raw is not None:
                prev_rn = np.array(rn_raw, dtype=np.float32).flatten()

            vec = self._extract_frame_vector(fr)                      # 154D
            seq.append(np.concatenate([
                vec,
                np.array(vel, dtype=np.float32),                      # 3D
                np.array([hand_chg], dtype=np.float32),               # 1D
            ]))                                                        # 158D

        seq_arr = np.array(seq, dtype=np.float32)  # (n, 158)
        n, d    = seq_arr.shape

        # Adjust to SEQ_LEN=60 via uniform sampling or zero-padding
        if n > SEQ_LEN:
            idx     = np.round(np.linspace(0, n - 1, SEQ_LEN)).astype(int)
            seq_arr = seq_arr[idx]
        elif n < SEQ_LEN:
            pad     = np.zeros((SEQ_LEN - n, d), dtype=np.float32)
            seq_arr = np.concatenate([seq_arr, pad], axis=0)

        # Per-feature normalisation (from training)
        if self._dynamic_mu is not None and self._dynamic_sigma is not None:
            seq_arr = (seq_arr - self._dynamic_mu) / (self._dynamic_sigma + 1e-8)

        try:
            proba    = self._dynamic_model.predict(seq_arr[np.newaxis], verbose=0)[0]
            best_idx = int(np.argmax(proba))
            best_conf = float(proba[best_idx])
            if best_conf >= self.confidence_threshold and self._dynamic_le is not None:
                word = self._dynamic_le.inverse_transform([best_idx])[0]
                return word, best_conf
        except Exception as e:
            print(f"[GestureClassifier] 동적 예측 오류: {e}")
        return None, 0.0

    # ── 154-D feature extraction (mirrors train_model.py exactly) ────────────

    def _extract_frame_vector(self, frame_data: dict) -> np.ndarray:
        """Extract the 154-D per-frame feature vector used by both models."""
        m = frame_data.get("manual", {})

        # ── right hand (63D) ──────────────────────────────────────────────────
        rh_norm = m.get("right_hand_norm")
        rh_feat = self._scale_hand(rh_norm)

        # ── left hand (63D) ───────────────────────────────────────────────────
        lh_norm = m.get("left_hand_norm")
        lh_feat = self._scale_hand(lh_norm)

        # ── palm direction (3D) + orientation (1D) ───────────────────────────
        pd  = np.array(m.get("palm_direction", [0.0, 0.0, 0.0]), dtype=np.float32)
        ori = np.array([m.get("hand_orientation", 0.0)], dtype=np.float32)
        manual = np.concatenate([rh_feat, lh_feat, pd, ori])          # 130D

        # ── non-manual (18D) ──────────────────────────────────────────────────
        nm   = frame_data.get("non_manual", {})
        fl   = nm.get("face_landmarks")
        eb_l = nm.get("eyebrow_left")
        eb_r = nm.get("eyebrow_right")
        lips = nm.get("lip_shape")
        hp   = nm.get("head_pose", {})
        bl   = nm.get("body_lean", {})

        if (fl and eb_l and eb_r
                and len(fl) > max(EYE_L_IDX, EYE_R_IDX)):
            left_raise  = float(fl[EYE_L_IDX][1]
                                - np.mean([p[1] for p in eb_l]))
            right_raise = float(fl[EYE_R_IDX][1]
                                - np.mean([p[1] for p in eb_r]))
        else:
            left_raise = right_raise = 0.0

        if lips and len(lips) >= 20:
            mouth_open = float(lips[14][1] - lips[5][1])
            lps_arr    = np.array(lips, dtype=np.float32)
            key_pts    = lps_arr[LIP_KEY_IDX, :2].flatten()           # 10D
        else:
            mouth_open = 0.0
            key_pts    = np.zeros(10, dtype=np.float32)

        non_manual = np.array([
            left_raise, right_raise, mouth_open,
            *key_pts,
            hp.get("pitch", 0.0), hp.get("yaw", 0.0), hp.get("roll", 0.0),
            bl.get("shoulder_angle", 0.0), bl.get("torso_direction", 0.0),
        ], dtype=np.float32)                                           # 18D

        # ── signing space (6D) ────────────────────────────────────────────────
        ss  = frame_data.get("signing_space", {})
        rf  = ss.get("hand_relative_to_face",     [0.0, 0.0, 0.0])
        rs  = ss.get("hand_relative_to_shoulder", [0.0, 0.0, 0.0])
        spatial = np.array(rf + rs, dtype=np.float32)                  # 6D

        return np.concatenate([manual, non_manual, spatial])           # 154D

    @staticmethod
    def _scale_hand(hand_norm: Optional[list]) -> np.ndarray:
        """Wrist-relative coords → middle-finger-length normalised, 63-D."""
        if hand_norm is None:
            return np.zeros(63, dtype=np.float32)
        arr = np.array(hand_norm, dtype=np.float32)
        if arr.shape != (21, 3):
            return np.zeros(63, dtype=np.float32)
        scale = float(np.linalg.norm(arr[12] - arr[9]))  # MID_TIP - MID_MCP
        if scale > 1e-6:
            arr = arr / scale
        return arr.flatten()

    # ── misc ──────────────────────────────────────────────────────────────────

    @property
    def classes(self) -> List[str]:
        if self._static_le is not None:
            return list(self._static_le.classes_)
        if self._dynamic_le is not None:
            return list(self._dynamic_le.classes_)
        return []
