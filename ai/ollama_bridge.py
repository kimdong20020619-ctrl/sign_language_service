"""
ai/ollama_bridge.py  —  Ollama + EXAONE 3.5 연동 브리지

[설치]
  1. https://ollama.com 에서 Ollama 설치
  2. CMD: ollama pull exaone3.5
  3. pip install ollama
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import requests
from PySide6.QtCore import QThread, Signal

try:
    import ollama as _ollama_lib
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

# ── 폴백 모델 순서 ────────────────────────────────────────────────────
_FALLBACK_MODELS = ["exaone3.5", "llama3.2"]

# ── 환경 모드별 추가 시스템 프롬프트 ──────────────────────────────────
_MODE_PROMPTS: dict[str, str] = {
    "카페":  "지금은 카페/식당 주문 상황이야. 메뉴 주문, 결제, 포장 등의 문맥으로 해석해줘.",
    "병원":  "지금은 병원 진료 상황이야. 증상 설명, 진료 요청 문맥으로 해석해줘.",
    "은행":  "지금은 은행/관공서 업무 상황이야. 금융 업무, 서류 처리 문맥으로 해석해줘.",
    "학교":  "지금은 학교 수업/상담 상황이야.",
    "마트":  "지금은 마트/편의점 쇼핑 상황이야.",
    "직장":  "지금은 직장 업무/회의 상황이야.",
}

# ── 규칙 기반 폴백: 비수지 신호 → 문장 변형 패턴 ─────────────────────
_QUESTION_SUFFIX  = ["이에요?", "인가요?", "하나요?", "할까요?"]
_NEGATIVE_PREFIX  = ["아니요, ", ""]
_NEGATIVE_SUFFIX  = [" 아닙니다", " 없습니다", " 안 해요"]
_EMPHASIS_SUFFIX  = ["!", "요!"]

_BASE_SYSTEM_PROMPT = (
    "너는 청각장애인의 수화 단어를 자연스러운 한국어 문장으로 바꿔주는 통역 AI야. "
    "수화는 어순이 다르고 조사가 생략되므로 문맥에 맞게 자연스럽게 완성해줘. "
    "반드시 짧고 명확한 문장 하나만 출력해. 설명이나 부연 없이 문장만 출력해."
)


# ══════════════════════════════════════════════════════════════════════
# 규칙 기반 문장 조합 (LLM 전부 실패 시 최후 폴백)
# ══════════════════════════════════════════════════════════════════════

def _rule_based(words: list[str], non_manual: dict) -> str:
    """단어 리스트 + 비수지 신호 → 간단한 규칙 기반 문장."""
    if not words:
        return ""

    base = " ".join(words)

    if non_manual.get("is_negative"):
        base = base + _NEGATIVE_SUFFIX[0]
    elif non_manual.get("is_question"):
        # 마지막 단어에 의문형 어미 시도
        base = base.rstrip("요") + "인가요?"
    elif non_manual.get("is_emphasis"):
        base = base + "!"
    else:
        if not base.endswith(("요", "다", "까", "요?", "다.")):
            base = base + "요."

    return base


# ══════════════════════════════════════════════════════════════════════
# 스트리밍 워커 (QThread)
# ══════════════════════════════════════════════════════════════════════

class _StreamWorker(QThread):
    """
    별도 스레드에서 Ollama 스트리밍 호출을 실행한다.
    UI 블로킹 없이 토큰을 실시간으로 ChatPanel에 전달한다.
    """

    text_chunk:        Signal = Signal(str)   # 토큰 1개씩 전송
    sentence_complete: Signal = Signal(str)   # 완성된 전체 문장
    error_occurred:    Signal = Signal(str)   # 오류 메시지

    def __init__(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        base_url: str,
        timeout: int,
        fallback_fn: Callable[[], str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model        = model
        self._system       = system_prompt
        self._user         = user_prompt
        self._base_url     = base_url
        self._timeout      = timeout
        self._fallback_fn  = fallback_fn

    def run(self) -> None:
        if not _OLLAMA_AVAILABLE:
            self._emit_fallback("ollama 패키지 미설치")
            return

        client   = _ollama_lib.Client(host=self._base_url)
        full     = ""
        models   = _FALLBACK_MODELS if self._model == _FALLBACK_MODELS[0] \
                   else ([self._model] + _FALLBACK_MODELS)

        for attempt_model in dict.fromkeys(models):  # 중복 제거 후 순서 유지
            try:
                stream = client.chat(
                    model=attempt_model,
                    messages=[
                        {"role": "system", "content": self._system},
                        {"role": "user",   "content": self._user},
                    ],
                    stream=True,
                    options={"num_predict": 256},
                )
                for chunk in stream:
                    token = chunk["message"]["content"]
                    full += token
                    self.text_chunk.emit(token)

                full = full.strip()
                if full:
                    self.sentence_complete.emit(full)
                    return

            except Exception as e:
                err_str = str(e)
                if "model" in err_str.lower() and "not found" in err_str.lower():
                    # 모델 미설치 → 다음 폴백 시도
                    continue
                # 네트워크·서버 오류
                self._emit_fallback(f"Ollama 오류: {e}")
                return

        # 모든 모델 실패 → 규칙 기반 / 단어 이어붙임
        self._emit_fallback("사용 가능한 모델 없음")

    def _emit_fallback(self, reason: str) -> None:
        self.error_occurred.emit(reason)
        result = self._fallback_fn()
        if result:
            self.sentence_complete.emit(result)


# ══════════════════════════════════════════════════════════════════════
# OllamaBridge
# ══════════════════════════════════════════════════════════════════════

class OllamaBridge:
    """
    수화 단어 리스트 → 자연스러운 한국어 문장 (Ollama EXAONE 3.5).

    [시그널 사용 예]
        bridge = OllamaBridge(config)
        worker = bridge.correct_sentence_stream(
            words=["커피", "아메리카노", "주세요"],
            mode="카페",
            non_manual={"is_question": False},
        )
        worker.text_chunk.connect(chat_panel.append_token)
        worker.sentence_complete.connect(chat_panel.finalize_bubble)
        worker.error_occurred.connect(lambda e: print(e))
        worker.start()
    """

    def __init__(self, config: dict) -> None:
        oc = config.get("ollama", {})
        self._model    = oc.get("model",    "exaone3.5")
        self._base_url = oc.get("base_url", "http://localhost:11434")
        self._timeout  = oc.get("timeout",  30)
        self._mode     = config.get("ui", {}).get("default_mode", "카페")

    # ── 모드 변경 ─────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    # ── 서버 상태 확인 ────────────────────────────────────────────────

    def is_server_running(self) -> bool:
        """Ollama 서버 실행 여부 확인 (앱 시작 시 호출)."""
        try:
            r = requests.get(
                f"{self._base_url}/api/tags", timeout=3
            )
            return r.status_code == 200
        except Exception:
            return False

    def list_local_models(self) -> list[str]:
        """로컬에 설치된 모델 이름 목록 반환."""
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ── 프롬프트 조합 ─────────────────────────────────────────────────

    def _build_system(self, mode: str) -> str:
        extra = _MODE_PROMPTS.get(mode, "")
        return f"{_BASE_SYSTEM_PROMPT}\n{extra}".strip()

    def _build_user_prompt(
        self, words: list[str], non_manual: dict
    ) -> str:
        joined = " ".join(words)
        hints: list[str] = []

        if non_manual.get("is_question"):
            hints.append("문장을 의문문으로 만들어줘")
        if non_manual.get("is_negative"):
            hints.append("부정 의미를 반영해줘")
        if non_manual.get("is_emphasis"):
            hints.append("강조 어조로 만들어줘")

        if hints:
            return f"수화 단어: {joined}\n조건: {', '.join(hints)}"
        return f"수화 단어: {joined}"

    # ── 동기 교정 (테스트·폴백 재사용용) ──────────────────────────────

    def correct_sentence(
        self,
        words: list[str],
        mode: Optional[str] = None,
        non_manual: Optional[dict] = None,
    ) -> str:
        """블로킹 호출. UI 스레드에서 직접 쓰지 말 것."""
        if not words:
            return ""
        nm      = non_manual or {}
        mode    = mode or self._mode
        system  = self._build_system(mode)
        user    = self._build_user_prompt(words, nm)

        if not _OLLAMA_AVAILABLE:
            return _rule_based(words, nm)

        client = _ollama_lib.Client(host=self._base_url)
        for attempt_model in dict.fromkeys(_FALLBACK_MODELS):
            try:
                resp = client.chat(
                    model=attempt_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    options={"num_predict": 256},
                )
                result = resp["message"]["content"].strip()
                if result:
                    return _clean_output(result)
            except Exception:
                continue

        return _rule_based(words, nm)

    # ── 비동기 스트리밍 교정 ──────────────────────────────────────────

    def correct_sentence_stream(
        self,
        words: list[str],
        mode: Optional[str] = None,
        non_manual: Optional[dict] = None,
    ) -> _StreamWorker:
        """
        _StreamWorker(QThread)를 반환한다.
        호출자가 시그널을 연결한 뒤 .start() 해야 실행된다.
        """
        nm     = non_manual or {}
        mode   = mode or self._mode
        system = self._build_system(mode)
        user   = self._build_user_prompt(words, nm)

        def _fallback() -> str:
            rb = _rule_based(words, nm)
            return rb if rb else " ".join(words)

        worker = _StreamWorker(
            model       = self._model,
            system_prompt = system,
            user_prompt = user,
            base_url    = self._base_url,
            timeout     = self._timeout,
            fallback_fn = _fallback,
        )
        return worker

    # ── 연결 테스트 ───────────────────────────────────────────────────

    def test_connection(self) -> tuple[bool, str]:
        """
        반환: (성공 여부, 상태 메시지)
        앱 시작 시 MainWindow.set_llm_status() 에 전달한다.
        """
        if not self.is_server_running():
            return False, "Ollama 서버 미실행 — CMD: ollama serve"

        models = self.list_local_models()
        has_exaone = any("exaone" in m.lower() for m in models)
        has_llama  = any("llama"  in m.lower() for m in models)

        if has_exaone:
            return True, f"EXAONE 3.5 연결됨"
        if has_llama:
            return True, f"llama3.2 폴백 사용 중 (exaone3.5 미설치)"
        return False, (
            "사용 가능한 모델 없음 — CMD: ollama pull exaone3.5"
        )


# ══════════════════════════════════════════════════════════════════════
# 출력 정제 (LLM이 부연 설명을 붙일 경우 제거)
# ══════════════════════════════════════════════════════════════════════

def _clean_output(text: str) -> str:
    """LLM 응답에서 첫 번째 완결 문장만 추출한다."""
    # 줄바꿈 이후 설명 제거
    text = text.split("\n")[0].strip()
    # "문장:" / "결과:" 같은 레이블 제거
    text = re.sub(r"^(문장|결과|출력|답변)\s*[:：]\s*", "", text)
    # 따옴표 제거
    text = text.strip("\"'""''")
    return text.strip()
