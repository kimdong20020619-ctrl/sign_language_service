from typing import Optional, Callable, Dict
import anthropic
from ai.prompt_templates import (
    get_system_prompt,
    get_correction_prompt,
    build_few_shot_messages,
)


class ClaudeBridge:
    """anthropic SDK로 Claude API에 연결해 수화 단어를 문장으로 교정한다.

    correct_sentence / correct_sentence_stream 모두 non_manual 딕셔너리를
    받아 비수지 신호(눈썹 올림 → 의문문, 고개 젓기 → 부정 등)를 프롬프트에 반영한다.
    """

    def __init__(self, api_key: str,
                 model: str = "claude-sonnet-4-6",
                 max_tokens: int = 1024):
        self.client       = anthropic.Anthropic(api_key=api_key)
        self.model        = model
        self.max_tokens   = max_tokens
        self.current_mode = "카페"

    # ── mode ──────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        self.current_mode = mode

    # ── sync correction ───────────────────────────────────────────────────────

    def correct_sentence(
        self,
        words: str,
        non_manual: Optional[Dict] = None,
    ) -> Optional[str]:
        """단어 나열 + 비수지 정보 → 자연스러운 문장 (동기, 비스트리밍)."""
        if not words.strip():
            return None
        try:
            messages = build_few_shot_messages(self.current_mode) + [{
                "role":    "user",
                "content": get_correction_prompt(words, self.current_mode, non_manual),
            }]
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=get_system_prompt(self.current_mode),
                messages=messages,
            )
            return response.content[0].text.strip()
        except anthropic.AuthenticationError:
            print("[ClaudeBridge] API 키 인증 실패")
            return None
        except Exception as e:
            print(f"[ClaudeBridge] API 오류: {e}")
            return None

    # ── streaming correction ──────────────────────────────────────────────────

    def correct_sentence_stream(
        self,
        words: str,
        on_token: Callable[[str], None],
        non_manual: Optional[Dict] = None,
    ) -> Optional[str]:
        """스트리밍으로 문장 교정. 토큰마다 on_token 콜백 호출."""
        if not words.strip():
            return None
        try:
            messages = build_few_shot_messages(self.current_mode) + [{
                "role":    "user",
                "content": get_correction_prompt(words, self.current_mode, non_manual),
            }]
            full = ""
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=get_system_prompt(self.current_mode),
                messages=messages,
            ) as stream:
                for token in stream.text_stream:
                    full += token
                    on_token(token)
            return full.strip()
        except Exception as e:
            print(f"[ClaudeBridge] 스트리밍 오류: {e}")
            return None

    # ── connection test ───────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """API 키 및 네트워크 연결 확인."""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            print(f"[ClaudeBridge] 연결 테스트 실패: {e}")
            return False
