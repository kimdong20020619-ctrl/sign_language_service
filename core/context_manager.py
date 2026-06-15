from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class Message:
    role: str          # "deaf" | "hearing"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    raw_words: Optional[str] = None


class ContextManager:
    """대화 컨텍스트 및 세션 상태를 관리한다."""

    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self.messages: List[Message] = []
        self.current_mode: str = "카페"
        self.session_start: datetime = datetime.now()

    # ------------------------------------------------------------------
    def add_deaf_message(self, content: str, raw_words: str = "") -> None:
        self._push(Message(role="deaf", content=content, raw_words=raw_words))

    def add_hearing_message(self, content: str) -> None:
        self._push(Message(role="hearing", content=content))

    def _push(self, msg: Message) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.max_history:
            self.messages.pop(0)

    # ------------------------------------------------------------------
    def get_claude_context(self, n: int = 6) -> List[Dict]:
        """Claude API messages 형식으로 최근 n개 대화 반환."""
        recent = self.messages[-n:]
        result = []
        for msg in recent:
            # Claude API는 user/assistant 교대 구조를 선호하므로
            # 청각장애인 발화 = user, 비장애인 답변 = assistant 로 매핑
            result.append({
                "role": "user" if msg.role == "deaf" else "assistant",
                "content": msg.content,
            })
        return result

    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self.current_mode = mode

    def clear(self) -> None:
        self.messages.clear()
        self.session_start = datetime.now()

    @property
    def message_count(self) -> int:
        return len(self.messages)
