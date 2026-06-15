from typing import Dict, List, Optional

# ── 비수지 신호 공통 지침 ─────────────────────────────────────────────────────
_NON_MANUAL_RULES = """\
[비수지 신호 처리 규칙]
수화는 손 동작(수지 신호) 외에 얼굴 표정·몸짓(비수지 신호)도 의미를 구성한다.
아래 규칙을 반드시 반영하여 문장을 생성하라.

- is_question=True  → 문장 끝에 "?"를 붙이고 의문문 어미(-요?, -나요?, -까?)로 변환
- is_negative=True  → 부정 의미를 문장에 반영 ("아니요", "안", "못", "없어요" 등 맥락에 맞게 선택)
- emotion=happy     → 밝고 긍정적인 어조 반영 (예: "정말", "너무" 등)
- emotion=sad       → 불편함·고통·우울한 어조 반영
- emotion=uncomfortable → 불편함·불만 어조 반영
- head_nod=True     → 긍정·확인 의미 추가 ("네", "맞아요" 등)
- emotion=neutral   → 특별한 어조 추가 없음\
"""

# ── 환경별 시스템 프롬프트 ─────────────────────────────────────────────────────
SYSTEM_PROMPTS: Dict[str, str] = {
    "카페": (
        "너는 청각장애인의 수화를 텍스트로 변환해주는 전문 통역 AI야.\n"
        "현재 상황: 카페 주문·결제 환경.\n"
        "음료·음식 주문, 포장 여부, 사이즈, Wi-Fi·자리 문의, 영수증 등의 맥락을 우선 고려해.\n"
        "짧고 자연스러운 한국어 주문/문의 문장 하나만 출력해. 설명 없이.\n\n"
        + _NON_MANUAL_RULES
    ),
    "병원": (
        "너는 청각장애인의 수화를 텍스트로 변환해주는 전문 통역 AI야.\n"
        "현재 상황: 병원·의원 진료 환경.\n"
        "증상 설명, 진료 예약, 검사 결과 문의, 약 처방 등의 맥락을 우선 고려해.\n"
        "증상은 구체적으로, 의료 용어는 이해하기 쉽게 변환해. 문장 하나만 출력해. 설명 없이.\n\n"
        + _NON_MANUAL_RULES
    ),
    "은행": (
        "너는 청각장애인의 수화를 텍스트로 변환해주는 전문 통역 AI야.\n"
        "현재 상황: 은행·금융 업무 환경.\n"
        "계좌 개설·조회, 송금, 카드 발급, 대출 상담 등의 맥락을 우선 고려해.\n"
        "정확하고 공식적인 금융 문장 하나만 출력해. 설명 없이.\n\n"
        + _NON_MANUAL_RULES
    ),
    "학교": (
        "너는 청각장애인의 수화를 텍스트로 변환해주는 전문 통역 AI야.\n"
        "현재 상황: 학교·교육기관 수업·상담 환경.\n"
        "수업 질문, 과제·성적 문의, 교내 행사, 진학 상담 등의 맥락을 우선 고려해.\n"
        "친근하고 명확한 교육적 문장 하나만 출력해. 설명 없이.\n\n"
        + _NON_MANUAL_RULES
    ),
    "마트": (
        "너는 청각장애인의 수화를 텍스트로 변환해주는 전문 통역 AI야.\n"
        "현재 상황: 마트·쇼핑몰 쇼핑 환경.\n"
        "상품 위치·가격 문의, 환불·교환, 포인트 적립 등의 맥락을 우선 고려해.\n"
        "간결하고 실용적인 쇼핑 문장 하나만 출력해. 설명 없이.\n\n"
        + _NON_MANUAL_RULES
    ),
    "직장": (
        "너는 청각장애인의 수화를 텍스트로 변환해주는 전문 통역 AI야.\n"
        "현재 상황: 직장 업무·회의 환경.\n"
        "업무 보고, 회의 발언, 요청·질문, 일정 조율 등의 맥락을 우선 고려해.\n"
        "공손하고 전문적인 비즈니스 문장 하나만 출력해. 설명 없이.\n\n"
        + _NON_MANUAL_RULES
    ),
}

_DEFAULT_MODE = "카페"

# ── 교정 프롬프트 템플릿 ──────────────────────────────────────────────────────
_CORRECTION_TEMPLATE = """\
수화 인식 시스템이 다음 정보를 감지했습니다.

인식된 단어: {words}
비수지 정보: {non_manual}
현재 환경: {environment}

[변환 규칙]
1. 단어들을 [{environment}] 환경에 맞는 자연스러운 한국어 문장 하나로 변환하세요.
2. 원래 의미를 최대한 보존하세요.
3. 비수지 정보를 반드시 반영하세요:
   - is_question=true  → 의문문 어미와 "?" 추가
   - is_negative=true  → 부정 표현("안", "못", "아니요", "없어요" 등) 반영
   - emotion=happy     → 밝고 긍정적인 어조
   - emotion=sad       → 불편함·고통 어조
   - emotion=uncomfortable → 불편함·불만 어조
   - head_nod=true     → 긍정·확인 의미("네", "맞아요") 추가
4. 설명 없이 변환된 문장만 출력하세요.\
"""

# ── 예시 few-shot (Claude가 출력 형식을 빠르게 파악하도록) ────────────────────
_FEW_SHOT_EXAMPLES: Dict[str, List[Dict]] = {
    "카페": [
        {
            "words": ["아이스", "아메리카노", "하나", "주세요"],
            "non_manual": {"is_question": False, "is_negative": False, "emotion": "neutral"},
            "output": "아이스 아메리카노 한 잔 주세요.",
        },
        {
            "words": ["화장실", "어디"],
            "non_manual": {"is_question": True, "is_negative": False, "emotion": "neutral"},
            "output": "화장실이 어디 있나요?",
        },
        {
            "words": ["설탕", "없어요", "주세요"],
            "non_manual": {"is_question": False, "is_negative": True, "emotion": "neutral"},
            "output": "설탕 없이 주세요.",
        },
    ],
    "병원": [
        {
            "words": ["머리", "아파요", "어제부터"],
            "non_manual": {"is_question": False, "is_negative": False, "emotion": "sad"},
            "output": "어제부터 머리가 많이 아픕니다.",
        },
        {
            "words": ["약", "처방", "받을 수 있어요"],
            "non_manual": {"is_question": True, "is_negative": False, "emotion": "neutral"},
            "output": "약을 처방받을 수 있나요?",
        },
    ],
    "은행": [
        {
            "words": ["통장", "만들고", "싶어요"],
            "non_manual": {"is_question": False, "is_negative": False, "emotion": "neutral"},
            "output": "통장을 새로 만들고 싶습니다.",
        },
    ],
    "학교": [
        {
            "words": ["과제", "제출", "언제"],
            "non_manual": {"is_question": True, "is_negative": False, "emotion": "neutral"},
            "output": "과제 제출 마감이 언제인가요?",
        },
    ],
    "마트": [
        {
            "words": ["사과", "어디", "있어요"],
            "non_manual": {"is_question": True, "is_negative": False, "emotion": "neutral"},
            "output": "사과는 어디에 있나요?",
        },
    ],
    "직장": [
        {
            "words": ["보고서", "내일", "제출", "어려워요"],
            "non_manual": {"is_question": False, "is_negative": False, "emotion": "uncomfortable"},
            "output": "보고서를 내일까지 제출하기가 좀 어렵습니다.",
        },
    ],
}


# ── public API ────────────────────────────────────────────────────────────────

def get_system_prompt(mode: str) -> str:
    """Return the system prompt for the given environment mode."""
    return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS[_DEFAULT_MODE])


def get_correction_prompt(
    words: str,
    environment: str,
    non_manual: Optional[Dict] = None,
) -> str:
    """Build a user-turn correction prompt.

    Parameters
    ----------
    words       : space-joined recognised word string, e.g. "아이스 아메리카노 하나 주세요"
    environment : one of the six mode keys
    non_manual  : dict from GestureClassifier, e.g.
                  {"is_question": False, "is_negative": False, "emotion": "neutral"}
    """
    if non_manual is None:
        non_manual = {"is_question": False, "is_negative": False, "emotion": "neutral"}

    nm_str = (
        f"is_question={str(non_manual.get('is_question', False)).lower()}, "
        f"is_negative={str(non_manual.get('is_negative', False)).lower()}, "
        f"emotion={non_manual.get('emotion', 'neutral')}"
        + (", head_nod=true" if non_manual.get("head_nod") else "")
    )

    return _CORRECTION_TEMPLATE.format(
        words=words,
        non_manual=nm_str,
        environment=environment,
    )


def get_few_shot_examples(mode: str) -> List[Dict]:
    """Return few-shot example list for the given mode (may be empty)."""
    return _FEW_SHOT_EXAMPLES.get(mode, [])


def build_few_shot_messages(mode: str) -> List[Dict[str, str]]:
    """Build a list of {'role': ..., 'content': ...} messages for few-shot prompting.

    Inject these between the system prompt and the actual user turn so Claude
    immediately learns the expected output format.
    """
    messages = []
    for ex in get_few_shot_examples(mode):
        nm = ex.get("non_manual", {})
        nm_str = (
            f"is_question={str(nm.get('is_question', False)).lower()}, "
            f"is_negative={str(nm.get('is_negative', False)).lower()}, "
            f"emotion={nm.get('emotion', 'neutral')}"
        )
        user_content = _CORRECTION_TEMPLATE.format(
            words=" ".join(ex["words"]),
            non_manual=nm_str,
            environment=mode,
        )
        messages.append({"role": "user",    "content": user_content})
        messages.append({"role": "assistant", "content": ex["output"]})
    return messages
