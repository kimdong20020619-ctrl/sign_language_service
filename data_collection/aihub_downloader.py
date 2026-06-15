#!/usr/bin/env python3
"""
AI Hub 수어 데이터셋(dataSetSn=103) 대용량 파일 다운로더.

사용법:
    python data_collection/aihub_downloader.py --preset minimal   # ~3 GB
    python data_collection/aihub_downloader.py --preset morpheme  # ~230 MB
    python data_collection/aihub_downloader.py --preset standard  # ~3.2 GB
    python data_collection/aihub_downloader.py --list-files       # 파일 목록 조회
    python data_collection/aihub_downloader.py --dry-run --preset minimal

실제 API 엔드포인트 (aihubshell 스크립트 기준):
    파일 목록: GET https://api.aihub.or.kr/info/103.do  (인증 불필요)
    다운로드:  GET https://api.aihub.or.kr/down/0.6/103.do?fileSn={id}
               헤더: apikey: {your_key}   ← 소문자 주의!

config.json 의 aihub.api_key 를 먼저 설정하세요.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:
    print("[오류] tqdm 미설치 → pip install tqdm")
    sys.exit(1)

# ── 프로젝트 루트 탐색 ────────────────────────────────────────────────
_THIS      = Path(__file__).resolve()
_ROOT      = _THIS.parent.parent          # sign_language_service/
CONFIG     = _ROOT / "config.json"
DATA       = _ROOT / "data" / "aihub"
DATA_SENT  = _ROOT / "data" / "aihub_sentences"
DATA_RAW   = _ROOT / "data" / "downloads"   # 압축 파일 임시 저장

# ══════════════════════════════════════════════════════════════════════
# 단어 카테고리 정의 (8개 카테고리, 총 350개)
# ══════════════════════════════════════════════════════════════════════

WORD_CATEGORIES: dict[str, list[str]] = {
    # ── 1. 공통기본 (60개) ─────────────────────────────────────────────
    "공통기본": [
        "안녕하세요", "감사합니다", "죄송합니다", "반갑습니다",
        "안녕히가세요", "안녕히계세요", "잘부탁드립니다",
        "네", "아니오", "맞아요", "괜찮아요", "좋아요", "싫어요",
        "모르겠어요", "알겠어요",
        "주세요", "도와주세요", "잠깐만요", "다시", "천천히",
        "빨리", "크게", "작게", "많이", "조금",
        "하나", "둘", "셋", "넷", "다섯",
        "여섯", "일곱", "여덟", "아홉", "열",
        "오늘", "내일", "어제", "지금", "나중에",
        "아침", "점심", "저녁", "밤", "항상",
        "빨강", "파랑", "초록", "노랑", "흰색", "검정",
        "왼쪽", "오른쪽", "앞", "뒤", "위", "아래",
        "여기", "저기", "어디",
    ],

    # ── 2. 일상동사 (50개) ─────────────────────────────────────────────
    "일상동사": [
        "가다", "오다", "먹다", "마시다", "자다",
        "일어나다", "씻다", "입다", "벗다", "사다",
        "팔다", "주다", "받다", "보다", "듣다",
        "말하다", "읽다", "쓰다", "배우다", "가르치다",
        "일하다", "쉬다", "놀다", "운동하다", "청소하다",
        "요리하다", "운전하다", "타다", "내리다", "걷다",
        "달리다", "앉다", "서다", "열다", "닫다",
        "찾다", "기다리다", "만나다", "연락하다", "전화하다",
        "예약하다", "확인하다", "신청하다", "시작하다", "끝내다",
        "부르다", "바꾸다", "선택하다", "도착하다", "출발하다",
    ],

    # ── 3. 감정·상태 (40개) ────────────────────────────────────────────
    "감정상태": [
        "기쁘다", "슬프다", "화나다", "무섭다", "놀라다",
        "부끄럽다", "외롭다", "행복하다", "불행하다", "걱정되다",
        "피곤하다", "배고프다", "배부르다", "목마르다", "졸리다",
        "아프다", "건강하다", "어지럽다", "열나다", "기침하다",
        "춥다", "덥다", "시원하다", "따뜻하다", "바쁘다",
        "한가하다", "좋다", "나쁘다", "힘들다", "쉽다",
        "어렵다", "중요하다", "필요하다", "급하다", "위험하다",
        "안전하다", "조용하다", "시끄럽다", "빠르다", "느리다",
    ],

    # ── 4. 장소·교통 (50개) ────────────────────────────────────────────
    "장소교통": [
        "집", "학교", "병원", "약국", "마트",
        "편의점", "식당", "카페", "은행", "경찰서",
        "공항", "기차역", "버스터미널", "지하철역", "화장실",
        "출구", "입구", "주차장", "엘리베이터", "계단",
        "버스", "지하철", "기차", "택시", "자동차",
        "자전거", "비행기", "배", "킥보드", "오토바이",
        "길", "지도", "주소", "노선", "정류장",
        "방향", "목적지", "횡단보도", "신호등", "환승",
        "요금", "표", "시간표", "탑승", "하차",
        "출발", "도착", "직진", "좌회전", "우회전",
    ],

    # ── 5. 사람·관계 (30개) ────────────────────────────────────────────
    "사람관계": [
        "나", "너", "우리", "가족", "아버지",
        "어머니", "형", "언니", "남동생", "여동생",
        "할아버지", "할머니", "남편", "아내", "아들",
        "딸", "친구", "선생님", "의사", "간호사",
        "경찰관", "손님", "직원", "동료", "이름",
        "나이", "남자", "여자", "어린이", "노인",
    ],

    # ── 6. 식사·쇼핑 (50개) ────────────────────────────────────────────
    "식사쇼핑": [
        "밥", "빵", "국수", "고기", "생선",
        "채소", "과일", "물", "커피", "주스",
        "우유", "아메리카노", "카페라떼", "김밥", "치킨",
        "포장", "배달", "영수증", "카드", "현금",
        "계산", "얼마", "할인", "주문", "메뉴",
        "추천", "자리", "예약", "영업시간", "봉투",
        "맛있다", "맵다", "달다", "짜다", "싱겁다",
        "뜨겁다", "차갑다", "크다", "작다", "신선하다",
        "세일", "교환", "환불", "무료", "유료",
        "비싸다", "싸다", "같이", "따로", "수량",
    ],

    # ── 7. 긴급·중요 (30개) ────────────────────────────────────────────
    "긴급중요": [
        "도움", "응급", "위험", "사고", "화재",
        "도둑", "분실", "미아", "구급차", "소방",
        "신고하다", "대피", "비상구", "조심", "멈추다",
        "구조", "약", "주사", "수술", "검사",
        "입원", "퇴원", "처방", "보험", "통증",
        "골절", "출혈", "기절", "알레르기", "증상",
    ],

    # ── 8. 직장·학교 (40개) ────────────────────────────────────────────
    "직장학교": [
        "회의", "업무", "보고", "제출", "마감",
        "휴가", "출근", "퇴근", "발표", "설명",
        "프로젝트", "계획", "결과", "면접", "취업",
        "급여", "계약", "서류", "서명", "승진",
        "수업", "공부", "숙제", "시험", "질문",
        "이해", "모르다", "졸업", "입학", "방학",
        "교육", "훈련", "평가", "목표", "성공",
        "실패", "노력", "동아리", "발표회", "상담",
    ],
}

ALL_PAIRS: list[tuple[str, str]] = [
    (word, cat)
    for cat, words in WORD_CATEGORIES.items()
    for word in words
]
ALL_WORDS: dict[str, str] = {w: c for w, c in ALL_PAIRS}

# ══════════════════════════════════════════════════════════════════════
# 문장 목록 (실용 빈출 문장 82개)
# ══════════════════════════════════════════════════════════════════════

SENTENCE_LIST: dict[str, list[str]] = {
    "인사소개": [
        "안녕하세요", "감사합니다", "죄송합니다", "반갑습니다",
        "잘 부탁드립니다", "안녕히 가세요", "다시 만나요",
        "도와드릴까요", "괜찮으세요", "처음 뵙겠습니다",
    ],
    "의사소통": [
        "천천히 말해주세요", "다시 말해주세요", "이해했어요",
        "모르겠어요", "잠깐만 기다려주세요", "조금 더 크게 말해주세요",
        "수화 할 수 있어요", "글로 써주세요", "통역사 불러주세요",
        "한 번 더 보여주세요",
    ],
    "위치길찾기": [
        "화장실이 어디예요", "출구가 어디예요", "지하철역이 어디예요",
        "버스 정류장이 어디예요", "가까운 병원이 어디예요",
        "이 길이 맞나요", "어떻게 가나요", "얼마나 걸리나요",
        "길을 잃었어요", "여기가 어디예요", "지도 보여주세요",
        "이 주소로 가주세요",
    ],
    "교통이동": [
        "이 버스 어디까지 가나요", "다음 정류장이 어디예요",
        "환승해야 하나요", "막차가 몇 시예요", "자리 있나요",
        "요금이 얼마예요", "카드 되나요", "내려주세요",
        "기차표 사고 싶어요", "택시 불러주세요",
        "몇 번 출구예요", "몇 정거장 남았어요",
    ],
    "쇼핑음식": [
        "이것 얼마예요", "할인 되나요", "영수증 주세요",
        "카드 결제 되나요", "교환 환불 되나요", "포장해 주세요",
        "아메리카노 한 잔 주세요", "물 한 잔 주세요",
        "이것 알레르기 있어요", "채식 메뉴 있나요",
        "추천 메뉴가 뭐예요", "포장 가능한가요",
    ],
    "의료응급": [
        "아파요", "도와주세요", "응급 상황이에요",
        "병원에 가야 해요", "약이 필요해요", "구급차 불러주세요",
        "여기가 아파요", "머리가 아파요", "배가 아파요",
        "넘어졌어요", "피가 나요", "숨쉬기가 힘들어요",
    ],
    "감정일상": [
        "배고파요", "목말라요", "피곤해요",
        "기다려 주세요", "급해요", "걱정돼요",
        "기뻐요", "슬퍼요", "화났어요", "무서워요",
        "오늘 날씨 좋네요", "몸이 안 좋아요",
    ],
}

# ══════════════════════════════════════════════════════════════════════
# AI Hub API 엔드포인트 (aihubshell 스크립트 기준)
# ══════════════════════════════════════════════════════════════════════

BASE_URL     = "https://api.aihub.or.kr"
EP_INFO      = "/info/{sn}.do"          # 파일 트리 조회 (인증 불필요)
EP_DOWNLOAD  = "/down/0.6/{sn}.do"      # 파일 다운로드 (apikey 헤더 필요)
DATASET_SN   = 103

# ══════════════════════════════════════════════════════════════════════
# 다운로드 프리셋
# fileSn 은 AI Hub 파일 트리(/info/103.do)에서 확인한 실제 값
# ══════════════════════════════════════════════════════════════════════

# 각 항목: (fileSn, 저장파일명, 예상크기MB, 데이터종류)
_PRESET_FILES: dict[str, list[tuple[int, str, int, str]]] = {
    "morpheme": [
        # 비수지(표정/머리/몸) 어노테이션만 — 약 230 MB
        (39601,  "train_real_word_morpheme.zip",   110, "word"),
        (39584,  "train_real_sen_morpheme.zip",     81, "sentence"),
        (39581,  "train_crowd_morpheme.zip",          8, "word"),
        (39478,  "val_real_word_morpheme.zip",       14, "word"),
        (39475,  "val_crowd_morpheme.zip",             1, "word"),
        (494853, "val_real_sen_morpheme.zip",        10, "sentence"),
    ],
    "minimal": [
        # 합성 키포인트 + 비수지 — 약 3 GB
        (39618,  "train_syn_word_keypoint.zip",   1023, "word"),
        (39617,  "train_syn_sen_keypoint.zip",     828, "sentence"),
        (39601,  "train_real_word_morpheme.zip",   110, "word"),
        (39584,  "train_real_sen_morpheme.zip",     81, "sentence"),
        (39481,  "val_syn_word_keypoint.zip",      450, "word"),
        (39480,  "val_syn_sen_keypoint.zip",       599, "sentence"),
    ],
    "standard": [
        # minimal + 검증 비수지 어노테이션 — 약 3.2 GB
        (39618,  "train_syn_word_keypoint.zip",   1023, "word"),
        (39617,  "train_syn_sen_keypoint.zip",     828, "sentence"),
        (39601,  "train_real_word_morpheme.zip",   110, "word"),
        (39584,  "train_real_sen_morpheme.zip",     81, "sentence"),
        (39581,  "train_crowd_morpheme.zip",          8, "word"),
        (39481,  "val_syn_word_keypoint.zip",      450, "word"),
        (39480,  "val_syn_sen_keypoint.zip",       599, "sentence"),
        (39478,  "val_real_word_morpheme.zip",      14, "word"),
        (39475,  "val_crowd_morpheme.zip",            1, "word"),
        (494853, "val_real_sen_morpheme.zip",       10, "sentence"),
    ],
}


def format_size(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


# ══════════════════════════════════════════════════════════════════════
# 설정 로드
# ══════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    if not CONFIG.exists():
        print(f"[오류] 설정 파일 없음: {CONFIG}")
        sys.exit(1)
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# HTTP 세션 (재시도 포함)
# ══════════════════════════════════════════════════════════════════════

def build_session(api_key: str, max_retries: int = 3,
                  retry_delay: float = 2.0) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "apikey":       api_key,   # AI Hub: 소문자 apikey
        "User-Agent":   "SignLanguageAI-Downloader/2.0",
    })
    retry = Retry(
        total=max_retries,
        backoff_factor=retry_delay,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# ══════════════════════════════════════════════════════════════════════
# 파일 트리 조회 (인증 불필요)
# ══════════════════════════════════════════════════════════════════════

def fetch_file_tree_raw(timeout: int = 30) -> Optional[str]:
    """GET /info/103.do — 응답 텍스트 반환 (인증 없이 가능)."""
    url = BASE_URL + EP_INFO.format(sn=DATASET_SN)
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        r.encoding = "utf-8"
        if r.status_code == 200:
            return r.text
        print(f"[HTTP {r.status_code}] 파일 트리 조회 실패: {url}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[네트워크 오류] {url} 연결 실패")
    except requests.exceptions.Timeout:
        print(f"[타임아웃] {url}")
    except Exception as e:
        print(f"[오류] 파일 트리 조회: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════
# 파일 다운로드
# ══════════════════════════════════════════════════════════════════════

def download_file(session: requests.Session, file_sn: int,
                  dest: Path, timeout: int = 300,
                  max_retries: int = 3) -> bool:
    """
    GET /down/0.6/103.do?fileSn={file_sn} 로 파일을 스트리밍 다운로드.
    헤더 'apikey' 는 세션에 이미 설정됨.
    """
    url    = BASE_URL + EP_DOWNLOAD.format(sn=DATASET_SN)
    params = {"fileSn": file_sn}

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, params=params,
                             stream=True, timeout=timeout,
                             allow_redirects=True) as r:
                if r.status_code == 401:
                    print(f"\n  [401 인증 실패] API 키를 확인하세요 (config.json → aihub.api_key)")
                    return False
                if r.status_code == 403:
                    print(f"\n  [403 접근 거부] AI Hub에서 해당 데이터셋 이용 신청 필요")
                    return False
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)

                with open(dest, "wb") as f, tqdm(
                    total=total or None,
                    unit="B", unit_scale=True,
                    desc=f"  {dest.name}",
                    leave=True,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 17):  # 128 KB
                        f.write(chunk)
                        bar.update(len(chunk))
            return True

        except requests.exceptions.ConnectionError:
            print(f"\n  [재시도 {attempt}/{max_retries}] 연결 오류: fileSn={file_sn}")
        except requests.exceptions.Timeout:
            print(f"\n  [재시도 {attempt}/{max_retries}] 타임아웃: fileSn={file_sn}")
        except Exception as e:
            print(f"\n  [재시도 {attempt}/{max_retries}] {e}")

        if dest.exists():
            dest.unlink()
        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return False


# ══════════════════════════════════════════════════════════════════════
# 압축 해제
# ══════════════════════════════════════════════════════════════════════

def _is_zip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def _is_tar(path: Path) -> bool:
    return tarfile.is_tarfile(str(path))


def extract_archive(archive: Path, dest_dir: Path) -> bool:
    """
    ZIP 또는 TAR → 압축 해제. TAR 안에 ZIP이 있으면 한 겹 더 해제.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  압축 해제: {archive.name} → {dest_dir}")

    try:
        if _is_zip(archive):
            with zipfile.ZipFile(archive, "r") as zf:
                members = zf.namelist()
                for m in tqdm(members, desc="  압축 해제", unit="파일", leave=False):
                    zf.extract(m, dest_dir)
            return True

        if _is_tar(archive):
            with tarfile.open(archive, "r:*") as tf:
                members = tf.getmembers()
                tf.extractall(dest_dir)
                # TAR 안의 ZIP도 재귀적으로 해제
                for m in members:
                    inner = dest_dir / m.name
                    if inner.suffix.lower() == ".zip" and _is_zip(inner):
                        sub_dir = inner.with_suffix("")
                        print(f"  내부 ZIP 해제: {inner.name}")
                        extract_archive(inner, sub_dir)
                        inner.unlink()
            return True

        print(f"  [경고] 알 수 없는 형식: {archive.name}")
        return False

    except Exception as e:
        print(f"  [오류] 압축 해제 실패: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# 목적지 결정 유틸
# ══════════════════════════════════════════════════════════════════════

def _dest_for(data_kind: str) -> Path:
    """'sentence' → DATA_SENT, 그 외 → DATA"""
    return DATA_SENT if data_kind == "sentence" else DATA


# ══════════════════════════════════════════════════════════════════════
# 용량 계산
# ══════════════════════════════════════════════════════════════════════

def measure_dir(path: Path) -> float:
    """MB 단위 실제 용량"""
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Hub 수어 데이터셋(103) 다운로더 — 프리셋 방식"
    )
    p.add_argument(
        "--preset", choices=list(_PRESET_FILES.keys()),
        metavar="|".join(_PRESET_FILES.keys()),
        help=(
            "morpheme: 비수지 어노테이션만 (~230 MB) | "
            "minimal: 합성 키포인트 + 비수지 (~3 GB) | "
            "standard: minimal + 검증 비수지 (~3.2 GB)"
        ),
    )
    p.add_argument(
        "--file-sn", type=int, metavar="N",
        help="단일 fileSn 직접 지정 (--out-name 과 함께 사용)",
    )
    p.add_argument(
        "--out-name", metavar="파일명",
        help="--file-sn 사용 시 저장 파일명 (예: my_file.zip)",
    )
    p.add_argument(
        "--no-extract", action="store_true",
        help="다운로드만 하고 압축 해제 건너뛰기",
    )
    p.add_argument(
        "--keep-archives", action="store_true",
        help="압축 해제 후 원본 압축 파일 보존",
    )
    p.add_argument(
        "--list-files", action="store_true",
        help="AI Hub 파일 트리 조회 후 종료 (인증 불필요)",
    )
    p.add_argument(
        "--list-words", action="store_true",
        help="단어/문장 목록 출력 후 종료",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="실제 다운로드 없이 계획만 출력",
    )
    return p.parse_args()


def print_word_list() -> None:
    print("\n[다운로드 대상 단어 목록]")
    total = 0
    for cat, words in WORD_CATEGORIES.items():
        print(f"\n  [{cat}] {len(words)}개")
        for i, w in enumerate(words, 1):
            print(f"    {i:2d}. {w}")
        total += len(words)
    total_s = sum(len(v) for v in SENTENCE_LIST.values())
    print(f"\n총 {total}개 단어, {total_s}개 문장")
    print("학습 데이터는 --preset minimal 또는 --preset morpheme 으로 다운로드하세요.")


def print_file_tree() -> None:
    print("\nAI Hub 파일 트리 조회 중 (인증 불필요)...")
    text = fetch_file_tree_raw()
    if text is None:
        print("[실패] 파일 트리를 가져올 수 없습니다.")
        print("  직접 확인: https://api.aihub.or.kr/info/103.do")
        return
    # 응답이 JSON인지 텍스트인지 판별
    try:
        data = json.loads(text)
        items = (
            data.get("result")
            or data.get("data")
            or (data if isinstance(data, list) else [])
        )
        print(f"\n[데이터셋 103 파일 목록] {len(items)}개 항목")
        for item in items:
            sn   = item.get("fileSn") or item.get("file_sn") or "?"
            name = item.get("fileNm") or item.get("fileName") or item.get("name") or "?"
            size = item.get("fileSize") or item.get("file_size") or 0
            mb   = int(size) / (1024 * 1024) if size else 0
            print(f"  fileSn={sn:>8}  {name:<45}  {format_size(mb):>8}")
    except json.JSONDecodeError:
        # 텍스트 형식 그대로 출력
        print("\n[데이터셋 103 파일 목록]")
        print(text)


def main() -> None:
    args = parse_args()

    if args.list_words:
        print_word_list()
        return

    if args.list_files:
        print_file_tree()
        return

    # ── 설정 로드 ──────────────────────────────────────────────────────
    cfg_raw = load_config()
    aihub   = cfg_raw.get("aihub", {})
    api_key = aihub.get("api_key", "")

    if not api_key or api_key.startswith("YOUR_"):
        print("[오류] API 키가 설정되지 않았습니다.")
        print("  config.json → aihub.api_key 에 AI Hub Open API 키를 입력하세요.")
        sys.exit(1)

    timeout     = aihub.get("timeout", 300)
    max_retries = aihub.get("max_retries", 3)
    retry_delay = aihub.get("retry_delay", 2.0)

    # ── 다운로드 대상 결정 ─────────────────────────────────────────────
    if args.file_sn:
        out_name = args.out_name or f"aihub_{args.file_sn}.zip"
        targets  = [(args.file_sn, out_name, 0, "word")]
    elif args.preset:
        targets = _PRESET_FILES[args.preset]
    else:
        print("[오류] --preset 또는 --file-sn 을 지정하세요.")
        print(f"  예시: python {Path(__file__).name} --preset minimal")
        sys.exit(1)

    total_est_mb = sum(t[2] for t in targets)

    print("=" * 60)
    print("  AI Hub 수어 데이터셋 다운로더 v2")
    print("=" * 60)
    if args.preset:
        print(f"  프리셋     : {args.preset}")
    print(f"  파일 수    : {len(targets)}개")
    print(f"  예상 용량  : {format_size(total_est_mb)}")
    print(f"  저장 경로  : {DATA_RAW}")
    print(f"  모드       : {'[DRY-RUN]' if args.dry_run else '실제 다운로드'}")
    print("=" * 60)
    for file_sn, name, size_mb, kind in targets:
        print(f"  fileSn={file_sn:<8}  {name:<45}  ~{format_size(size_mb):>7}  [{kind}]")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] 위 파일들을 다운로드하고 압축 해제합니다.")
        print("실제 실행: --dry-run 플래그를 제거하세요.")
        return

    # ── 다운로드 + 압축 해제 ───────────────────────────────────────────
    session = build_session(api_key,
                            max_retries=max_retries,
                            retry_delay=retry_delay)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    ok_count = fail_count = 0
    for file_sn, name, size_mb, data_kind in targets:
        archive = DATA_RAW / name
        dest    = _dest_for(data_kind)

        # 이미 압축 해제된 경우 스킵
        if not archive.exists() and dest.exists() and any(dest.rglob("*.json")):
            print(f"  [SKIP] {name} — 이미 추출됨")
            ok_count += 1
            continue

        print(f"\n[다운로드] fileSn={file_sn}  →  {name}")

        if archive.exists() and archive.stat().st_size > 1024:
            print(f"  이미 다운로드된 파일 사용: {archive}")
        else:
            ok = download_file(session, file_sn, archive,
                               timeout=timeout, max_retries=max_retries)
            if not ok:
                print(f"  [실패] {name} 다운로드 실패")
                fail_count += 1
                continue

        if not args.no_extract:
            extract_dir = dest / Path(name).stem
            ok = extract_archive(archive, extract_dir)
            if ok:
                if not args.keep_archives:
                    archive.unlink(missing_ok=True)
                ok_count += 1
            else:
                fail_count += 1
        else:
            ok_count += 1

    # ── 요약 ─────────────────────────────────────────────────────────
    word_mb = measure_dir(DATA)
    sent_mb = measure_dir(DATA_SENT)
    print("\n" + "=" * 60)
    print("  완료 요약")
    print("=" * 60)
    print(f"  성공: {ok_count}개  /  실패: {fail_count}개")
    print(f"  data/aihub          : {format_size(word_mb)}")
    print(f"  data/aihub_sentences: {format_size(sent_mb)}")
    if ok_count > 0 and fail_count == 0:
        print("\n  다음 단계: python data_collection/train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
