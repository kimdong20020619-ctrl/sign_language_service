# 🤟 수화 소통 서비스

한국 수화(KSL)를 실시간으로 인식하고 자연스러운 한국어 문장으로 번역해주는 데스크톱 애플리케이션입니다.

## 주요 기능

- **실시간 수화 인식**: 웹캠으로 손 동작을 캡처하여 MediaPipe로 랜드마크 추출
- **이중 분류 모델**
  - 정적 수화: RandomForest (단일 프레임 특징벡터)
  - 동적 수화: LSTM 시퀀스 분류기
- **AI 문장 교정**: Ollama(EXAONE 3.5) 또는 Claude로 수화 단어 → 자연스러운 한국어 문장 변환
- **상황별 번역 모드**: 카페, 병원, 은행, 학교, 마트, 직장
- **STT**: Whisper로 음성 → 텍스트 입력 지원
- **TTS**: gTTS로 번역 결과 음성 출력
- **대화 기록**: SQLite DB에 세션 저장

## 프로젝트 구조

```
sign_language_service/
├── main.py                   # 진입점
├── config.json               # 설정 파일
├── requirements.txt
├── core/
│   ├── camera.py             # 카메라 스레드 (QThread)
│   ├── mediapipe_engine.py   # MediaPipe 손 랜드마크 추출
│   ├── mediapipe_worker.py   # MediaPipe 워커 스레드
│   ├── gesture_classifier.py # 정적/동적 수화 분류기
│   ├── sentence_builder.py   # 단어 확정 → 문장 조합
│   └── context_manager.py    # 상황 모드 관리
├── ai/
│   ├── ollama_bridge.py      # Ollama(EXAONE) 연동
│   ├── claude_bridge.py      # Claude API 연동
│   └── prompt_templates.py   # 시스템 프롬프트 템플릿
├── ui/
│   ├── main_window.py        # PySide6 메인 윈도우
│   ├── camera_widget.py      # 카메라 뷰 위젯
│   ├── chat_panel.py         # 번역 결과 채팅 패널
│   ├── quick_buttons.py      # 상황별 빠른 버튼
│   └── settings_dialog.py    # 설정 다이얼로그
├── stt/
│   ├── whisper_engine.py     # Whisper STT 엔진
│   └── stt_engine.py        # STT 베이스 클래스
├── tts/
│   ├── gtts_engine.py        # gTTS + pygame 재생
│   └── tts_engine.py        # TTS 베이스 클래스
├── db/
│   └── history_db.py         # SQLite 대화 기록
├── data_collection/
│   ├── aihub_downloader.py   # AI Hub 수어 데이터셋 다운로드
│   ├── collect_data.py       # 웹캠으로 직접 데이터 수집
│   └── train_model.py        # 모델 학습 파이프라인
└── data/                     # 학습 데이터 (git 제외)
    └── aihub/                # AI Hub 수어 데이터셋
```

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Ollama 설치 및 모델 다운로드

```bash
# https://ollama.com 에서 Ollama 설치 후
ollama pull exaone3.5
```

### 3. config.json 설정

```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "exaone3.5"
  },
  "claude": {
    "api_key": "YOUR_CLAUDE_API_KEY_HERE"
  }
}
```

## 실행

```bash
python main.py
```

> 수화 인식 모델(`models/gesture_model.pkl`)이 없으면 데모 모드로 실행됩니다.

## 모델 학습

### 방법 1: AI Hub 데이터셋 사용

AI Hub([aihub.or.kr](https://aihub.or.kr))에서 **수어 영상** 데이터셋을 신청·다운로드한 후:

```bash
# 1. 데이터 전처리 및 특징 추출
python data_collection/collect_data.py

# 2. 모델 학습 (RandomForest + LSTM)
python data_collection/train_model.py
```

### 방법 2: 웹캠으로 직접 수집

```bash
python data_collection/collect_data.py --source webcam
```

학습 완료 후 `models/gesture_model.pkl`이 생성됩니다.

## 데이터 흐름

```
웹캠 프레임
  → MediaPipe (손 랜드마크 21 × 3)
  → 특징 추출
  → 정적/동적 분류기 (15프레임 확정)
  → 단어 시퀀스 → 문장 조합 (3초 정지)
  → Ollama/Claude 문장 교정
  → 채팅 패널 표시 + gTTS 음성 출력
  → SQLite 저장
```

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | PySide6 |
| 손 인식 | MediaPipe |
| 정적 분류 | scikit-learn (RandomForest) |
| 동적 분류 | TensorFlow / Keras (LSTM) |
| AI 번역 | Ollama (EXAONE 3.5), Claude API |
| STT | OpenAI Whisper |
| TTS | gTTS + pygame |
| DB | SQLite |
