# 칭구칭구 🫧
> 청소년 마음 보조 일기 어플리케이션

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://langchain-ai.github.io/langgraph)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-orange.svg)](https://swagger.io)

---

## 📖 서비스 개요

청소년의 정서 건강을 위한 일기 기반 감정 분석 + 주간 리포트 자동 생성 서비스입니다.
<img width="1232" height="680" alt="image" src="https://github.com/user-attachments/assets/63a9233b-5e5a-4d5d-a1a1-94ad4765d5e9" />

- **KoBERT 기반** 감정 분석 (6가지 감정)
- **LangGraph 5노드 워크플로우** 기반 주간 리포트 자동 생성
- **PHQ-9** 우울도 자동 평가
- **Escalation HITL** — 고위험군 탐지 시 상담사·교사 알림

---

## 🏗️ 시스템 아키텍처

```
[실시간]
사용자 일기 작성
    → POST /diaries
    → KoBERT 감정 분석
    → emotion_result 저장
    → user.top_emotion 갱신
    → 메인 화면 캐릭터 반영

[주간 배치]
APScheduler (매주 일요일 23:59)
    → LangGraph 5노드 워크플로우
    → weekly_report 저장
    → 리포트 조회 가능
```

---

## ⚡ LangGraph 워크플로우
<img width="1234" height="696" alt="image" src="https://github.com/user-attachments/assets/66aea7c2-e1bc-48f9-95c7-4110a86f2b23" />

```
NodeA (데이터 수집·유효성 검사)
  ├── 일기 0건 → 즉시 종료 (row 미생성)
  └── 1건 이상 → NodeB

NodeB (데이터 충분성 분기) ← conditional edge
  ├── 충분 → full 경로 → NodeC
  └── 부족 → lite (report_status=insufficient_data)

NodeC (PHQ-9 및 위험도 평가)
  → LLM 기반 PHQ-9 proxy scoring
  → depression_level, self_harm_risk 산출

NodeD (위험도 기반 후속 생성 분기) ← conditional edge
  ├── normal → 일반 요약 + 조언
  └── risk   → 위기 문구 + 상담 리소스
              + Escalation HITL (self_harm_risk=active 시)

NodeE (저장 및 상태 기록)
  → weekly_report INSERT
  → user.top_emotion 재계산
  → 실패 시 checkpoint → NodeE만 재실행
```

---

## 🗄️ 데이터 모델

| 테이블 | 설명 | 핵심 설계 |
|---|---|---|
| `user` | 회원 정보 | `top_emotion` 캐시 필드 (메인화면 성능) |
| `user_setting` | 앱 설정 | `user_id` UNIQUE로 1:1 보장 |
| `diary` | 일별 일기 | `(user_id, diary_date)` UNIQUE → 하루 1개 |
| `emotion_result` | KoBERT 감정 분석 결과 | `user_id` 비정규화 → 주간 집계 JOIN 제거 |
| `weekly_report` | 주간 리포트 | `report_status`, `node_b/d_branch`, `escalation_status` |

---

## 🚀 실행 방법

```bash
# 1. 클론
git clone https://github.com/your-id/chingguchingg.git
cd chingguchingg

# 2. 가상환경
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 실행
python main.py

# 5. 브라우저
open http://127.0.0.1:8000
```

---

## 🔑 데모 계정

| 항목 | 값 |
|---|---|
| 아이디 | `demo` |
| 비밀번호 | `demo1234` |

---

## 📱 데모 흐름

1. `demo / demo1234` 로그인
2. 메인화면 → 감정 캐릭터 확인
3. 일기 탭 → 일기 작성 (KoBERT 감정 분석)
4. **리포트 탭 → ⚡ LangGraph 주간 워크플로우 실행**
   - NodeA → B → C → D → E 실행 로그 표시
   - 감정 분포 그래프 + PHQ-9 점수 + GPT 조언 확인

---

## 📡 API 명세

```
Swagger UI: http://127.0.0.1:8000/docs
OpenAPI:    http://127.0.0.1:8000/openapi.json
```

| 태그 | 엔드포인트 | 설명 |
|---|---|---|
| Auth | `POST /auth/login` | 로그인 → JWT 발급 |
| Auth | `POST /auth/signup` | 회원가입 |
| Auth | `GET /auth/check-username` | 아이디 중복 확인 |
| Diaries | `POST /diaries` | 일기 작성 + KoBERT 분석 |
| Diaries | `GET /diaries` | 일기 목록 |
| Diaries | `DELETE /diaries/{id}` | 일기 삭제 |
| Reports | `GET /reports` | 주간 리포트 목록 |
| Reports | `GET /reports/{week_start}` | 리포트 상세 |

---

## 🔧 기술 스택

| 분류 | 기술 |
|---|---|
| Backend | FastAPI, Python 3.10+ |
| AI Workflow | LangGraph, APScheduler |
| AI Models | KoBERT (감정 분석), GPT (PHQ-9 추론) |
| API 명세 | OpenAPI 3.0 (Swagger) |
| DB | MySQL (데모: 인메모리) |
| Auth | JWT Bearer |

---

## 📁 프로젝트 구조

```
chingguchingg/
├── main.py                      # FastAPI 서버 + API 엔드포인트
├── requirements.txt
├── templates/
│   └── index.html               # 프론트엔드 (모바일 UI)
├── langgraph_workflow/
│   ├── __init__.py
│   └── workflow.py              # LangGraph NodeA~E 구현
└── README.md
```

---

## 👩‍💻 개발자

**@kxmdoyn**

---

## 📄 License

MIT License
