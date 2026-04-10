from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import random
from datetime import datetime, date, timedelta

from langgraph_workflow.workflow import run_weekly_workflow

app = FastAPI(title="칭구칭구 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 인메모리 DB (데모용) ──────────────────────
fake_users = {
    "demo": {
        "user_id": "uuid-demo-001",
        "password": "demo1234",
        "nickname": "칭구",
        "birthdate": "2007-03-15",
        "gender": "female",
        "top_emotion": "sad",
        "created_at": "2025-01-01T00:00:00"
    }
}

fake_diaries = [
    {
        "diary_id": "d-001", "user_id": "uuid-demo-001",
        "diary_date": "2025-10-30", "week_start": "2025-10-27",
        "content": "오늘 배드민턴을 치다가 넘어졌는데 너무 분했어. 친구들이 웃어서 더 속상했어.",
        "char_count": 42,
        "top_emotion": "angry",
        "emotions": [
            {"label": "angry", "prob": 0.82}, {"label": "hurt", "prob": 0.61},
            {"label": "embarrassed", "prob": 0.41}, {"label": "sad", "prob": 0.28},
            {"label": "anxious", "prob": 0.15}
        ],
        "created_at": "2025-10-30T05:10:00", "updated_at": "2025-10-30T05:10:00"
    },
    {
        "diary_id": "d-002", "user_id": "uuid-demo-001",
        "diary_date": "2025-10-24", "week_start": "2025-10-20",
        "content": "수학 문제를 풀다 같은 곳에서 또 막혀서 진짜 짜증났어. 아 진짜 하고 책상 두드렸는데 손이 아팠어.",
        "char_count": 55,
        "top_emotion": "angry",
        "emotions": [
            {"label": "angry", "prob": 0.74}, {"label": "sad", "prob": 0.55},
            {"label": "anxious", "prob": 0.38}, {"label": "hurt", "prob": 0.22},
            {"label": "embarrassed", "prob": 0.10}
        ],
        "created_at": "2025-10-24T18:13:00", "updated_at": "2025-10-24T18:13:00"
    },
    {
        "diary_id": "d-003", "user_id": "uuid-demo-001",
        "diary_date": "2025-10-22", "week_start": "2025-10-20",
        "content": "새벽에 잠이 살짝 깨서 하루 종일 졸렸어. 학교 가서도 집중이 안 됐고 그래도 학교 끝나고 20분 낮잠 잤고",
        "char_count": 58,
        "top_emotion": "sad",
        "emotions": [
            {"label": "sad", "prob": 0.68}, {"label": "anxious", "prob": 0.49},
            {"label": "hurt", "prob": 0.33}, {"label": "angry", "prob": 0.21},
            {"label": "embarrassed", "prob": 0.12}
        ],
        "created_at": "2025-10-22T14:22:00", "updated_at": "2025-10-22T14:22:00"
    },
]

fake_reports = {}

# ── KoBERT 감정 분석 시뮬레이션 ──────────────
EMOTION_KEYWORDS = {
    "sad":          ["슬프","눈물","울","힘들","외로","우울"],
    "angry":        ["화나","짜증","분했","열받","억울","화가"],
    "anxious":      ["불안","걱정","무서","두려","긴장"],
    "happy":        ["기뻐","행복","좋아","즐거","설레","신나"],
    "hurt":         ["상처","아파","속상","서운","미워"],
    "embarrassed":  ["당황","어색","민망","부끄","창피"],
}

def simulate_kobert(text: str):
    import random
    emotions = list(EMOTION_KEYWORDS.keys())
    weights = [random.uniform(0.1, 1.0) for _ in emotions]
    total = sum(weights)
    probs = [round(w / total, 2) for w in weights]
    result = sorted(zip(emotions, probs), key=lambda x: -x[1])

    # 키워드 기반 보정
    top = result[0][0]
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(k in text for k in keywords):
            top = emotion
            break

    warning = "text_too_short" if len(text) < 50 else None
    return top, [{"label": e, "prob": p} for e, p in result[:5]], warning


# ══════════════════════════════════════════════
# 요청 모델
# ══════════════════════════════════════════════
class LoginRequest(BaseModel):
    user_name: str
    user_password: str

class SignUpRequest(BaseModel):
    user_name: str
    user_password: str
    nickname: str
    birthdate: Optional[str] = None
    gender: Optional[str] = None

class DiaryCreateRequest(BaseModel):
    content: str
    diary_date: str

class UserUpdateRequest(BaseModel):
    nickname: Optional[str] = None
    birthdate: Optional[str] = None
    gender: Optional[str] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class SettingUpdateRequest(BaseModel):
    push_enabled: Optional[bool] = None
    theme: Optional[str] = None


# ══════════════════════════════════════════════
# 메인 페이지
# ══════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


# ══════════════════════════════════════════════
# Auth API
# ══════════════════════════════════════════════
@app.post("/auth/login")
async def login(req: LoginRequest):
    user = fake_users.get(req.user_name)
    if not user or user["password"] != req.user_password:
        raise HTTPException(401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return {
        "data": {
            "access_token": f"demo-jwt-{req.user_name}",
            "token_type": "bearer",
            "user": {
                "user_name": req.user_name,
                "nickname": user["nickname"],
                "top_emotion": user["top_emotion"]
            }
        },
        "message": "로그인이 완료되었습니다."
    }

@app.post("/auth/signup", status_code=201)
async def signup(req: SignUpRequest):
    if req.user_name in fake_users:
        raise HTTPException(409, detail="이미 사용 중인 아이디입니다.")
    if len(req.user_password) < 8:
        raise HTTPException(400, detail="비밀번호는 8자 이상이어야 합니다.")
    fake_users[req.user_name] = {
        "user_id": f"uuid-{req.user_name}",
        "password": req.user_password,
        "nickname": req.nickname,
        "birthdate": req.birthdate,
        "gender": req.gender,
        "top_emotion": None,
        "created_at": datetime.now().isoformat()
    }
    return {"data": {"user_name": req.user_name}, "message": "회원가입이 완료되었습니다."}

@app.get("/auth/check-username")
async def check_username(user_name: str):
    return {"available": user_name not in fake_users}

@app.post("/auth/logout")
async def logout():
    return {"message": "로그아웃이 완료되었습니다."}


# ══════════════════════════════════════════════
# Users API
# ══════════════════════════════════════════════
@app.get("/users/me")
async def get_me():
    u = fake_users["demo"]
    return {
        "data": {
            "user_id": u["user_id"],
            "user_name": "demo",
            "nickname": u["nickname"],
            "birthdate": u.get("birthdate"),
            "gender": u.get("gender"),
            "top_emotion": u["top_emotion"],
            "created_at": u.get("created_at")
        }
    }

@app.patch("/users/me")
async def update_me(req: UserUpdateRequest):
    u = fake_users["demo"]
    if req.nickname: u["nickname"] = req.nickname
    if req.birthdate: u["birthdate"] = req.birthdate
    if req.gender: u["gender"] = req.gender
    return {"data": {"nickname": u["nickname"]}, "message": "정보가 수정되었습니다."}

@app.patch("/users/me/password")
async def change_password(req: PasswordChangeRequest):
    u = fake_users["demo"]
    if u["password"] != req.current_password:
        raise HTTPException(401, detail="현재 비밀번호가 올바르지 않습니다.")
    if len(req.new_password) < 8:
        raise HTTPException(400, detail="비밀번호는 8자 이상이어야 합니다.")
    u["password"] = req.new_password
    return {"message": "비밀번호가 변경되었습니다."}


# ══════════════════════════════════════════════
# Diaries API
# ══════════════════════════════════════════════
@app.get("/diaries")
async def get_diaries():
    return {"total": len(fake_diaries), "diaries": fake_diaries}

@app.post("/diaries", status_code=201)
async def create_diary(req: DiaryCreateRequest):
    # 하루 1개 제한 체크
    existing = [d for d in fake_diaries if d["diary_date"] == req.diary_date]
    if existing:
        raise HTTPException(409, detail="해당 날짜의 일기가 이미 존재합니다.")

    top_emotion, emotions, warning = simulate_kobert(req.content)

    # week_start 계산 (해당 날짜의 월요일)
    dt = datetime.strptime(req.diary_date, "%Y-%m-%d")
    week_start = (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")

    diary_id = f"d-{len(fake_diaries)+1:03d}"
    new_diary = {
        "diary_id": diary_id,
        "user_id": "uuid-demo-001",
        "diary_date": req.diary_date,
        "week_start": week_start,
        "content": req.content,
        "char_count": len(req.content),
        "top_emotion": top_emotion,
        "emotions": emotions,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    fake_diaries.insert(0, new_diary)
    fake_users["demo"]["top_emotion"] = top_emotion

    return {
        "data": {
            "diary_id": diary_id,
            "diary_date": req.diary_date,
            "week_start": week_start,
            "emotion_result": {
                "result_id": f"r-{diary_id}",
                "top_emotion": top_emotion,
                "emotions": emotions,
                "created_at": datetime.now().isoformat()
            },
            "warning": warning
        },
        "message": "일기가 저장되었습니다."
    }

@app.get("/diaries/{diary_id}")
async def get_diary(diary_id: str):
    d = next((d for d in fake_diaries if d["diary_id"] == diary_id), None)
    if not d:
        raise HTTPException(404, detail="일기를 찾을 수 없습니다.")
    return {"data": d}

@app.delete("/diaries/{diary_id}")
async def delete_diary(diary_id: str):
    global fake_diaries
    before = len(fake_diaries)
    fake_diaries = [d for d in fake_diaries if d["diary_id"] != diary_id]
    if len(fake_diaries) == before:
        raise HTTPException(404, detail="일기를 찾을 수 없습니다.")

    # top_emotion 재계산
    if fake_diaries:
        fake_users["demo"]["top_emotion"] = fake_diaries[0]["top_emotion"]
    else:
        fake_users["demo"]["top_emotion"] = None

    return {"message": "일기가 삭제되었습니다."}


# ══════════════════════════════════════════════
# Reports API
# ══════════════════════════════════════════════
@app.get("/reports")
async def get_reports():
    if not fake_reports:
        return {"total": 0, "reports": []}
    return {"total": len(fake_reports), "reports": list(fake_reports.values())}

@app.get("/reports/{week_start}")
async def get_report(week_start: str):
    if week_start not in fake_reports:
        raise HTTPException(404, detail="해당 주 리포트 데이터가 없습니다.")
    return {"data": fake_reports[week_start]}


# ══════════════════════════════════════════════
# Settings API
# ══════════════════════════════════════════════
fake_settings = {"push_enabled": True, "theme": "light", "updated_at": datetime.now().isoformat()}

@app.get("/settings")
async def get_settings():
    return {"data": fake_settings}

@app.patch("/settings")
async def update_settings(req: SettingUpdateRequest):
    if req.push_enabled is not None:
        fake_settings["push_enabled"] = req.push_enabled
    if req.theme:
        if req.theme not in ["light", "dark"]:
            raise HTTPException(400, detail="theme은 light 또는 dark만 허용됩니다.")
        fake_settings["theme"] = req.theme
    fake_settings["updated_at"] = datetime.now().isoformat()
    return {"data": fake_settings, "message": "설정이 변경되었습니다."}


# ══════════════════════════════════════════════
# App API
# ══════════════════════════════════════════════
@app.get("/app/info")
async def app_info():
    return {
        "app_name": "칭구칭구",
        "version": "1.0.0",
        "description": "청소년 마음 보조 일기 어플리케이션",
        "contact": "support@chingguchingg.com"
    }


# ══════════════════════════════════════════════
# LangGraph 수동 트리거 (데모용)
# ══════════════════════════════════════════════
@app.post("/internal/run-workflow")
async def run_workflow():
    diaries_text = [d["content"] for d in fake_diaries]
    emotions_data = [{"top_emotion": d["top_emotion"], "emotions": d["emotions"]} for d in fake_diaries]

    today = date.today()
    week_start = str(today - timedelta(days=today.weekday()))
    week_end = str(today - timedelta(days=today.weekday()) + timedelta(days=6))

    result = run_weekly_workflow(
        user_id="uuid-demo-001",
        user_nickname=fake_users["demo"]["nickname"],
        diary_count=len(fake_diaries),
        diaries_text=diaries_text,
        emotions_data=emotions_data,
        week_start=week_start,
        week_end=week_end,
    )

    result["week_start"] = week_start
    result["week_end"] = week_end
    result["generated_at"] = datetime.now().isoformat()
    fake_reports[week_start] = result

    if result.get("report_status") == "generated" and result.get("top_emotions"):
        fake_users["demo"]["top_emotion"] = result["top_emotions"][0]["label"]

    return {"message": "워크플로우 실행 완료", "result": result}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
