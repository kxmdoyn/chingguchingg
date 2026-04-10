"""
칭구칭구 LangGraph 주간 배치 워크플로우
NodeA → NodeB → NodeC → NodeD → NodeE
"""
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime

# ── State 정의 ────────────────────────────────
class WorkflowState(TypedDict):
    # 입력
    user_id: str
    user_nickname: str
    diary_count: int
    diaries_text: List[str]
    emotions_data: List[Dict]
    week_start: str
    week_end: str

    # NodeA 결과
    total_chars: int
    valid_emotion_count: int
    node_a_status: str          # "ok" | "skip"

    # NodeB 결과
    report_type: str            # "full" | "lite"
    node_b_branch: str          # "full" | "lite"

    # NodeC 결과
    top_emotions: Optional[List[Dict]]
    phq9_scores: Optional[Dict]
    phq9_total: Optional[int]
    depression_level: Optional[str]
    self_harm_risk: str         # "none" | "passive" | "active"

    # NodeD 결과
    node_d_branch: str          # "normal" | "risk"
    weekly_summary: Optional[str]
    advice: Optional[List[str]]
    resources: Optional[List[Dict]]

    # NodeE 결과
    report_status: str          # "generated" | "insufficient_data" | "failed"
    node_e_status: str
    diary_warning: Optional[str]
    escalation_status: str      # "none" | "sent"
    escalation_sent_at: Optional[str]
    langgraph_state: Optional[Dict]

    errors: List[str]


# ══════════════════════════════════════════════
# NodeA: 주간 데이터 수집 및 유효성 검사
# ══════════════════════════════════════════════
def node_a(state: WorkflowState) -> WorkflowState:
    print(f"\n{'='*50}")
    print(f"[NodeA] 주간 데이터 수집 및 유효성 검사")
    print(f"{'='*50}")
    try:
        diary_count = state["diary_count"]
        diaries_text = state["diaries_text"]

        if diary_count == 0:
            print("[NodeA] 일기 0건 → 즉시 종료 (row 미생성)")
            return {**state,
                    "node_a_status": "skip",
                    "report_status": "no_diary",
                    "errors": state.get("errors", [])}

        total_chars = sum(len(t) for t in diaries_text)
        valid_emotion_count = sum(
            1 for e in state["emotions_data"] if e.get("top_emotion")
        )

        print(f"[NodeA] 일기 {diary_count}건 / 총 {total_chars}자 / 감정결과 {valid_emotion_count}건 확인")
        return {**state,
                "total_chars": total_chars,
                "valid_emotion_count": valid_emotion_count,
                "node_a_status": "ok"}
    except Exception as e:
        print(f"[NodeA] 오류: {e}")
        return {**state,
                "node_a_status": "skip",
                "report_status": "failed",
                "errors": state.get("errors", []) + [f"NodeA: {str(e)}"]}


# ══════════════════════════════════════════════
# NodeB: 데이터 충분성 분기 (품질 검증)
# ══════════════════════════════════════════════
def node_b(state: WorkflowState) -> WorkflowState:
    print(f"\n{'='*50}")
    print(f"[NodeB] 데이터 충분성 분기")
    print(f"{'='*50}")
    try:
        diary_count = state["diary_count"]
        total_chars = state.get("total_chars", 0)
        valid_emotion_count = state.get("valid_emotion_count", 0)

        # 충분성 기준: 일기 2건 이상 + 총 100자 이상 + 감정 결과 존재
        is_sufficient = (
            diary_count >= 2 and
            total_chars >= 100 and
            valid_emotion_count >= 1
        )

        # 일기는 있지만 데이터 부족 시 diary_warning 설정
        diary_warning = None
        if diary_count < 3:
            diary_warning = "insufficient"

        if is_sufficient:
            print(f"[NodeB] 데이터 충분 → full 정식 경로")
            return {**state,
                    "report_type": "full",
                    "node_b_branch": "full",
                    "diary_warning": diary_warning}
        else:
            print(f"[NodeB] 데이터 부족 → lite 경량 경로 (report_status=insufficient_data)")
            return {**state,
                    "report_type": "lite",
                    "node_b_branch": "lite",
                    "report_status": "insufficient_data",
                    "diary_warning": diary_warning}
    except Exception as e:
        print(f"[NodeB] 오류: {e}")
        return {**state,
                "node_b_branch": "lite",
                "report_status": "insufficient_data",
                "errors": state.get("errors", []) + [f"NodeB: {str(e)}"]}


# ══════════════════════════════════════════════
# NodeC: PHQ-9 및 위험도 평가 (리스크 평가 노드)
# ══════════════════════════════════════════════
def node_c(state: WorkflowState) -> WorkflowState:
    print(f"\n{'='*50}")
    print(f"[NodeC] PHQ-9 및 위험도 평가 (LLM 기반 proxy scoring)")
    print(f"{'='*50}")
    try:
        diaries_text = state["diaries_text"]
        emotions_data = state["emotions_data"]
        full_text = " ".join(diaries_text)

        # ── 주간 감정 집계 ──────────────────────
        emotion_counts: Dict[str, int] = {}
        emotion_probs: Dict[str, float] = {}
        for e_data in emotions_data:
            for e in e_data.get("emotions", []):
                label = e["label"]
                emotion_counts[label] = emotion_counts.get(label, 0) + 1
                emotion_probs[label] = emotion_probs.get(label, 0) + e["prob"]

        top_emotions = sorted([
            {"label": k,
             "count": v,
             "avg_prob": round(emotion_probs[k] / v, 2)}
            for k, v in emotion_counts.items()
        ], key=lambda x: -x["count"])[:5]

        print(f"[NodeC] 주간 상위 감정: {[e['label'] for e in top_emotions[:3]]}")

        # ── GPT PHQ-9 proxy scoring 시뮬레이션 ──
        phq9_scores = _simulate_phq9(full_text)
        phq9_total = sum(phq9_scores.values())

        # 우울도 기준: 0-4=normal, 5-9=mild, 10-14=moderate, 15-19=moderately_severe, 20-27=severe
        if phq9_total <= 4:
            depression_level = "normal"
        elif phq9_total <= 9:
            depression_level = "mild"
        elif phq9_total <= 14:
            depression_level = "moderate"
        elif phq9_total <= 19:
            depression_level = "moderately_severe"
        else:
            depression_level = "severe"

        # 자해 위험 판정 (q9 기반)
        q9 = phq9_scores.get("q9", 0)
        if q9 == 0:
            self_harm_risk = "none"
        elif q9 == 1:
            self_harm_risk = "passive"
        else:
            self_harm_risk = "active"

        print(f"[NodeC] PHQ-9 총점: {phq9_total} → {depression_level}")
        print(f"[NodeC] self_harm_risk: {self_harm_risk}")

        return {**state,
                "top_emotions": top_emotions,
                "phq9_scores": phq9_scores,
                "phq9_total": phq9_total,
                "depression_level": depression_level,
                "self_harm_risk": self_harm_risk}
    except Exception as e:
        print(f"[NodeC] 오류: {e}")
        return {**state,
                "top_emotions": None,
                "phq9_scores": None,
                "phq9_total": None,
                "depression_level": None,
                "self_harm_risk": "none",
                "errors": state.get("errors", []) + [f"NodeC: {str(e)}"]}


def _simulate_phq9(text: str) -> Dict[str, int]:
    """GPT PHQ-9 proxy scoring 시뮬레이션 (규칙 기반 위험도 분류)"""
    return {
        "q1": 2 if any(k in text for k in ["재미없","흥미없","하기 싫","의욕"]) else 1,
        "q2": 2 if any(k in text for k in ["슬프","우울","절망","힘들","눈물"]) else 1,
        "q3": 2 if any(k in text for k in ["잠","수면","못 잠","새벽"]) else 0,
        "q4": 1 if any(k in text for k in ["피곤","졸","피로","지쳐"]) else 0,
        "q5": 1 if any(k in text for k in ["밥","식욕","먹기"]) else 0,
        "q6": 1 if any(k in text for k in ["내 잘못","못났","바보","자신없"]) else 0,
        "q7": 1 if any(k in text for k in ["집중","멍","산만"]) else 0,
        "q8": 0,
        "q9": 1 if any(k in text for k in ["죽고","사라지","없어지고","힘들어서 못"]) else 0,
    }


# ══════════════════════════════════════════════
# NodeD: 위험도 기반 후속 생성 분기
# ══════════════════════════════════════════════
def node_d(state: WorkflowState) -> WorkflowState:
    print(f"\n{'='*50}")
    print(f"[NodeD] 위험도 기반 후속 생성 분기")
    print(f"{'='*50}")
    try:
        self_harm_risk = state.get("self_harm_risk", "none")
        depression_level = state.get("depression_level", "normal")
        top_emotions = state.get("top_emotions", [])
        nickname = state.get("user_nickname", "친구")
        diaries_text = state["diaries_text"]

        is_risk = (
            self_harm_risk in ["passive", "active"] or
            depression_level in ["moderate", "moderately_severe", "severe"]
        )

        escalation_status = "none"
        escalation_sent_at = None

        if is_risk:
            print(f"[NodeD] 주의/고위험 경로 → risk")
            node_d_branch = "risk"
            weekly_summary = _gen_risk_summary(nickname, top_emotions, depression_level)
            advice = [
                "지금 많이 힘들지? 혼자 견디려 하지 말고 신뢰하는 어른이나 친구에게 털어놓아봐.",
                "잠깐이라도 밖에 나가 산책하거나 좋아하는 음악을 들어봐.",
                "오늘 하루를 버텨낸 것만으로도 충분히 잘한 거야.",
            ]
            resources = [
                {"name": "청소년상담1388", "phone": "1388", "desc": "24시간 무료 상담"},
                {"name": "자살예방상담전화", "phone": "1393", "desc": "24시간 위기상담"},
            ]

            # Escalation HITL: self_harm_risk=active 시 상담사·교사 알림 전달
            if self_harm_risk == "active":
                print(f"[NodeD] ⚠️  Escalation HITL 실행 → 상담사·교사 알림 전달")
                escalation_status = "sent"
                escalation_sent_at = datetime.now().isoformat()
                print(f"[NodeD] escalation_sent_at: {escalation_sent_at}")
        else:
            print(f"[NodeD] 일반 경로 → normal")
            node_d_branch = "normal"
            weekly_summary = _gen_normal_summary(nickname, top_emotions, diaries_text)
            advice = [
                "하루 10분 가벼운 산책을 꾸준히 해봐.",
                "잠들기 전 오늘 좋았던 것 한 가지만 떠올려봐.",
                "수면 시간을 일정하게 유지해봐.",
            ]
            resources = None

        return {**state,
                "node_d_branch": node_d_branch,
                "weekly_summary": weekly_summary,
                "advice": advice,
                "resources": resources,
                "escalation_status": escalation_status,
                "escalation_sent_at": escalation_sent_at}
    except Exception as e:
        print(f"[NodeD] 오류: {e}")
        return {**state,
                "node_d_branch": "normal",
                "weekly_summary": "이번 주도 수고했어.",
                "advice": [],
                "resources": None,
                "escalation_status": "none",
                "escalation_sent_at": None,
                "errors": state.get("errors", []) + [f"NodeD: {str(e)}"]}


def _gen_normal_summary(nickname: str, top_emotions: list, diaries_text: list) -> str:
    emo_kr = {
        "sad": "슬픔", "angry": "분노", "anxious": "불안",
        "happy": "기쁨", "hurt": "상처", "embarrassed": "당황"
    }
    top = top_emotions[0]["label"] if top_emotions else "sad"
    return (
        f"{nickname} 이는 이번 주에 주로 {emo_kr.get(top, '복잡한 감정')}이 많이 느껴졌던 것 같아. "
        f"여러 일들이 겹쳐서 마음이 무거웠던 것 같고, 그래도 한 주 잘 버텨냈어! "
        f"다음 주도 조금씩 나아질 거야, 응원해."
    )


def _gen_risk_summary(nickname: str, top_emotions: list, depression_level: str) -> str:
    level_kr = {
        "moderate": "중등도", "moderately_severe": "중등도-중증", "severe": "중증"
    }
    return (
        f"{nickname} 이는 이번 주에 많이 힘들었던 것 같아. "
        f"우울도가 {level_kr.get(depression_level, '높게')} 나왔어. "
        f"혹시 전문가의 도움이나 상담이 필요하다면 아래 전화번호로 전화를 걸어서 "
        f"어떤 일이 있었는지 말씀드려보는 건 어때? "
        f"{nickname}이에게 무슨 일이 있었는지 친절하게 답해줄 거야."
    )


# ══════════════════════════════════════════════
# NodeE: 저장 및 상태 기록 (checkpoint)
# ══════════════════════════════════════════════
def node_e(state: WorkflowState) -> WorkflowState:
    print(f"\n{'='*50}")
    print(f"[NodeE] 저장 및 상태 기록")
    print(f"{'='*50}")
    try:
        # 실제 구현에서는 DB INSERT
        # 데모에서는 state 반환으로 대체
        print(f"[NodeE] weekly_report 저장 완료")
        print(f"[NodeE] report_type    : {state.get('report_type')}")
        print(f"[NodeE] depression     : {state.get('depression_level')}")
        print(f"[NodeE] self_harm_risk : {state.get('self_harm_risk')}")
        print(f"[NodeE] escalation     : {state.get('escalation_status')}")
        print(f"[NodeE] user.top_emotion 재계산 → {state.get('top_emotions', [{}])[0].get('label', 'null')}")

        langgraph_state = {
            "node_a_status": state.get("node_a_status"),
            "node_b_branch": state.get("node_b_branch"),
            "node_d_branch": state.get("node_d_branch"),
            "report_type": state.get("report_type"),
            "depression_level": state.get("depression_level"),
            "self_harm_risk": state.get("self_harm_risk"),
            "escalation_status": state.get("escalation_status"),
            "errors": state.get("errors", []),
        }

        return {**state,
                "report_status": "generated",
                "node_e_status": "ok",
                "langgraph_state": langgraph_state}
    except Exception as e:
        print(f"[NodeE] 저장 실패: {e}")
        print(f"[NodeE] checkpoint 기록 → NodeE만 재실행 가능")
        return {**state,
                "report_status": "failed",
                "node_e_status": "failed",
                "errors": state.get("errors", []) + [f"NodeE: {str(e)}"]}


# ══════════════════════════════════════════════
# 조건 분기 함수 (conditional edges)
# ══════════════════════════════════════════════
def route_after_a(state: WorkflowState) -> str:
    if state.get("node_a_status") == "skip":
        print("[Router] NodeA → END (일기 0건)")
        return "end"
    return "node_b"


def route_after_b(state: WorkflowState) -> str:
    if state.get("node_b_branch") == "lite":
        print("[Router] NodeB → END (데이터 부족, insufficient_data 저장)")
        return "end"
    return "node_c"


# ══════════════════════════════════════════════
# 그래프 빌드 및 컴파일
# ══════════════════════════════════════════════
def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)
    graph.add_node("node_d", node_d)
    graph.add_node("node_e", node_e)

    graph.set_entry_point("node_a")

    # conditional edges
    graph.add_conditional_edges("node_a", route_after_a, {
        "end": END,
        "node_b": "node_b"
    })
    graph.add_conditional_edges("node_b", route_after_b, {
        "end": END,
        "node_c": "node_c"
    })

    # 일반 edges
    graph.add_edge("node_c", "node_d")
    graph.add_edge("node_d", "node_e")
    graph.add_edge("node_e", END)

    return graph.compile()


# ══════════════════════════════════════════════
# 워크플로우 실행 진입점
# ══════════════════════════════════════════════
def run_weekly_workflow(
    user_id: str,
    user_nickname: str,
    diary_count: int,
    diaries_text: list,
    emotions_data: list,
    week_start: str,
    week_end: str,
) -> dict:
    print(f"\n{'#'*60}")
    print(f"칭구칭구 LangGraph 주간 워크플로우 시작")
    print(f"user: {user_nickname} | week: {week_start} ~ {week_end}")
    print(f"{'#'*60}")

    app = build_graph()

    initial: WorkflowState = {
        "user_id": user_id,
        "user_nickname": user_nickname,
        "diary_count": diary_count,
        "diaries_text": diaries_text,
        "emotions_data": emotions_data,
        "week_start": week_start,
        "week_end": week_end,
        "total_chars": 0,
        "valid_emotion_count": 0,
        "node_a_status": "",
        "report_type": "full",
        "node_b_branch": "full",
        "top_emotions": None,
        "phq9_scores": None,
        "phq9_total": None,
        "depression_level": None,
        "self_harm_risk": "none",
        "node_d_branch": "normal",
        "weekly_summary": None,
        "advice": None,
        "resources": None,
        "report_status": "",
        "node_e_status": "",
        "diary_warning": None,
        "escalation_status": "none",
        "escalation_sent_at": None,
        "langgraph_state": None,
        "errors": [],
    }

    final = app.invoke(initial)

    print(f"\n{'#'*60}")
    print(f"워크플로우 완료 → report_status: {final.get('report_status')}")
    print(f"{'#'*60}\n")

    return final
