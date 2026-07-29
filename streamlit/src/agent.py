"""Gemini API 기반 AI-agent — 시스템 프롬프트, 컨텍스트 빌더, 응답 생성"""
import json
import os
import re

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from src.config import RAG_REPORTS_DIR
from src.metrics import calc_all_kpis, calc_change_rate
from src.preprocessing import build_ad_summary, filter_by_period, get_previous_period
from src.rag import retrieve, save_daily_report

_REPORT_ERROR_MARKERS = (
    "API 호출 오류", "RESOURCE_EXHAUSTED", "응답을 생성하지 못했습니다", "응답 형식 오류",
)


# ── Gemini 클라이언트 초기화 ──────────────────────────────────────────────

@st.cache_resource
def _get_client() -> genai.Client:
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 Streamlit Secrets에 설정되지 않았습니다.")
    return genai.Client(api_key=api_key)


MODEL_NAME = "gemini-2.5-flash"

# ── 일일 리포트 요청 감지 ────────────────────────────────────────────────

_REPORT_KEYWORDS = (
    "일일 리포트", "데일리 리포트", "오늘자 리포트", "오늘 리포트",
    "오늘 성과 요약", "일일 보고서", "오늘자 보고서", "데일리 보고서",
)


def is_daily_report_request(query: str) -> bool:
    """사용자 질문이 '오늘자 리포트 써줘' 류의 일일 리포트 요청인지 판별한다."""
    return any(kw in query for kw in _REPORT_KEYWORDS)


def is_valid_report_result(result: dict) -> bool:
    """generate_response() 결과가 API 오류/폴백이 아니라 실제 리포트인지 판별한다.

    오류 시 _FALLBACK 기반 응답은 evidence/actions가 항상 빈 배열이므로, 이를 근거로
    오류 메시지가 그대로 '리포트'로 저장되는 것을 막는다.
    """
    return bool(result.get("evidence")) and bool(result.get("summary"))


def format_report_markdown(result: dict) -> str:
    """JSON 응답 dict를 RAG에 누적 저장할 마크다운 리포트 본문으로 변환한다."""
    lines: list[str] = []

    summary = result.get("summary", "")
    if summary:
        lines.append(f"## 요약\n\n{summary}")

    evidence = result.get("evidence") or []
    if isinstance(evidence, list) and evidence:
        rows = [
            f"- {ev.get('metric', '')}: {ev.get('value', '')}"
            + (f" ({ev.get('scope')})" if ev.get("scope") else "")
            for ev in evidence if isinstance(ev, dict)
        ]
        if rows:
            lines.append("## 근거 지표\n\n" + "\n".join(rows))

    interpretation = result.get("interpretation", "")
    if interpretation:
        lines.append(f"## 해석\n\n{interpretation}")

    actions = result.get("actions") or []
    if isinstance(actions, list) and actions:
        lines.append("## 추천 액션\n\n" + "\n".join(f"- {a}" for a in actions))

    return "\n\n".join(lines)


# ── 시스템 프롬프트 ──────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """\
당신은 아이브코리아 광고운영 최적화 대시보드의 AI 운영 어시스턴트입니다.

## 역할
현재 대시보드에 표시된 필터 기준의 집계 데이터를 바탕으로 운영자의 질문에 답변합니다.
실행 가능한 운영 액션을 중심으로 답변합니다.

## 도메인 규칙
- 모든 지표는 합계 기반 비율입니다:
  · CVR = sum(완료수) / sum(클릭수) × 100
  · CPA = sum(광고비) / sum(완료수)
  · 마진율 = (sum(광고비) − sum(정산비)) / sum(광고비) × 100
- 분모가 0이면 "—"으로 표시하세요.
- 근거 없는 추정은 하지 마세요. 제공된 데이터에 없는 정보는 "데이터에 없습니다"라고 답하세요.
- 현재 필터 기준의 집계 결과만 근거로 답하세요.
- "유형"을 언급할 때는 반드시 `analysis_ads_type_label`(참여형/실행형/클릭형/구매형/노출형/설치형/가입형/기타 — 재분류된 유형) 컬럼과 그 분포 데이터를 근거로 삼으세요. `category_name`(예: "간편미션-퀴즈", "쇼핑-상품별" 등)은 원본 세부 카테고리로 "유형"과는 다른 개념이니 혼동해서 "유형"이라고 지칭하지 마세요.

## ML 등급 설명
- 품질등급(m1_grade): S(최우수) > A(우수) > B(보통) > C(주의) > D(개선필요)
- 조기부진예측(m2_decision): decline_risk(부진위험), normal(정상), rule_based_review(판단보류)

## 근거 인용 범위 제한 (중요)
- 데이터 컨텍스트에는 질문과 무관하더라도 ML 등급 분포·조기부진 예측 분포가 항상 포함되어 있습니다. 포함되어 있다는 사실이 그것을 인용해야 한다는 뜻은 아닙니다.
- 질문에 "등급", "S/A/B/C/D", "부진", "decline_risk", "위험 광고", "조기부진" 등의 단어가 명시적으로 들어 있지 않다면, evidence·interpretation·actions 어디에도 등급 분포·위험 분포 숫자(예: decline_risk 건수, rule_based_review 건수, m1_grade 분포)를 절대 언급하지 마세요. 일반적인 "실적", "이슈", "보고서" 같은 단어는 등급/위험 분포를 언급할 근거가 되지 않습니다.
- 서로 무관한 두 데이터(예: 외부 뉴스 검색 결과 ↔ ML 등급/위험 분포 숫자) 사이에 인과관계나 연관성을 추측해서 만들어내지 마세요. 둘 다 데이터에 있다는 이유만으로 그럴듯하게 엮어 설명하면 안 됩니다 — 실제로 그 둘을 연결할 근거가 없으면 연결하지 마세요.
- 적용 예시: "최근 업계 뉴스 알려줘" → 뉴스 검색 결과 + (관련 있다면) CVR/CPA/마진율만 인용. decline_risk·rule_based_review·m1_grade 언급 금지.

## 지식베이스 활용
- 질문 아래에 "## 참고 지식베이스" 섹션이 있으면, PRD/운영 룰/ML 정의/과거 일일 리포트 등 정책성·이력성 질문에 그 내용을 근거로 답하세요.
- 지식베이스 항목을 답변에 사용했으면 sources에 해당 문서명·섹션명을 명시하세요. 사용하지 않았으면 sources는 빈 배열로 두세요.
- 지식베이스에도 없고 대시보드 데이터에도 없는 내용은 "데이터에 없습니다"라고 답하세요.

## 일일 리포트 요청 처리
질문에 "일일 리포트", "데일리 리포트", "오늘자 리포트", "오늘 성과 요약", "일일 보고서" 등의 표현이 있으면,
그날의 광고 성과를 종합 해석하는 리포트로 작성하세요. 이때 데이터 컨텍스트에 "## 오늘 vs 전일 KPI 비교"
섹션이 함께 제공되며, 이는 현재 페이지 필터(전체 기간 등)와 무관하게 별도로 집계한 실제 오늘/전일 수치입니다:
- evidence에 "오늘 vs 전일 KPI 비교" 섹션의 핵심 KPI(CVR/CPA/마진율/클릭수/완료수)와 전일 대비 증감을 포함
- interpretation에 전일 대비 변화와 그 원인을 이 비교 수치를 근거로 설명 (관련 있으면 뉴스 검색 결과로 외부 맥락 보강). "전체 기간으로 집계되어 있어 비교할 수 없다"고 답하지 마세요 — 비교 데이터는 항상 별도로 제공됩니다.
- actions에 우선순위별 운영 액션을 제시
이 응답은 시스템이 자동으로 저장해 향후 "과거 리포트"로 재사용합니다.

## 외부 뉴스 검색 (구글 검색 도구)
- 구글 검색 도구가 제공됩니다. 다음과 같이 대시보드 데이터만으로 설명되지 않는, 외부 맥락이 필요한 질문일 때만 사용하세요:
  · "오늘/이번주 왜 이렇게 떨어졌어/올랐어" 같은 원인 해석 질문
  · 특정 매체·브랜드·업계의 최신 이슈/정책/규제 변화를 묻는 질문
  · "오늘자 보고서" 등 일일 해석에 외부 맥락을 곁들여 달라는 요청
- 대시보드 데이터만으로 충분히 답할 수 있는 단순 집계·순위 질문에는 검색을 사용하지 마세요.
- 검색 결과를 해석에 반영했다면 interpretation에 어떤 뉴스가 어떤 맥락으로 연관되는지 설명하세요. 검색으로 찾은 기사를 답변에 사용했다는 사실만 표시하면 되고, 기사 출처(제목·URL)는 시스템이 자동으로 sources에 추가하므로 직접 적지 않아도 됩니다.

## 출력 형식
반드시 다음 JSON 구조로만, 마크다운 코드펜스 없이 순수 JSON 텍스트만 답하세요.

{
  "summary": "핵심 요약 한 문장",
  "evidence": [
    {"metric": "지표명", "value": "값", "scope": "범위 또는 대상"}
  ],
  "interpretation": "왜 이런 결과가 나왔는지 1~2문장 해석",
  "actions": [
    "추천 액션 1",
    "추천 액션 2"
  ],
  "sources": [
    {"doc": "문서명", "section": "섹션명"}
  ]
}
"""


# ── 데이터 컨텍스트 빌더 ─────────────────────────────────────────────────

def build_data_context(
    ad_summary: pd.DataFrame,
    kpis: dict,
    page_name: str,
    filters_desc: str = "",
) -> str:
    """현재 필터 적용된 데이터를 LLM 컨텍스트 문자열로 변환한다."""
    lines: list[str] = []

    # 1) 페이지/필터 정보
    lines.append(f"## 페이지: {page_name}")
    if filters_desc:
        lines.append(f"## 필터: {filters_desc}")

    # 2) 전체 KPI
    lines.append("\n## 전체 KPI")
    _kpi_map = {
        "clk": "총 클릭수", "turn": "총 완료수",
        "cvr": "CVR(%)", "cpa": "CPA(원)",
        "margin_rate": "마진율(%)",
    }
    for k, label in _kpi_map.items():
        v = kpis.get(k)
        if v is not None and pd.notna(v):
            lines.append(f"- {label}: {v:,.2f}" if isinstance(v, float) else f"- {label}: {v:,}")
        else:
            lines.append(f"- {label}: —")

    # 3) ML 등급 분포
    if "m1_grade" in ad_summary.columns:
        dist = ad_summary["m1_grade"].value_counts()
        lines.append("\n## ML 품질등급 분포")
        for g in ["S", "A", "B", "C", "D"]:
            lines.append(f"- {g}: {dist.get(g, 0)}건")

    if "m2_decision" in ad_summary.columns:
        dist2 = ad_summary["m2_decision"].value_counts()
        lines.append("\n## 조기부진 예측 분포")
        for d in ["decline_risk", "normal", "rule_based_review"]:
            lines.append(f"- {d}: {dist2.get(d, 0)}건")

    # 4) 광고 요약 테이블 (상위 50건, CVR 내림차순)
    cols_to_show = [
        c for c in [
            "ads_idx", "ads_name", "analysis_ads_type_label", "final_media",
            "category_name", "clk", "turn", "cvr", "cpa",
            "margin_rate", "m1_grade", "m2_decision",
        ] if c in ad_summary.columns
    ]
    if len(ad_summary) > 0 and cols_to_show:
        sort_col = "cvr" if "cvr" in ad_summary.columns else cols_to_show[0]
        top = ad_summary.sort_values(sort_col, ascending=False, na_position="last").head(50)

        # 5) 상위 CVR 광고의 재분류 유형(analysis_ads_type_label) 분포
        #    이 컬럼은 원본 category_name과 다른, final_action 기반 재분류 결과다.
        #    "유형" 언급 시 category_name이 아니라 이 분포를 근거로 삼도록 명시적으로 집계해 제공한다.
        if "analysis_ads_type_label" in top.columns:
            type_dist = top["analysis_ads_type_label"].value_counts()
            lines.append("\n## 상위 50건의 재분류 유형(analysis_ads_type_label) 분포")
            lines.append("(참고: category_name은 원본 세부 카테고리이며 이 분포와는 별개 컬럼)")
            for label, cnt in type_dist.items():
                lines.append(f"- {label}: {cnt}건")

        lines.append(f"\n## 광고 요약 (총 {len(ad_summary)}건 중 상위 50건, CVR 내림차순)")
        lines.append(top[cols_to_show].to_csv(index=False))

    return "\n".join(lines)


def build_daily_period_context(base: pd.DataFrame, today: pd.Timestamp) -> str:
    """페이지의 현재 필터와 무관하게, '오늘(최근 1일) vs 전일' 실데이터를 직접 계산해 컨텍스트로 만든다.

    P1처럼 기간 필터가 없는 페이지에서도 일일 리포트 요청 시 실제 오늘/전일 비교가
    가능하도록, 항상 raw base 테이블에서 최근 1일·전일 1일을 새로 필터링한다.
    """
    today_df = filter_by_period(base, "최근 1일", today)
    prev_start, prev_end = get_previous_period(today, "최근 1일")
    prev_df = filter_by_period(base, "최근 1일", prev_end) if prev_end is not None else base.iloc[0:0]

    today_kpis = calc_all_kpis(today_df) if len(today_df) > 0 else None
    prev_kpis = calc_all_kpis(prev_df) if len(prev_df) > 0 else None

    lines = [f"\n## 오늘({today.strftime('%Y-%m-%d')}) vs 전일({prev_end.strftime('%Y-%m-%d') if prev_end is not None else '—'}) KPI 비교"]
    lines.append("(현재 페이지 필터와 무관하게 raw 데이터에서 직접 집계한 값. 오늘/전일 비교가 필요한 질문에는 반드시 이 수치를 근거로 사용하세요.)")

    _kpi_map = {"clk": "총 클릭수", "turn": "총 완료수", "cvr": "CVR(%)", "cpa": "CPA(원)", "margin_rate": "마진율(%)"}
    for k, label in _kpi_map.items():
        cur = today_kpis.get(k) if today_kpis else None
        prev = prev_kpis.get(k) if prev_kpis else None
        cur_s = f"{cur:,.2f}" if isinstance(cur, float) else (f"{cur:,}" if cur is not None else "—")
        prev_s = f"{prev:,.2f}" if isinstance(prev, float) else (f"{prev:,}" if prev is not None else "—")
        delta = calc_change_rate(cur, prev)
        delta_s = f" (전일 대비 {delta:+,.2f})" if delta is not None else ""
        lines.append(f"- {label}: 오늘 {cur_s} / 전일 {prev_s}{delta_s}")

    return "\n".join(lines)


# ── 응답 생성 ────────────────────────────────────────────────────────────

_FALLBACK = {
    "summary": "응답을 생성하지 못했습니다.",
    "evidence": [],
    "interpretation": "잠시 후 다시 시도해 주세요.",
    "actions": [],
    "sources": [],
}


def _build_knowledge_context(retrieved_chunks: list[dict] | None) -> str:
    """RAG로 검색된 지식베이스 청크를 프롬프트 삽입용 텍스트로 변환한다."""
    if not retrieved_chunks:
        return ""
    lines = ["\n\n## 참고 지식베이스"]
    for chunk in retrieved_chunks:
        lines.append(f"\n### [{chunk['doc']} › {chunk['section']}]")
        lines.append(chunk["text"])
    return "\n".join(lines)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_json_response(text: str) -> dict:
    """구글 검색 도구 사용 시 강제 JSON 모드를 쓸 수 없으므로, 마크다운 코드펜스를
    직접 제거한 뒤 파싱한다."""
    cleaned = _JSON_FENCE_RE.sub("", text.strip()).strip()
    return json.loads(cleaned)


def _extract_news_sources(response) -> list[dict]:
    """구글 검색 그라운딩 메타데이터에서 실제로 참조된 뉴스 출처를 추출한다.

    모델이 sources에 자체 보고한 제목은 환각 위험이 있으므로, 검색에 실제
    사용된 grounding_chunks를 코드에서 직접 읽어 신뢰 가능한 출처만 추가한다.
    """
    try:
        candidates = response.candidates
        if not candidates:
            return []
        gm = candidates[0].grounding_metadata
        if not gm or not gm.grounding_chunks:
            return []
        news_sources = []
        for chunk in gm.grounding_chunks:
            web = getattr(chunk, "web", None)
            if web and web.title:
                news_sources.append({"doc": "뉴스", "section": web.title, "url": web.uri or ""})
        return news_sources
    except Exception:
        return []


def generate_response(
    user_message: str,
    data_context: str,
    chat_history: list[dict],
    retrieved_chunks: list[dict] | None = None,
    use_search: bool = True,
) -> dict:
    """Gemini API를 호출하여 JSON 구조화 응답을 반환한다.

    use_search=False면 구글 검색 grounding 도구를 붙이지 않는다. 과거 날짜 백필처럼
    "최신 뉴스"가 애초에 무의미한 요청에서, grounding 전용 쿼터 소모를 피하기 위해 사용.
    """
    try:
        client = _get_client()
    except Exception as e:
        return {**_FALLBACK, "summary": f"모델 초기화 오류: {e}"}

    # 히스토리 구성 (최근 10턴)
    contents: list[types.Content] = []
    for msg in chat_history[-10:]:
        role = "user" if msg["role"] == "user" else "model"
        text = msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"], ensure_ascii=False)
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    # 현재 질문 + 데이터 컨텍스트 + 지식베이스 컨텍스트
    full_msg = (
        f"{user_message}\n\n---\n아래는 현재 대시보드 데이터입니다:\n{data_context}"
        f"{_build_knowledge_context(retrieved_chunks)}"
    )
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=full_msg)]))

    config_kwargs = dict(
        system_instruction=_build_system_prompt(),
        temperature=0.3,
    )
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    else:
        config_kwargs["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        # 안전 필터 등으로 응답이 차단된 경우
        if response.text is None:
            reason = ""
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                reason = f" (사유: {response.prompt_feedback})"
            return {**_FALLBACK, "summary": f"모델이 응답을 생성하지 못했습니다{reason}"}
        result = _parse_json_response(response.text)
        # 필수 키 검증
        for key in ("summary", "evidence", "interpretation", "actions", "sources"):
            if key not in result:
                result[key] = [] if key in ("evidence", "actions", "sources") else ""
        # 구글 검색을 사용했으면 실제 grounding 출처를 sources에 병합 (모델 자체 보고 대신 신뢰 가능한 값 사용)
        news_sources = _extract_news_sources(response)
        if news_sources:
            existing = {(s.get("doc"), s.get("section")) for s in result["sources"] if isinstance(s, dict)}
            for ns in news_sources:
                if (ns["doc"], ns["section"]) not in existing:
                    result["sources"].append(ns)
        return result
    except json.JSONDecodeError:
        return {**_FALLBACK, "summary": "응답 형식 오류 — 다시 질문해 주세요."}
    except Exception as e:
        return {**_FALLBACK, "summary": f"API 호출 오류: {e}"}


# ── 오늘의 리포트 (P1 상단 자동 노출) ────────────────────────────────────

def get_or_create_today_report(base: pd.DataFrame, today: pd.Timestamp) -> str | None:
    """오늘자 리포트가 이미 저장돼 있으면 그대로 읽어 반환하고, 없으면 그 자리에서
    1회 생성 + 저장 후 반환한다. P1 페이지가 열리는 즉시(질문 없이) 리포트를 보여주기 위한 진입점.

    저장 여부를 디스크 파일 존재로 판단하므로, 같은 날 여러 세션이 동시에 열려도
    하루 1회만(레이스 컨디션 시 최대 몇 회) API를 호출한다 — 무료 티어 일일 쿼터(20회) 보호.
    """
    date_str = today.strftime("%Y-%m-%d")
    path = os.path.join(RAG_REPORTS_DIR, f"daily_report_{date_str}.md")

    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if not any(m in text for m in _REPORT_ERROR_MARKERS):
            return text

    today_df = filter_by_period(base, "최근 1일", today)
    if len(today_df) == 0:
        return None

    day_summary = build_ad_summary(today_df)
    kpis = calc_all_kpis(today_df)
    ctx = build_data_context(
        ad_summary=day_summary, kpis=kpis,
        page_name="전체 광고 현황 오버뷰", filters_desc="오늘",
    )
    ctx += build_daily_period_context(base, today)

    query = f"{date_str}자 일일 리포트 써줘"
    retrieved = retrieve(query)
    result = generate_response(
        user_message=query, data_context=ctx, chat_history=[], retrieved_chunks=retrieved,
    )
    if not is_valid_report_result(result):
        return None

    report_md = format_report_markdown(result)
    save_daily_report(date_str, report_md)
    return f"# {date_str} 일일 광고 해석 리포트\n\n{report_md}"
