"""광고 상세 페이지(05_광고_상세)용 배지별 데이터 준비 모듈."""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    MODEL1_ARTIFACTS, MODEL2_ARTIFACTS,
    ML_GRADE_LABELS, MEDIA_NAME_MAP,
)


# ── 모델 메타데이터 로드 ──

@st.cache_resource
def _load_model_meta(artifacts_dir: str) -> dict:
    path = os.path.join(artifacts_dir, "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def _load_threshold_info() -> dict:
    path = os.path.join(MODEL2_ARTIFACTS, "threshold_info.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_metas() -> tuple[dict, dict, dict]:
    """M1 meta, M2 meta, threshold_info 반환."""
    m1 = _load_model_meta(MODEL1_ARTIFACTS)
    m2 = _load_model_meta(MODEL2_ARTIFACTS)
    thr = _load_threshold_info()
    return m1, m2, thr


@st.cache_resource
def _load_grade_info() -> dict:
    path = os.path.join(MODEL1_ARTIFACTS, "grade_info.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_grade_info() -> dict:
    """grade_info.json 반환 (bins, labels, grade_order)."""
    return _load_grade_info()


# ── 운영 일수 ──

def get_ad_running_days(ads_idx: int, sched: pd.DataFrame) -> int | None:
    """sched에서 campaign_n_day 최대값 반환."""
    s = sched[sched["ads_idx"] == ads_idx]
    if s.empty or s["campaign_n_day"].isna().all():
        return None
    return int(s["campaign_n_day"].max())


# ── 매체확장: 매체별 CVR 비교 ──

def get_media_cvr_comparison(
    ads_idx: int,
    base: pd.DataFrame,
    ad_type: str,
    ad_category: str,
    current_media: str,
    *,
    _use_cache: bool = True,
) -> pd.DataFrame:
    """
    같은 유형·카테고리 광고의 매체별 평균 CVR 비교 DataFrame.
    columns: [media, cvr, is_current]
    """
    # session_state 캐싱 — 동일 ads_idx에서 재렌더링 시 재계산 방지
    if _use_cache:
        cache_key = f"_media_cvr_{ads_idx}"
        cached = st.session_state.get(cache_key)
        if cached is not None:
            return cached

    same = base[
        (base["analysis_ads_type_label"] == ad_type)
        & (base["category_name"] == ad_category)
    ]
    if same.empty:
        same = base[base["analysis_ads_type_label"] == ad_type]

    agg = same.groupby("final_media").agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
    ).reset_index()
    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, 0)
    agg = agg[agg["clk"] >= 10].sort_values("cvr", ascending=False).head(6)
    agg["is_current"] = agg["final_media"] == current_media
    agg = agg.rename(columns={"final_media": "media"})

    # 현재 매체가 없으면 추가
    if not agg["is_current"].any():
        ad_data = base[base["ads_idx"] == ads_idx]
        c, t = ad_data["rpt_time_clk"].sum(), ad_data["rpt_time_turn"].sum()
        cvr = t / c * 100 if c > 0 else 0
        cur = pd.DataFrame([{"media": current_media, "clk": c, "turn": t, "cvr": cvr, "is_current": True}])
        agg = pd.concat([cur, agg], ignore_index=True)

    # 현재 매체를 맨 위로
    agg = agg.sort_values(["is_current", "cvr"], ascending=[False, False])
    result = agg[["media", "cvr", "is_current"]].reset_index(drop=True)

    if _use_cache:
        st.session_state[cache_key] = result
    return result


# ── 승급추진: 일별 CVR/클릭 추이 ──

def get_daily_kpi_trend(ads_idx: int, base: pd.DataFrame) -> pd.DataFrame:
    """일별 clk, turn, cvr 추이. columns: [date, clk, turn, cvr]"""
    ad = base[base["ads_idx"] == ads_idx]
    if ad.empty:
        return pd.DataFrame(columns=["date", "clk", "turn", "cvr"])
    daily = ad.groupby("rpt_time_date").agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
    ).reset_index().rename(columns={"rpt_time_date": "date"})
    daily["cvr"] = np.where(daily["clk"] > 0, daily["turn"] / daily["clk"] * 100, 0)
    daily = daily.sort_values("date")
    return daily


# ── 즉시조치: click_day 급감 ──

def get_click_decline_data(
    ads_idx: int,
    sched: pd.DataFrame,
    base: pd.DataFrame,
    ad_type: str,
) -> pd.DataFrame:
    """
    이 광고의 D+N별 클릭을 코호트 평균 대비 비율(%)로 산출.
    Returns: DataFrame [campaign_n_day, click_cnt, cohort_avg, vs_cohort_pct]
      vs_cohort_pct: 코호트 평균 대비 이 광고 비율 (%), 100 미만이면 부진
    """
    ad = sched[(sched["ads_idx"] == ads_idx) & sched["campaign_n_day"].notna()]
    ad_clicks = ad.groupby("campaign_n_day")["click_cnt"].sum().reset_index()
    ad_clicks = ad_clicks.sort_values("campaign_n_day")

    # 코호트 평균: 같은 유형 광고의 D+N별 평균 클릭
    cohort = sched[
        (sched["analysis_ads_type_label"] == ad_type)
        & sched["campaign_n_day"].notna()
    ]
    cohort_avg = cohort.groupby("campaign_n_day")["click_cnt"].mean().reset_index()
    cohort_avg = cohort_avg.rename(columns={"click_cnt": "cohort_avg"})

    # 병합
    merged = ad_clicks.merge(cohort_avg, on="campaign_n_day", how="left")

    # D+0 ~ D+5 범위만
    max_day = min(5, merged["campaign_n_day"].max()) if not merged.empty else 5
    merged = merged[merged["campaign_n_day"] <= max_day]

    # 코호트 평균 대비 비율
    merged["vs_cohort_pct"] = np.where(
        merged["cohort_avg"] > 0,
        merged["click_cnt"] / merged["cohort_avg"] * 100,
        np.nan,
    )

    return merged


# ── 우선검토: click_day1 → day3 비율 ──

def get_click_day_ratio(ads_idx: int, sched: pd.DataFrame, *, _pre_filtered: bool = False) -> float | None:
    """D+1 클릭 대비 D+3 클릭 비율 계산 (click retention)."""
    if _pre_filtered:
        ad = sched[sched["campaign_n_day"].notna()]
    else:
        ad = sched[(sched["ads_idx"] == ads_idx) & sched["campaign_n_day"].notna()]
    if ad.empty:
        return None
    ad_clicks = ad.groupby("campaign_n_day")["click_cnt"].sum()
    day1 = ad_clicks.get(1, 0)
    day3 = ad_clicks.get(3, 0)
    if day1 <= 0:
        return None
    return day3 / day1


# ── 우선검토: threshold gauge 데이터 ──

def get_threshold_gauge_data(
    ads_idx: int,
    model_scores: pd.DataFrame,
    threshold_info: dict,
) -> dict:
    """부진 확률 위치, threshold 값, 추적 지표."""
    sc = model_scores[model_scores["ads_idx"] == ads_idx]
    proba = float(sc.iloc[0]["m2_proba"]) if not sc.empty and pd.notna(sc.iloc[0].get("m2_proba")) else 0
    best_f1 = threshold_info.get("best_f1", 0.5)
    immediate_thr = 0.7  # 즉시조치 기준
    return {
        "proba": proba,
        "best_f1": best_f1,
        "immediate_thr": immediate_thr,
    }


# ── 히어로 카드 텍스트 생성 ──

def build_hero_content(
    badge: str,
    kpis: dict,
    scores: pd.Series | None,
    ad_info: pd.Series,
    media_count: int | None = None,
    grade_info: dict | None = None,
) -> dict:
    """배지 유형에 따른 히어로 카드 콘텐츠 생성."""
    grade = scores.get("m1_grade", "-") if scores is not None else "-"
    m1_score = scores.get("m1_score", 0) if scores is not None else 0
    m2_proba = scores.get("m2_proba", 0) if scores is not None else 0
    grade_label = ML_GRADE_LABELS.get(grade, "")
    cvr = kpis.get("cvr")
    cvr_str = f"{cvr:.1f}%" if cvr is not None else "-"
    media = ad_info.get("final_media", "-")

    if badge == "매체확장":
        mc = media_count or 1
        return {
            "label": "매체 확장 기회",
            "headline": f"{media} 단독 운영 · <span style='color:#4A6BAA;'>{grade} 등급({m1_score:.0f}점)</span> 유지 — 다른 매체로 확장 시 추가 정산수익 기대",
            "explain": f"현재 클릭의 <strong>100%가 {media}에서만 발생</strong>해요. 같은 유형 광고의 평균 매체 수는 {mc}개.",
        }
    elif badge == "승급추진":
        s_cut = grade_info["bins"][4] if grade_info else 80.7
        gap = max(0, s_cut - m1_score)
        return {
            "label": "S 등급 승급 후보",
            "headline": f"{grade} 등급 <span style='color:#B57C00;'>{m1_score:.0f}점</span> — S 진입 cutpoint {s_cut:.0f}점까지 {gap:.1f}점 차이",
            "explain": "A 등급 광고 — 추가 운영·개입으로 점수가 오르면 S 진입 가능성이 있어요. 미리 인지·준비 권장.",
        }
    elif badge == "즉시조치":
        return {
            "label": "D+3 시점 부진 위험 — 즉시 조치 필요",
            "headline": f"부진 확률 <span style='color:#9A4842;'>{m2_proba:.2f}</span> — 완료 지표 심각 부진",
            "explain": f"모델이 부진 패턴을 감지했어요. <strong>즉각적인 운영 점검이 필요</strong>합니다.",
        }
    else:  # 우선검토
        return {
            "label": "D+3 시점 부진 위험 — 우선 검토",
            "headline": f"부진 확률 <span style='color:#E65100;'>{m2_proba:.2f}</span> — 즉시 조치까지는 아니지만 모니터링 권장",
            "explain": f"위험 기준선을 약간 초과한 상태. <strong>현재 운영 지속</strong>하면서 추세 추적 권장.",
        }


# ── "왜?" 설명 스텝 생성 ──

def build_why_steps(
    badge: str,
    kpis: dict,
    scores: pd.Series | None,
    ad_info: pd.Series,
    grade_info: dict | None = None,
) -> list[dict]:
    """
    "왜 X로 분류됐나?" 설명 스텝 리스트.
    각 dict: {"num": str, "title": str, "text": str, "caveat": bool}
    """
    grade = scores.get("m1_grade", "-") if scores is not None else "-"
    m1_score = scores.get("m1_score", 0) if scores is not None else 0
    m2_proba = scores.get("m2_proba", 0) if scores is not None else 0
    cvr = kpis.get("cvr")
    cvr_str = f"{cvr:.1f}%" if cvr is not None else "-"
    media = ad_info.get("final_media", "-")
    ad_type = ad_info.get("analysis_ads_type_label", "-")

    if badge == "매체확장":
        return [
            {"num": "1", "title": "광고 자체는 매우 우수해요",
             "text": f"{grade} 등급 · CVR {cvr_str}로 같은 유형 평균보다 높음", "caveat": False},
            {"num": "2", "title": "그런데 한 매체에서만 운영 중이에요",
             "text": f"현재 {media}에서만 운영 · 단일 매체는 트래픽 변동에 취약", "caveat": False},
            {"num": "!", "title": "추천 매체 수치는 같은 유형 평균이에요",
             "text": "이 광고의 실제 성과는 운영해봐야 알 수 있어요. <strong>소액 테스트 검증 후 본격 확장 권장</strong>.", "caveat": True},
        ]
    elif badge == "승급추진":
        s_cut = grade_info["bins"][4] if grade_info else 80.7
        gap = max(0, s_cut - m1_score)
        return [
            {"num": "1", "title": "A 등급 광고예요",
             "text": f"점수 {m1_score:.1f}점 · S 등급(정상 운영 중 가장 우수한 등급) 바로 아래 등급", "caveat": False},
            {"num": "2", "title": "S 진입 cutpoint까지 남은 점수",
             "text": f"{gap:.1f}점 차이 · 추가 운영·개입(매체 확장·소재 최적화 등)으로 점수가 오르면 S 진입 가능", "caveat": False},
            {"num": "!", "title": "점수 차이가 클수록 진입 가능성은 낮아요",
             "text": f"{gap:.1f}점 차이가 작을수록(모델 변동성 범위 안일수록) 실현 가능성이 높아요. <strong>S 진입을 보장하는 정보가 아니라 \"현재 위치\"라는 참고 정보</strong>로 활용 권장.", "caveat": True},
        ]
    elif badge == "즉시조치":
        return [
            {"num": "1", "title": "완료가 발생하지 않고 있어요",
             "text": f"같은 유형은 보통 클릭 대비 완료가 발생 · 광고/랜딩 정합성 문제 의심", "caveat": False},
            {"num": "2", "title": "클릭이 빠르게 줄고 있어요",
             "text": "초기 클릭 감소 패턴은 과거 부진 광고에서 자주 발견된 패턴", "caveat": False},
            {"num": "!", "title": f"'{m2_proba:.0%} 부진'이 아니라 '{m2_proba:.0%} 유사'예요",
             "text": "모델은 보조 도구 · <strong>매체 일시 변동 가능성도 있으니 최종 판단은 운영자</strong>.", "caveat": True},
        ]
    else:  # 우선검토
        return [
            {"num": "1", "title": "클릭이 줄고 CVR도 평균보다 낮아요",
             "text": "점진적 감소이지 급감은 아님", "caveat": False},
            {"num": "2", "title": "위험 기준선을 약간만 넘었어요",
             "text": f"부진 확률 {m2_proba:.2f} · threshold 대비 소폭 초과", "caveat": False},
            {"num": "!", "title": "'우선 검토'는 부진 확정이 아니에요",
             "text": "회복 가능성 충분 · <strong>3일 추적 후 악화/회복 판단 권장</strong>.", "caveat": True},
        ]


# ── KPI 비교 컨텍스트 ──

def _get_type_averages(base: pd.DataFrame) -> dict:
    """유형별 평균 KPI를 사전 집계하여 캐싱. session_state에 저장."""
    cache_key = "_detail_type_averages"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    agg = base.groupby("analysis_ads_type_label").agg(
        tc=("rpt_time_clk", "sum"),
        tt=("rpt_time_turn", "sum"),
        ta=("rpt_time_acost", "sum"),
        te=("rpt_time_earn", "sum"),
    )
    result = {}
    for ad_type, row in agg.iterrows():
        tc, tt, ta, te = row["tc"], row["tt"], row["ta"], row["te"]
        result[ad_type] = {
            "cvr": tt / tc * 100 if tc > 0 else None,
            "cpa": ta / tt if tt > 0 else None,
            "margin": (ta - te) / ta * 100 if ta > 0 else None,
        }
    st.session_state[cache_key] = result
    return result


def get_kpi_contexts(
    kpis: dict,
    base: pd.DataFrame,
    ads_idx: int,
    ad_type: str,
) -> dict[str, tuple[str, str]]:
    """
    5개 KPI의 비교 컨텍스트 (text, css_class).
    css_class: "good" | "bad" | ""
    """
    # 캐싱된 유형별 평균 사용
    type_avgs = _get_type_averages(base)
    avgs = type_avgs.get(ad_type, {})
    type_cvr = avgs.get("cvr")
    type_cpa = avgs.get("cpa")
    type_margin = avgs.get("margin")

    ctx = {}

    # CVR
    if kpis.get("cvr") is not None and type_cvr is not None:
        diff = kpis["cvr"] - type_cvr
        sign = "+" if diff >= 0 else ""
        cls = "good" if diff >= 0 else "bad"
        ctx["cvr"] = (f"{ad_type} 평균 {sign}{diff:.1f}%p", cls)
    else:
        ctx["cvr"] = ("", "")

    # CPA
    if kpis.get("cpa") is not None and type_cpa is not None:
        diff_pct = (kpis["cpa"] - type_cpa) / type_cpa * 100
        cls = "good" if diff_pct <= 0 else "bad"
        sign = "+" if diff_pct >= 0 else ""
        ctx["cpa"] = (f"유형 평균 대비 {sign}{diff_pct:.0f}%", cls)
    else:
        ctx["cpa"] = ("", "")

    # 마진율
    if kpis.get("margin_rate") is not None and type_margin is not None:
        diff = kpis["margin_rate"] - type_margin
        sign = "+" if diff >= 0 else ""
        cls = "good" if diff >= 0 else "bad"
        ctx["margin_rate"] = (f"유형 평균 {sign}{diff:.1f}%p", cls)
    else:
        ctx["margin_rate"] = ("", "")

    # 클릭수, 완료수 — 단순 표시
    ctx["clk"] = ("", "")
    ctx["turn"] = ("", "")

    return ctx
