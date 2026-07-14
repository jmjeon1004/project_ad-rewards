"""ML 인사이트 섹션용 데이터 준비 모듈."""
import os
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

from src.config import MODEL1_ARTIFACTS, MODEL2_ARTIFACTS

# 프로세스 전역 캐시 — src.metrics._GLOBAL_CACHE와 동일한 목적 (세션 무관 공유)
_INSIGHT_CACHE: dict[str, dict] = {}
_INSIGHT_CACHE_LOCK = threading.Lock()


# ── 모델 메타데이터 로드 (캐싱) ──

@st.cache_resource
def _load_model_meta(artifacts_dir: str) -> dict:
    """모델 metadata.json 로드."""
    path = os.path.join(artifacts_dir, "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def _load_threshold_info() -> dict:
    """Model2 threshold_info.json 로드."""
    path = os.path.join(MODEL2_ARTIFACTS, "threshold_info.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Threshold 재적용 ──

def reapply_threshold(scores_df: pd.DataFrame, threshold_value: float) -> pd.DataFrame:
    """
    캐시된 m2_proba에 새 threshold를 적용하여 m2_decision 재계산.
    재추론 불필요 — 즉시 반환.
    """
    df = scores_df.copy()
    df["m2_decision"] = np.where(
        df["m2_proba"].isna(),
        "rule_based_review",
        np.where(df["m2_proba"] >= threshold_value, "decline_risk", "normal"),
    )
    return df


# ── 기회 광고 배지 로직 ──

def _assign_opportunity_badge(row: pd.Series) -> str:
    """S/A 등급 광고에 기회 배지 부여 (우선순위: 승급추진 > 매체확장)."""
    grade = row.get("m1_grade", "")

    # 승급추진: A등급 + 75점 이상 (S등급 근접)
    score = row.get("m1_score", 0)
    if grade == "A" and pd.notna(score) and score >= 75:
        return "승급추진"

    # 매체확장: S등급 또는 나머지 A등급
    return "매체확장"


# ── 위험 광고 배지 로직 ──

def _assign_risk_badge(row: pd.Series) -> str:
    """부진위험 광고에 액션 배지 부여."""
    proba = row.get("m2_proba", 0)
    if pd.notna(proba) and proba >= 0.7:
        return "즉시조치"
    return "우선검토"


# ── 캐싱 래퍼 ──

def get_cached_ml_insight_data(page_key: str, fingerprint: str, **kwargs) -> dict:
    """
    프로세스 전역 메모이제이션 래퍼.
    동일 fingerprint이면 (다른 세션이 계산해둔 결과라도) 캐시 반환, 달라지면 재계산.
    원본 데이터는 앱 수명 동안 불변이므로 세션 간 공유해도 안전하다.
    """
    cache_key = f"{page_key}|{fingerprint}"
    cached = _INSIGHT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = prepare_ml_insight_data(**kwargs)
    with _INSIGHT_CACHE_LOCK:
        _INSIGHT_CACHE[cache_key] = result
    return result


# ── 메인 데이터 준비 ──

def prepare_ml_insight_data(
    filtered: pd.DataFrame,
    model_scores: pd.DataFrame,
    ad_master: pd.DataFrame,
    sched: pd.DataFrame,
    attr: pd.DataFrame,
    classification: pd.DataFrame,
    page_name: str,
    threshold_key: str = "best_f1",
    opp_top_n: int = 4,
    risk_top_n: int = 4,
    sched_agg_precomp: pd.DataFrame | None = None,
    early_click_precomp: pd.DataFrame | None = None,
    master_cols_precomp: pd.DataFrame | None = None,
) -> dict:
    """
    ML 인사이트 섹션에 필요한 모든 데이터를 준비.

    Parameters
    ----------
    filtered : 현재 페이지 필터가 적용된 base table
    model_scores : score_all_ads() 결과 (전체)
    ad_master : 광고 마스터 (mentioned_media_cnt 등)
    sched : 스케줄 데이터 (campaign_n_day, click_cnt 등)
    attr : 광고 속성 (ads_reward_price 등)
    classification : 매체/액션 분류
    page_name : 현재 페이지명
    threshold_key : M2 threshold 키 ("best_f1", "default_05", etc.)
    """
    # 모델 메타 & threshold
    m1_meta = _load_model_meta(MODEL1_ARTIFACTS)
    m2_meta = _load_model_meta(MODEL2_ARTIFACTS)
    threshold_info = _load_threshold_info()

    # threshold 값 결정
    if threshold_key in threshold_info:
        thr_value = threshold_info[threshold_key]
    elif threshold_key in threshold_info.get("recall_targets", {}):
        thr_value = threshold_info["recall_targets"][threshold_key]
    else:
        thr_value = threshold_info["best_f1"]

    # threshold 재적용
    scores = reapply_threshold(model_scores, thr_value)

    # 스코핑: filtered에 있는 광고만
    filtered_ads_idx = filtered["ads_idx"].unique()
    scoped = scores[scores["ads_idx"].isin(filtered_ads_idx)].copy()
    active_count = len(scoped)

    # ── Tab 1: 품질 등급 ──
    grade_order = ["S", "A", "B", "C", "D"]
    grade_counts = {g: int((scoped["m1_grade"] == g).sum()) for g in grade_order}
    grade_total = sum(grade_counts.values())
    avg_score = float(scoped["m1_score"].mean()) if len(scoped) > 0 else 0
    sa_count = grade_counts["S"] + grade_counts["A"]
    sa_pct = sa_count / grade_total * 100 if grade_total > 0 else 0

    # 기회 광고 enrichment (S+A)
    sa_scores = scoped[scoped["m1_grade"].isin(["S", "A"])].copy()

    # base에서 광고별 집계 지표 계산
    ad_agg = filtered.groupby("ads_idx").agg(
        ads_name=("ads_name", "first"),
        analysis_ads_type_label=("analysis_ads_type_label", "first"),
        final_media=("final_media", "first"),
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        acost=("rpt_time_acost", "sum"),
        earn=("rpt_time_earn", "sum"),
    ).reset_index()
    ad_agg["cvr"] = np.where(ad_agg["clk"] > 0, ad_agg["turn"] / ad_agg["clk"] * 100, np.nan)
    ad_agg["cpa"] = np.where(ad_agg["turn"] > 0, ad_agg["acost"] / ad_agg["turn"], np.nan)

    # 사전 집계 값 사용 (없으면 즉석 계산)
    if master_cols_precomp is not None:
        master_cols = master_cols_precomp
    else:
        master_cols = ad_master[["ads_idx", "mentioned_media_cnt", "ads_reward_price"]].drop_duplicates("ads_idx")

    if sched_agg_precomp is not None:
        sched_agg = sched_agg_precomp
    else:
        sched_agg = sched.groupby("ads_idx").agg(
            campaign_n_day_max=("campaign_n_day", "max"),
            total_clicks_sched=("click_cnt", "sum"),
        ).reset_index()

    if early_click_precomp is not None:
        early = early_click_precomp
    else:
        early = sched[sched["campaign_n_day"] < 3].groupby("ads_idx")["click_cnt"].sum().reset_index()
        early.columns = ["ads_idx", "early_click"]

    # S/A 기회 광고 조립
    opp = sa_scores.merge(ad_agg, on="ads_idx", how="left")
    opp = opp.merge(master_cols, on="ads_idx", how="left")
    opp = opp.merge(sched_agg, on="ads_idx", how="left")

    # 배지 부여 (벡터화)
    opp["opp_badge"] = np.where(
        (opp["m1_grade"] == "A") & (opp["m1_score"] >= 75),
        "승급추진",
        "매체확장",
    )

    # 카운트
    media_count = int((opp["opp_badge"] == "매체확장").sum())
    trend_count = int((opp["opp_badge"] == "승급추진").sum())

    # 배지별 상위 opp_top_n건씩 추천 (매체확장 / 승급추진)
    badge_order = ["매체확장", "승급추진"]
    opp_top3_list = []
    for _badge in badge_order:
        subset = opp[opp["opp_badge"] == _badge].sort_values("m1_score", ascending=False).head(opp_top_n)
        opp_top3_list.extend(row for _, row in subset.iterrows())
    opp_top3 = pd.DataFrame(opp_top3_list)
    opportunity_ads = []
    for _, r in opp_top3.iterrows():
        d_plus = ""
        if pd.notna(r.get("campaign_n_day_max")):
            d_plus = f"D+{int(r['campaign_n_day_max'])}"
        opportunity_ads.append({
            "ads_idx": int(r["ads_idx"]),
            "ads_name": r.get("ads_name", ""),
            "m1_grade": r.get("m1_grade", ""),
            "m1_score": round(float(r.get("m1_score", 0)), 1),
            "badge": r.get("opp_badge", "승급추진"),
            "type_label": r.get("analysis_ads_type_label", ""),
            "media": r.get("final_media", ""),
            "cvr": round(float(r["cvr"]), 1) if pd.notna(r.get("cvr")) else None,
            "reward_price": int(r["ads_reward_price"]) if pd.notna(r.get("ads_reward_price")) else None,
            "d_plus": d_plus,
            "clk": int(r["clk"]) if pd.notna(r.get("clk")) else 0,
            "turn": int(r["turn"]) if pd.notna(r.get("turn")) else 0,
        })

    # ── Tab 2: 조기부진 ──
    risk_counts = {
        "decline_risk": int((scoped["m2_decision"] == "decline_risk").sum()),
        "normal": int((scoped["m2_decision"] == "normal").sum()),
        "rule_based_review": int((scoped["m2_decision"] == "rule_based_review").sum()),
    }
    risk_total = sum(risk_counts.values())
    risk_pct = risk_counts["decline_risk"] / risk_total * 100 if risk_total > 0 else 0
    ml_applicable = risk_counts["decline_risk"] + risk_counts["normal"]
    rule_based = risk_counts["rule_based_review"]

    # 전체 평균 위험 비중 (필터 무관)
    all_decline = int((scores["m2_decision"] == "decline_risk").sum())
    all_total = len(scores)
    overall_avg_risk_pct = all_decline / all_total * 100 if all_total > 0 else 0

    # 위험 광고 enrichment
    risk_scores = scoped[scoped["m2_decision"] == "decline_risk"].copy()
    risk_enriched = risk_scores.merge(ad_agg, on="ads_idx", how="left")
    risk_enriched = risk_enriched.merge(master_cols, on="ads_idx", how="left")
    risk_enriched = risk_enriched.merge(early, on="ads_idx", how="left")
    risk_enriched = risk_enriched.merge(sched_agg, on="ads_idx", how="left")
    risk_enriched["risk_badge"] = np.where(
        risk_enriched["m2_proba"].notna() & (risk_enriched["m2_proba"] >= 0.7),
        "즉시조치", "우선검토",
    )
    risk_immediate_count = int((risk_enriched["risk_badge"] == "즉시조치").sum())
    risk_review_count = int((risk_enriched["risk_badge"] == "우선검토").sum())

    # 배지별 상위 risk_top_n건씩 (각각 m2_proba 내림차순)
    _immediate = risk_enriched[risk_enriched["risk_badge"] == "즉시조치"].sort_values("m2_proba", ascending=False).head(risk_top_n)
    _review = risk_enriched[risk_enriched["risk_badge"] == "우선검토"].sort_values("m2_proba", ascending=False).head(risk_top_n)
    risk_top3 = pd.concat([_immediate, _review]).sort_values("m2_proba", ascending=False)
    risk_ads = []
    for _, r in risk_top3.iterrows():
        d_plus = ""
        if pd.notna(r.get("campaign_n_day_max")):
            d_plus = f"D+{int(r['campaign_n_day_max'])}"
        risk_ads.append({
            "ads_idx": int(r["ads_idx"]),
            "ads_name": r.get("ads_name", ""),
            "m2_proba": round(float(r.get("m2_proba", 0)), 2),
            "badge": r.get("risk_badge", "우선검토"),
            "type_label": r.get("analysis_ads_type_label", ""),
            "media": r.get("final_media", ""),
            "cvr": round(float(r["cvr"]), 1) if pd.notna(r.get("cvr")) else None,
            "early_click": int(r["early_click"]) if pd.notna(r.get("early_click")) else None,
            "reward_price": int(r["ads_reward_price"]) if pd.notna(r.get("ads_reward_price")) else None,
            "d_plus": d_plus,
            "turn": int(r["turn"]) if pd.notna(r.get("turn")) else 0,
            "clk": int(r["clk"]) if pd.notna(r.get("clk")) else 0,
        })

    # 추론 시각 계산
    m1_created = m1_meta.get("created_at", "")
    m2_created = m2_meta.get("created_at", "")

    return {
        "page_name": page_name,
        "active_count": active_count,
        # Tab 1
        "grade_counts": grade_counts,
        "grade_total": grade_total,
        "avg_score": round(avg_score, 1),
        "sa_pct": round(sa_pct, 1),
        "opportunity_count": sa_count,
        "single_media_count": media_count,
        "trend_up_count": trend_count,
        "opportunity_ads": opportunity_ads,
        # Tab 2
        "risk_counts": risk_counts,
        "risk_total": risk_total,
        "risk_pct": round(risk_pct, 1),
        "ml_applicable_count": ml_applicable,
        "rule_based_count": rule_based,
        "overall_avg_risk_pct": round(overall_avg_risk_pct, 1),
        "risk_immediate_count": risk_immediate_count,
        "risk_review_count": risk_review_count,
        "risk_ads": risk_ads,
        # Meta
        "m1_meta": {
            "algorithm": m1_meta.get("algorithm", "XGBClassifier"),
            "test_auc": round(m1_meta.get("test_auc", 0), 3),
            "created_at": m1_created,
        },
        "m2_meta": {
            "algorithm": m2_meta.get("algorithm", "LightGBM"),
            "test_prauc": round(m2_meta.get("test_prauc", 0), 3),
            "created_at": m2_created,
        },
        "threshold_info": threshold_info,
        "current_threshold": threshold_key,
    }
