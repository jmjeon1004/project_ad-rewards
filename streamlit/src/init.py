"""통합 데이터 초기화 — 모든 페이지에서 한 번만 로딩 후 session_state 공유."""
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

from src.data_loader import (
    load_hourly_report, load_ad_attr_map, load_ad_classification,
    load_today_date,
    load_ad_master_clean, load_sched_clean, load_ad_outcome_full,
)
from src.preprocessing import build_base_table, build_ad_summary
from src.model import score_all_ads


def ensure_data_loaded():
    """
    핵심 데이터를 한 번만 로딩하여 st.session_state에 저장.
    이후 페이지 전환 시 캐시 해시 검증 없이 즉시 반환.
    """
    if st.session_state.get("_core_data_loaded"):
        return

    # 1) 7개 parquet 파일 병렬 로딩 (각 @st.cache_resource가 독립 락을 가짐)
    with ThreadPoolExecutor(max_workers=7) as exe:
        f_hourly  = exe.submit(load_hourly_report)
        f_attr    = exe.submit(load_ad_attr_map)
        f_class   = exe.submit(load_ad_classification)
        f_outcome = exe.submit(load_ad_outcome_full)
        f_today   = exe.submit(load_today_date)
        f_master  = exe.submit(load_ad_master_clean)
        f_sched   = exe.submit(load_sched_clean)

    hourly         = f_hourly.result()
    attr           = f_attr.result()
    classification = f_class.result()
    ad_outcome_full = f_outcome.result()
    today          = f_today.result()
    ad_master      = f_master.result()
    sched          = f_sched.result()

    # ad_outcome 슬라이스 (base table용) — 별도 parquet 읽기 불필요
    outcome = ad_outcome_full[["ads_idx", "avg_ctit"]]

    # 2) 베이스 테이블 + 광고 요약
    base = build_base_table(hourly, attr, classification, outcome)
    ad_summary = build_ad_summary(base)

    # 3) ML 스코어링
    model_scores = score_all_ads(attr, ad_master, classification, ad_outcome_full, sched)

    # 4) ad_summary에 모델 스코어 병합
    ad_summary = ad_summary.merge(
        model_scores[['ads_idx', 'm1_score', 'm1_grade', 'm2_proba', 'm2_decision']],
        on='ads_idx', how='left',
    )

    # 5) ML insight용 사전 집계 (필터 무관 — 전 페이지 공통)
    sched_agg = sched.groupby("ads_idx").agg(
        campaign_n_day_max=("campaign_n_day", "max"),
        total_clicks_sched=("click_cnt", "sum"),
    ).reset_index()

    early_click_df = (
        sched[sched["campaign_n_day"] < 3]
        .groupby("ads_idx")["click_cnt"]
        .sum()
        .reset_index()
        .rename(columns={"click_cnt": "early_click"})
    )

    master_cols = (
        ad_master[["ads_idx", "mentioned_media_cnt", "ads_reward_price"]]
        .drop_duplicates("ads_idx")
    )

    # 6) session_state에 저장
    st.session_state["base"]           = base
    st.session_state["today"]          = today
    st.session_state["ad_summary"]     = ad_summary
    st.session_state["attr"]           = attr
    st.session_state["classification"] = classification
    st.session_state["ad_master"]      = ad_master
    st.session_state["sched"]          = sched
    st.session_state["ad_outcome_full"] = ad_outcome_full
    st.session_state["model_scores"]   = model_scores
    st.session_state["sched_agg"]      = sched_agg
    st.session_state["early_click_df"] = early_click_df
    st.session_state["master_cols"]    = master_cols
    st.session_state["_core_data_loaded"] = True

    # 7) P1 ML insight 캐시 선점 (첫 페이지 진입 즉시 렌더링)
    from src.ml_insight_data import get_cached_ml_insight_data
    get_cached_ml_insight_data(
        page_key="p1",
        fingerprint="p1|all|best_f1",
        filtered=base,
        model_scores=model_scores,
        ad_master=ad_master,
        sched=sched,
        attr=attr,
        classification=classification,
        page_name="전체 오버뷰",
        threshold_key="best_f1",
        sched_agg_precomp=sched_agg,
        early_click_precomp=early_click_df,
        master_cols_precomp=master_cols,
    )
