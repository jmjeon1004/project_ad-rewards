"""통합 데이터 초기화 — 모든 페이지에서 한 번만 로딩 후 session_state 공유."""
import streamlit as st

from src.data_loader import (
    load_hourly_report, load_ad_attr_map, load_ad_classification,
    load_ad_outcome, load_today_date,
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

    # 1) 원본 데이터 로딩 (개별 함수의 @st.cache_* 캐싱 활용)
    hourly = load_hourly_report()
    attr = load_ad_attr_map()
    classification = load_ad_classification()
    outcome = load_ad_outcome()

    # 2) 베이스 테이블 + 광고 요약
    base = build_base_table(hourly, attr, classification, outcome)
    today = load_today_date()
    ad_summary = build_ad_summary(base)

    # 3) ML 스코어링
    ad_master = load_ad_master_clean()
    sched = load_sched_clean()
    ad_outcome_full = load_ad_outcome_full()
    model_scores = score_all_ads(attr, ad_master, classification, ad_outcome_full, sched)

    # 4) ad_summary에 모델 스코어 병합 (app.py에서 하던 작업을 여기서 통합)
    ad_summary = ad_summary.merge(
        model_scores[['ads_idx', 'm1_score', 'm1_grade', 'm2_proba', 'm2_decision']],
        on='ads_idx', how='left',
    )

    # 5) session_state에 저장
    st.session_state["base"] = base
    st.session_state["today"] = today
    st.session_state["ad_summary"] = ad_summary
    st.session_state["attr"] = attr
    st.session_state["classification"] = classification
    st.session_state["ad_master"] = ad_master
    st.session_state["sched"] = sched
    st.session_state["ad_outcome_full"] = ad_outcome_full
    st.session_state["model_scores"] = model_scores
    st.session_state["_core_data_loaded"] = True
