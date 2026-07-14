"""Parquet 데이터 로딩 + Streamlit 캐싱"""
import os
import datetime
import pandas as pd
import streamlit as st
from src.config import DATA_DIR, ANALYSIS_DATE_START, ANALYSIS_DATE_END

# PyArrow filters용 date 객체 (date32[day] 컬럼과 타입 일치)
_DATE_START = pd.Timestamp(ANALYSIS_DATE_START).date()
_DATE_END = pd.Timestamp(ANALYSIS_DATE_END).date()


@st.cache_resource(show_spinner="데이터 로딩 중...")
def load_hourly_report() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "hourly_report.parquet")
    df = pd.read_parquet(
        path,
        filters=[
            ("rpt_time_date", ">=", _DATE_START),
            ("rpt_time_date", "<=", _DATE_END),
        ],
    )
    df["rpt_time_date"] = pd.to_datetime(df["rpt_time_date"])
    return df


@st.cache_resource(show_spinner="광고 속성 로딩 중...")
def load_ad_attr_map() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ad_attr_map.parquet")
    df = pd.read_parquet(path)
    # category 타입을 string으로 변환
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    return df


@st.cache_resource(show_spinner="매체 분류 로딩 중...")
def load_ad_classification() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ive_ad_classification.parquet")
    df = pd.read_parquet(path)
    return df[["ads_idx", "final_media", "final_action"]].copy()


@st.cache_resource(show_spinner="완료 결과 로딩 중...")
def load_ad_outcome() -> pd.DataFrame:
    return load_ad_outcome_full()[["ads_idx", "avg_ctit"]].copy()


@st.cache_resource(show_spinner="퍼널 데이터 로딩 중...")
def load_main_funnel() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "main_funnel.parquet")
    df = pd.read_parquet(path)
    return df


def get_today_date(funnel_df: pd.DataFrame) -> pd.Timestamp:
    """main_funnel의 click_date 최댓값을 '오늘'로 간주 (시간 성분 제거)"""
    return pd.Timestamp(funnel_df["click_date"].max()).normalize()


@st.cache_resource(show_spinner="기준 날짜 계산 중...")
def load_today_date() -> pd.Timestamp:
    """main_funnel에서 click_date 컬럼만 읽어 '오늘' 날짜 반환 (메모리 절약)"""
    path = os.path.join(DATA_DIR, "main_funnel.parquet")
    df = pd.read_parquet(path, columns=["click_date"])
    return pd.Timestamp(df["click_date"].max()).normalize()


# ── ML 모델용 추가 로더 ──

@st.cache_resource(show_spinner="광고 마스터 로딩 중...")
def load_ad_master_clean() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ad_master_clean.parquet")
    cols = [
        "ads_idx", "regdate", "ads_os_type", "ads_require_adid",
        "action_target_cnt", "mentioned_media_cnt", "target_media_cnt",
        "ads_summary", "ads_reward_price",
    ]
    df = pd.read_parquet(path, columns=cols)
    return df


@st.cache_resource(show_spinner="스케줄 데이터 로딩 중...")
def load_sched_clean() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "sched_clean.parquet")
    df = pd.read_parquet(path)
    return df


@st.cache_resource(show_spinner="완료 결과(전체) 로딩 중...")
def load_ad_outcome_full() -> pd.DataFrame:
    """ad_outcome 전체 컬럼 (모델 피처용)"""
    path = os.path.join(DATA_DIR, "ad_outcome.parquet")
    df = pd.read_parquet(path)
    return df
