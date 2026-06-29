"""JOIN, 날짜/시간 변환, base table 생성"""
import pandas as pd
import numpy as np
import streamlit as st
from src.config import ACTION_TYPE_MAP, MEDIA_NAME_MAP


@st.cache_resource(show_spinner="기본 테이블 생성 중...")
def build_base_table(
    _hourly: pd.DataFrame,
    _attr: pd.DataFrame,
    _classification: pd.DataFrame,
    _outcome: pd.DataFrame,
) -> pd.DataFrame:
    """hourly_report를 중심으로 광고 속성, 매체 분류, CTIT를 JOIN"""
    hourly, attr, classification, outcome = _hourly, _attr, _classification, _outcome
    # 속성에서 필요한 컬럼만 선택
    attr_cols = [
        "ads_idx", "ads_name", "ads_type_label",
        "category_name",
        "ads_reward_price", "reward_band", "ads_rejoin_type",
        "ads_save_way", "ads_day_cap", "ads_sdate", "ads_edate",
    ]
    attr_sub = attr[attr_cols].drop_duplicates(subset=["ads_idx"])

    # JOIN
    df = hourly.merge(attr_sub, on="ads_idx", how="left")
    df = df.merge(classification, on="ads_idx", how="left")
    df = df.merge(outcome, on="ads_idx", how="left")

    # 요일 추가
    df["weekday"] = df["rpt_time_date"].dt.dayofweek  # 0=Mon
    df["weekday_name"] = df["rpt_time_date"].dt.day_name()
    weekday_map = {
        "Monday": "월", "Tuesday": "화", "Wednesday": "수",
        "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"
    }
    df["weekday_kr"] = df["weekday_name"].map(weekday_map)
    df["is_weekend"] = df["weekday"].isin([5, 6])

    # final_action → analysis_ads_type_label (재분류된 광고 유형)
    df["analysis_ads_type_label"] = df["final_action"].map(ACTION_TYPE_MAP)

    # final_media → 한글 매체명
    df["final_media"] = df["final_media"].map(MEDIA_NAME_MAP).fillna(df["final_media"])

    # category_name 정리: "선택안함" → "기타"
    df["category_name"] = df["category_name"].replace({"선택안함": "기타"})

    # 결측 채우기
    df["analysis_ads_type_label"] = df["analysis_ads_type_label"].fillna("기타")
    df["final_media"] = df["final_media"].fillna("기타")
    df["category_name"] = df["category_name"].fillna("기타")

    return df


@st.cache_resource(show_spinner="광고별 집계 중...")
def build_ad_summary(_base: pd.DataFrame) -> pd.DataFrame:
    """광고별 전체 기간 성과 집계"""
    base = _base
    agg = base.groupby("ads_idx").agg(
        ads_name=("ads_name", "first"),
        analysis_ads_type_label=("analysis_ads_type_label", "first"),
        final_media=("final_media", "first"),
        category_name=("category_name", "first"),
        ads_sdate=("ads_sdate", "first"),
        avg_ctit=("avg_ctit", "first"),
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        scost=("rpt_time_scost", "sum"),
        acost=("rpt_time_acost", "sum"),
        cost=("rpt_time_cost", "sum"),
        earn=("rpt_time_earn", "sum"),
    ).reset_index()

    # 지표 계산
    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, np.nan)
    agg["cpa"] = np.where(agg["turn"] > 0, agg["acost"] / agg["turn"], np.nan)
    agg["margin"] = agg["acost"] - agg["earn"]
    agg["margin_rate"] = np.where(
        agg["acost"] > 0,
        (agg["acost"] - agg["earn"]) / agg["acost"] * 100,
        np.nan,
    )

    return agg


def filter_by_period(df: pd.DataFrame, period: str, today: pd.Timestamp,
                     custom_start=None, custom_end=None) -> pd.DataFrame:
    """관측 시점 필터 적용"""
    if period == "최근 1일":
        return df[df["rpt_time_date"] == today]
    elif period == "최근 7일":
        start = today - pd.Timedelta(days=6)
        return df[(df["rpt_time_date"] >= start) & (df["rpt_time_date"] <= today)]
    elif period == "전체":
        return df
    elif period == "사용자 지정" and custom_start and custom_end:
        s = pd.Timestamp(custom_start)
        e = pd.Timestamp(custom_end)
        return df[(df["rpt_time_date"] >= s) & (df["rpt_time_date"] <= e)]
    return df


def filter_by_hour(df: pd.DataFrame, hour_start: int, hour_end: int) -> pd.DataFrame:
    """시간대 필터"""
    return df[(df["rpt_time_time"] >= hour_start) & (df["rpt_time_time"] <= hour_end)]


def get_previous_period(today: pd.Timestamp, period: str, custom_start=None, custom_end=None):
    """이전 비교 기간 시작/종료일 반환"""
    if period == "최근 1일":
        prev_end = today - pd.Timedelta(days=1)
        return prev_end, prev_end
    elif period == "최근 7일":
        cur_start = today - pd.Timedelta(days=6)
        prev_end = cur_start - pd.Timedelta(days=1)
        prev_start = prev_end - pd.Timedelta(days=6)
        return prev_start, prev_end
    elif period == "사용자 지정" and custom_start and custom_end:
        s = pd.Timestamp(custom_start)
        e = pd.Timestamp(custom_end)
        days = (e - s).days
        prev_end = s - pd.Timedelta(days=1)
        prev_start = prev_end - pd.Timedelta(days=days)
        return prev_start, prev_end
    return None, None
