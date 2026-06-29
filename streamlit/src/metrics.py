"""합계 기반 비율 지표 계산 — 분모 0 → null 처리"""
import threading
import numpy as np
import pandas as pd
import streamlit as st

# 프로세스 전역 캐시 — 원본 데이터는 앱 수명 동안 불변이므로 fingerprint가 같으면
# 세션(사용자)이 달라도 동일 결과를 공유한다. 세션별 재계산 비용을 제거해
# 페이지 로딩 시간을 줄인다.
_GLOBAL_CACHE: dict[tuple[str, str], object] = {}
_GLOBAL_CACHE_LOCK = threading.Lock()


def _cached_compute(cache_key: str, fingerprint: str, compute_fn, *args, **kwargs):
    """프로세스 전역 핑거프린트 캐싱. 동일 fingerprint이면 캐시 반환 (세션 무관 공유)."""
    full_key = (cache_key, fingerprint)
    cached = _GLOBAL_CACHE.get(full_key)
    if cached is not None:
        return cached
    result = compute_fn(*args, **kwargs)
    with _GLOBAL_CACHE_LOCK:
        _GLOBAL_CACHE[full_key] = result
    return result


def _cached_filter(cache_key: str, fingerprint: str, base: pd.DataFrame,
                   filter_fn, *args, **kwargs) -> pd.DataFrame:
    """필터링된 DataFrame을 프로세스 전역으로 캐싱. 동일 fingerprint이면 즉시 반환."""
    full_key = (cache_key, fingerprint)
    cached = _GLOBAL_CACHE.get(full_key)
    if cached is not None:
        return cached
    result = filter_fn(base, *args, **kwargs)
    with _GLOBAL_CACHE_LOCK:
        _GLOBAL_CACHE[full_key] = result
    return result


def calc_cvr(df: pd.DataFrame) -> float | None:
    """CVR = sum(turn) / sum(clk) * 100"""
    clk = df["rpt_time_clk"].sum()
    if clk == 0:
        return None
    return df["rpt_time_turn"].sum() / clk * 100


def calc_cpa(df: pd.DataFrame) -> float | None:
    """CPA = sum(acost) / sum(turn)"""
    turn = df["rpt_time_turn"].sum()
    if turn == 0:
        return None
    return df["rpt_time_acost"].sum() / turn


def calc_margin_rate(df: pd.DataFrame) -> float | None:
    """마진율 = (sum(acost) - sum(earn)) / sum(acost) * 100"""
    acost = df["rpt_time_acost"].sum()
    if acost == 0:
        return None
    return (acost - df["rpt_time_earn"].sum()) / acost * 100


def calc_all_kpis(df: pd.DataFrame) -> dict:
    """모든 KPI를 dict로 반환"""
    return {
        "clk": int(df["rpt_time_clk"].sum()),
        "turn": int(df["rpt_time_turn"].sum()),
        "acost": int(df["rpt_time_acost"].sum()),
        "scost": int(df["rpt_time_scost"].sum()),
        "earn": int(df["rpt_time_earn"].sum()),
        "cvr": calc_cvr(df),
        "cpa": calc_cpa(df),
        "margin_rate": calc_margin_rate(df),
    }


def calc_group_kpis(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """그룹별 KPI 집계"""
    agg = df.groupby(group_col).agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        acost=("rpt_time_acost", "sum"),
        scost=("rpt_time_scost", "sum"),
        earn=("rpt_time_earn", "sum"),
    ).reset_index()

    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, np.nan)
    agg["cpa"] = np.where(agg["turn"] > 0, agg["acost"] / agg["turn"], np.nan)
    agg["margin"] = agg["acost"] - agg["earn"]
    agg["margin_rate"] = np.where(
        agg["acost"] > 0, (agg["acost"] - agg["earn"]) / agg["acost"] * 100, np.nan
    )

    return agg


def calc_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """일자별 KPI"""
    agg = df.groupby("rpt_time_date").agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        acost=("rpt_time_acost", "sum"),
        scost=("rpt_time_scost", "sum"),
        earn=("rpt_time_earn", "sum"),
        is_weekend=("is_weekend", "first"),
    ).reset_index()

    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, np.nan)
    agg["cpa"] = np.where(agg["turn"] > 0, agg["acost"] / agg["turn"], np.nan)
    agg["margin_rate"] = np.where(
        agg["acost"] > 0, (agg["acost"] - agg["earn"]) / agg["acost"] * 100, np.nan
    )

    return agg.sort_values("rpt_time_date")


def calc_hourly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 KPI"""
    agg = df.groupby("rpt_time_time").agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        acost=("rpt_time_acost", "sum"),
        scost=("rpt_time_scost", "sum"),
        earn=("rpt_time_earn", "sum"),
    ).reset_index()

    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, np.nan)
    return agg.sort_values("rpt_time_time")


def calc_change_rate(current_val, previous_val) -> float | None:
    """이전 기간 대비 증감율"""
    if current_val is None or previous_val is None:
        return None
    if previous_val == 0:
        return None
    return current_val - previous_val
