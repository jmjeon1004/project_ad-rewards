"""Page 3: 매체별 운영 탐색"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

st.markdown("""<style>
.block-container { padding-top: 3rem; }
[data-testid="stSidebar"] { min-width: 200px; }
div[data-testid="stVerticalBlockBorderWrapper"] { margin-top: 0; margin-bottom: 0; }
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div { background: #FFFFFF !important; }
div[data-testid="stExpander"] { background: #FFFFFF !important; }
div[data-testid="stExpanderDetails"] { background: #FFFFFF !important; }
section[data-testid="stMain"] { background: #FFFFFF !important; }
.stMainBlockContainer { background: #FFFFFF !important; }
div[data-testid="stMainBlockContainer"] { background: #FFFFFF !important; }
</style>""", unsafe_allow_html=True)

from src.init import ensure_data_loaded
from src.preprocessing import filter_by_period, filter_by_hour, get_previous_period
from src.metrics import calc_all_kpis, calc_group_kpis, calc_daily_kpis, calc_change_rate, _cached_compute, _cached_filter
from src.components import (
    render_kpi_card, render_context_message,
    render_ad_list_table_with_scores, render_ml_insight,
    render_chat_section, render_media_summary_table,
    render_media_highlight_card, render_media_material_card,
)
from src.charts import (
    make_bubble_chart,
)
from src.filters import render_page_header, render_filters, render_applied_chips
from src.rules import classify_quadrant, build_media_insight_cards
from src.ml_insight_data import get_cached_ml_insight_data
from src.config import COLORS, fmt_pct, fmt_currency, fmt_number

# ── 데이터 로딩 (session_state 통합 초기화) ──
ensure_data_loaded()
base = st.session_state["base"]
today = st.session_state["today"]
attr = st.session_state["attr"]
classification = st.session_state["classification"]
ad_master = st.session_state["ad_master"]
sched = st.session_state["sched"]
model_scores = st.session_state["model_scores"]

# ── 페이지 헤더 ──
render_page_header("매체별 운영 탐색")

# ── 필터 ──
group_col = "final_media"
group_items = sorted(base[group_col].dropna().unique().tolist())
type_items = sorted(base["analysis_ads_type_label"].dropna().unique().tolist())

period, selected_groups, (h_start, h_end), (c_start, c_end), sub_selected = render_filters(
    page_key="p3",
    group_col=group_col,
    group_items=group_items,
    group_label="매체",
    data_date_min=base["rpt_time_date"].min(),
    data_date_max=base["rpt_time_date"].max(),
    show_sub_filter=True,
    sub_filter_items=type_items,
    sub_filter_label="유형",
)

# ── 캐싱용 핑거프린트 ──
_filter_fp = f"p3|{period}|{sorted(selected_groups)}|{sorted(sub_selected)}|{h_start}-{h_end}|{c_start}-{c_end}"
_group_fp = f"p3g|{sorted(selected_groups)}|{sorted(sub_selected)}|{h_start}-{h_end}"

# ── 데이터 필터링 (session_state 캐싱) ──
def _apply_group_filter_p3(df, groups=selected_groups, sub=sub_selected,
                           ti=type_items, hs=h_start, he=h_end):
    gf = df[df[group_col].isin(groups)]
    if sub and len(sub) < len(ti):
        gf = gf[gf["analysis_ads_type_label"].isin(sub)]
    return filter_by_hour(gf, hs, he)

group_filtered = _cached_filter("p3_group_filtered", _group_fp, base, _apply_group_filter_p3)

def _apply_period_filter_p3(df, p=period, t=today, cs=c_start, ce=c_end):
    return filter_by_period(df, p, t, cs, ce)

filtered = _cached_filter("p3_filtered", _filter_fp, group_filtered, _apply_period_filter_p3)

has_custom = c_start is not None and c_end is not None
render_context_message(period, has_custom,
                       data_date_min=base["rpt_time_date"].min(),
                       data_date_max=base["rpt_time_date"].max(),
                       custom_start=c_start, custom_end=c_end)

render_applied_chips(
    page_key="p3", filtered_count=len(filtered),
    group_items=group_items, group_label="매체",
    show_sub_filter=True, sub_filter_items=type_items, sub_filter_label="유형",
)

if len(filtered) == 0:
    st.warning("선택한 필터 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ════════════════════════════════════════════════════
# KPI 카드 4종
# ════════════════════════════════════════════════════
kpis = _cached_compute("p3_kpis", _filter_fp, calc_all_kpis, filtered)
full_daily_kpis = _cached_compute("p3_daily_kpis", _group_fp, calc_daily_kpis, group_filtered)
avg_daily_turn = kpis["turn"] / max(len(full_daily_kpis), 1)

if period == "최근 1일":
    _hl_days = 1
elif period == "최근 7일":
    _hl_days = 7
else:
    _hl_days = 0
_hl_start = c_start if period == "사용자 지정" else None
_hl_end = c_end if period == "사용자 지정" else None

prev_start, prev_end = get_previous_period(today, period, c_start, c_end)
change_cvr, change_cpa, change_turn = None, None, None
if prev_start is not None:
    _prev_fp = f"p3_prev|{sorted(selected_groups)}|{sorted(sub_selected)}|{h_start}-{h_end}|{prev_start}|{prev_end}"

    def _compute_prev_kpis(prev_df=base, groups=selected_groups, ps=prev_start, pe=prev_end, hs=h_start, he=h_end):
        prev_df = prev_df[prev_df[group_col].isin(groups)]
        prev_df = prev_df[(prev_df["rpt_time_date"] >= ps) & (prev_df["rpt_time_date"] <= pe)]
        prev_df = filter_by_hour(prev_df, hs, he)
        return calc_all_kpis(prev_df) if len(prev_df) > 0 else None

    prev_kpis = _cached_compute("p3_prev_kpis", _prev_fp, _compute_prev_kpis)
    if prev_kpis:
        change_cvr = calc_change_rate(kpis["cvr"], prev_kpis["cvr"])
        change_cpa = calc_change_rate(kpis["cpa"], prev_kpis["cpa"])
        change_turn = calc_change_rate(kpis["turn"], prev_kpis["turn"])

k1, k2, k3 = st.columns(3)
with k1:
    render_kpi_card("평균 CVR", fmt_pct(kpis["cvr"]), change=change_cvr,
                    daily_df=full_daily_kpis, kpi_col="cvr",
                    avg_val=kpis["cvr"],
                    period=period, highlight_days=_hl_days, unit="%",
                    highlight_start=_hl_start, highlight_end=_hl_end)
with k2:
    render_kpi_card("평균 CPA", fmt_currency(kpis["cpa"]), change=change_cpa,
                    change_unit="원", daily_df=full_daily_kpis, kpi_col="cpa",
                    avg_val=kpis["cpa"],
                    invert_color=True, period=period, highlight_days=_hl_days, unit="원",
                    highlight_start=_hl_start, highlight_end=_hl_end)
with k3:
    render_kpi_card("총 완료수", fmt_number(kpis["turn"]),
                    change=change_turn, change_unit="건",
                    daily_df=full_daily_kpis, kpi_col="turn",
                    avg_val=avg_daily_turn, avg_label="일평균 완료수",
                    agg_method="sum",
                    period=period, highlight_days=_hl_days,
                    highlight_start=_hl_start, highlight_end=_hl_end)

# KPI 범례
st.markdown(f"""
<div style="
    background: {COLORS['legend_bg']}; border-radius: 8px;
    padding: 10px 20px; margin-top: 4px;
    display: flex; align-items: center; gap: 28px;
    font-size: 13px; color: #444;
">
    <span><span style="display:inline-block; width:14px; height:14px; background:{COLORS['bar_dim']};
        border-radius:2px; vertical-align:middle; margin-right:6px;"></span>일별 값</span>
    <span><span style="display:inline-block; width:14px; height:14px; background:{COLORS['bar_highlight']};
        border-radius:2px; vertical-align:middle; margin-right:6px;"></span>선택한 기간 (강조)</span>
    <span><span style="display:inline-block; width:28px; border-top:2px dashed #8A867F;
        vertical-align:middle; margin-right:6px;"></span>평균 KPI값</span>
    <span style="margin-left:12px; color:#888;">CPA는 ▼ 가 좋음, 나머지는 ▲ 가 좋음</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 매체별 종합 평가 버블 차트
# ════════════════════════════════════════════════════
group_kpis = _cached_compute("p3_group_kpis", _filter_fp, calc_group_kpis, filtered, group_col)

with st.container(border=True):
    _main_left, _main_right = st.columns([1, 1])

    with _main_left:
        st.markdown("### 매체별 종합 평가")
        st.caption("규모지표 완료수")

        _bubble_fp = f"p3_bubble|{_filter_fp}"
        _bubble_cached = _cached_compute("p3_bubble_result", _bubble_fp,
                                          lambda: make_bubble_chart(group_kpis, group_col,
                                                                     x_metric="cpa", y_metric="margin_rate",
                                                                     size_metric="turn", show_breakeven=False,
                                                                     show_acost=False))
        bubble_fig, overlap_groups = _bubble_cached
        bubble_clicked = st.plotly_chart(bubble_fig, use_container_width=True, on_select="rerun",
                                          key="p3_bubble")

    # 버블 클릭 처리
    _clicked_name = None
    if bubble_clicked and bubble_clicked.get("selection") and bubble_clicked["selection"].get("points"):
        pts = bubble_clicked["selection"]["points"]
        if pts:
            _clicked_name = pts[0].get("customdata", [None])[0] if pts[0].get("customdata") else None

    # 닫기 버튼 세션 관리
    if _clicked_name and _clicked_name != st.session_state.get("p3_last_click_name"):
        st.session_state.p3_hide_adlist = False
        st.session_state.p3_last_click_name = _clicked_name
    if not _clicked_name:
        st.session_state.p3_last_click_name = None

    _show_panel = _clicked_name and not st.session_state.get("p3_hide_adlist", False)

    with _main_right:
        if _show_panel:
            overlap_list = overlap_groups.get(_clicked_name)
            if overlap_list and len(overlap_list) > 1:
                _tg, _xc = st.columns([5, 1])
                with _tg:
                    _selected_name = st.radio(
                        "겹치는 매체 선택", overlap_list,
                        horizontal=True, key="p3_overlap_toggle",
                    )
                with _xc:
                    if st.button("✕", key="p3_close_adlist"):
                        st.session_state.p3_hide_adlist = True
                        st.rerun()
            else:
                _selected_name = _clicked_name
                _, _xc = st.columns([5, 1])
                with _xc:
                    if st.button("✕", key="p3_close_adlist"):
                        st.session_state.p3_hide_adlist = True
                        st.rerun()

            st.markdown(f"#### {_selected_name} 광고 Top 10")
            st.caption("CPA 오름차순 · 클릭 30건 이상")
            ad_filtered = filtered[filtered[group_col] == _selected_name]
            ad_agg = ad_filtered.groupby("ads_idx").agg(
                ads_name=("ads_name", "first"),
                analysis_ads_type_label=("analysis_ads_type_label", "first"),
                final_media=("final_media", "first"),
                clk=("rpt_time_clk", "sum"),
                turn=("rpt_time_turn", "sum"),
                acost=("rpt_time_acost", "sum"),
                scost=("rpt_time_scost", "sum"),
                earn=("rpt_time_earn", "sum"),
            ).reset_index()
            ad_agg["cvr"] = np.where(ad_agg["clk"] > 0, ad_agg["turn"] / ad_agg["clk"] * 100, np.nan)
            ad_agg["cpa"] = np.where(ad_agg["turn"] > 0, ad_agg["acost"] / ad_agg["turn"], np.nan)
            ad_agg = ad_agg[ad_agg["clk"] >= 30]
            render_ad_list_table_with_scores(ad_agg, model_scores, sort_col="cpa", ascending=True,
                                              hide_cols=("매체",))
        else:
            st.markdown(
                "<div style='display:flex; align-items:center; justify-content:center; height:500px;"
                "color:#999; font-size:14px;'>버블을 클릭하면 광고 리스트가 표시됩니다</div>",
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════
    # 매체 인사이트 카드 4종 (최우수·비효율·단가협업·소재개선)
    # ════════════════════════════════════════════════════
    quadrants = classify_quadrant(group_kpis, x_metric="cpa", y_metric="margin_rate")
    insight_cards = _cached_compute("p3_insight_cards", _filter_fp,
                                    build_media_insight_cards, quadrants, group_kpis, filtered, group_col, period,
                                    trend_df=group_filtered)

    cc1, cc2 = st.columns(2)
    with cc1:
        if insight_cards["best"]:
            render_media_highlight_card(**insight_cards["best"])
    with cc2:
        if insight_cards["loss"]:
            render_media_highlight_card(**insight_cards["loss"])
        elif insight_cards["loss_empty"]:
            st.markdown(
                f"<div style='border:1.5px solid #E0E0E0; border-radius:12px; padding:16px 18px;"
                f"height:100%; min-height:340px; display:flex; align-items:center; justify-content:center;"
                f"color:#888; font-size:13px; text-align:center;'>✅ {insight_cards['loss_empty']}</div>",
                unsafe_allow_html=True,
            )

    if insight_cards["material"]:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        render_media_material_card(**insight_cards["material"])

# ════════════════════════════════════════════════════
# 매체별 볼륨·효율·수익성 요약
# ════════════════════════════════════════════════════
render_media_summary_table(
    group_kpis, group_col,
    ads_df=filtered, model_scores=model_scores,
    group_label="매체", widget_key_prefix="p3_summary",
)

# ════════════════════════════════════════════════════
# ML 인사이트 (기회발굴 + 손실방어)
# ════════════════════════════════════════════════════
with st.container(border=True):
    _p3_thr_key = st.session_state.get("ml_m2_threshold_p3", "best_f1")
    _p3_fp = f"p3|{period}|{sorted(selected_groups)}|{h_start}-{h_end}|{c_start}-{c_end}|{_p3_thr_key}"
    _p3_insight = get_cached_ml_insight_data(
        page_key="p3",
        fingerprint=_p3_fp,
        filtered=filtered,
        model_scores=model_scores,
        ad_master=ad_master,
        sched=sched,
        attr=attr,
        classification=classification,
        page_name="매체별 운영 탐색",
        threshold_key=_p3_thr_key,
        opp_top_n=1,
        risk_top_n=1,
        sched_agg_precomp=st.session_state.get("sched_agg"),
        early_click_precomp=st.session_state.get("early_click_df"),
        master_cols_precomp=st.session_state.get("master_cols"),
    )
    render_ml_insight(_p3_insight, page_key="p3")

# ── AI-agent 채팅 ──
from src.preprocessing import build_ad_summary as _build_ad_summary

def _build_chat_summary():
    summary = _build_ad_summary(filtered)
    if model_scores is not None:
        _ms_cols = [c for c in ['ads_idx', 'm1_score', 'm1_grade', 'm2_proba', 'm2_decision'] if c in model_scores.columns]
        summary = summary.merge(model_scores[_ms_cols], on='ads_idx', how='left')
    return summary

_p3_ad_summary = _cached_compute("p3_ad_summary", _filter_fp, _build_chat_summary)

render_chat_section(
    page_key="p3",
    ad_summary=_p3_ad_summary,
    kpis=kpis,
    page_name="매체별 운영 탐색",
    filters_desc=f"기간: {period} / 매체: {', '.join(selected_groups)} / 시간: {h_start}~{h_end}시",
)
