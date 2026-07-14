"""Page 1: 전체 광고 현황 오버뷰"""
import streamlit as st
import pandas as pd
import numpy as np

st.session_state["_last_page"] = "p1"

# ── CSS 스타일 ──
st.markdown("""
<style>
    .block-container { padding-top: 3rem; }
    [data-testid="stSidebar"] { min-width: 200px; }
    div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E8E4DC; border-radius: 10px; padding: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF; border-radius: 8px; padding: 8px 16px;
        font-size: 13px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #4A7C59 !important; color: white !important; }
    /* 박스 컨테이너 간 여백 초기화 — SECTION_GAP 으로 통일 */
    div[data-testid="stVerticalBlockBorderWrapper"] { margin-top: 0; margin-bottom: 0; }
    /* 모든 bordered 컨테이너 배경 #FFFFFF */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div { background: #FFFFFF !important; }
    div[data-testid="stExpander"] { background: #FFFFFF !important; }
    div[data-testid="stExpanderDetails"] { background: #FFFFFF !important; }
    /* 메인 영역 및 모든 섹션 배경 흰색 */
    section[data-testid="stMain"] { background: #FFFFFF !important; }
    .stMainBlockContainer { background: #FFFFFF !important; }
    div[data-testid="stMainBlockContainer"] { background: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

from src.init import ensure_data_loaded
from src.preprocessing import build_ad_summary
from src.metrics import (
    calc_all_kpis, calc_group_kpis, calc_daily_kpis, calc_hourly_kpis,
    calc_cvr, calc_margin_rate, _cached_compute,
)
from src.components import (
    render_kpi_card,
    render_top3_card, render_alert_card, render_context_message,
    render_grade_distribution, render_risk_summary,
    render_ad_list_table_with_scores,
    render_ml_recommendation_banner, render_chat_section,
    get_kpi_title_html, get_heading_tooltip_html,
)
from src.charts import make_heatmap, make_hourly_cvr_trend, make_weekday_weekend_mini_bar
from src.ml_insight_data import get_cached_ml_insight_data
from src.config import COLORS, ML_GRADE_COLORS, fmt_number, fmt_pct

# ════════════════════════════════════════════════════
# 데이터 로딩 (session_state 통합 초기화)
# ════════════════════════════════════════════════════
ensure_data_loaded()
base = st.session_state["base"]
today = st.session_state["today"]
ad_summary = st.session_state["ad_summary"]
ad_master = st.session_state["ad_master"]
sched = st.session_state["sched"]
attr = st.session_state["attr"]
classification = st.session_state["classification"]
model_scores = st.session_state["model_scores"]

# 전체 기간 정보
date_min = base["rpt_time_date"].min()
date_max = base["rpt_time_date"].max()

# ════════════════════════════════════════════════════
# 상단 헤더
# ════════════════════════════════════════════════════
_current_view_mode = st.session_state.get("p1_view_mode") or "전체 기간"

st.image("assets/logo.png", width=120)
col_title, col_date = st.columns([3, 1])
with col_title:
    st.markdown("## 전체 광고 현황 오버뷰")
with col_date:
    if _current_view_mode != "전체 기간":
        st.markdown(
            f"<div style='text-align:right; padding-top:16px; color:#666; font-size:13px;'>"
            f"데이터 {today.strftime('%Y-%m-%d')}</div>",
            unsafe_allow_html=True,
        )

_bg_legend = COLORS["legend_bg"]
st.markdown(
    f"<div style='background:{_bg_legend}; padding:8px 16px; border-radius:8px; "
    f"font-size:13px; color:#555; display:flex; justify-content:space-between;'>"
    f"<span>분석 범위: {date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')} (전체 기간)</span>"
    f"<span>데이터 {len(base):,}건</span></div>",
    unsafe_allow_html=True,
)
SECTION_GAP = "<div style='margin-top:1.5rem;'></div>"
st.markdown(SECTION_GAP, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 보기 모드 토글 (전체 기간 / 당일 vs 평시)
# ════════════════════════════════════════════════════
view_mode = st.segmented_control(
    "보기 모드", options=["전체 기간", "당일 vs 평시"],
    default="전체 기간", key="p1_view_mode", label_visibility="collapsed",
)
if view_mode is None:
    view_mode = "전체 기간"

st.markdown(SECTION_GAP, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# KPI 카드 (전체 기간: 4종 / 당일 vs 평시: 3종)
# ════════════════════════════════════════════════════
_p1_data_fp = "p1_all"
kpis = _cached_compute("p1_kpis", _p1_data_fp, calc_all_kpis, base)
daily_kpis = _cached_compute("p1_daily_kpis", _p1_data_fp, calc_daily_kpis, base)

with st.container(border=False):
    if view_mode == "전체 기간":
        k1, k2, k3, k4 = st.columns(4)
    else:
        k1, k2, k3 = st.columns(3)
    _kpi_first_bar_color = COLORS["weekend"] if view_mode == "당일 vs 평시" else None
    _kpi_last_bar_color = COLORS["weekday"] if view_mode == "당일 vs 평시" else None
    with k1:
        avg_daily_clk = kpis["clk"] / max(len(daily_kpis), 1)
        render_kpi_card("총 클릭수", fmt_number(kpis["clk"]),
                        daily_df=daily_kpis, kpi_col="clk",
                        avg_val=avg_daily_clk,
                        agg_method="sum", first_bar_color=_kpi_first_bar_color,
                        last_bar_color=_kpi_last_bar_color)
    with k2:
        render_kpi_card("평균 CVR", fmt_pct(kpis["cvr"]),
                        daily_df=daily_kpis, kpi_col="cvr",
                        avg_val=kpis["cvr"], unit="%", first_bar_color=_kpi_first_bar_color,
                        last_bar_color=_kpi_last_bar_color)
    with k3:
        render_kpi_card("평균 마진율", fmt_pct(kpis["margin_rate"]),
                        daily_df=daily_kpis, kpi_col="margin_rate",
                        avg_val=kpis["margin_rate"], unit="%", first_bar_color=_kpi_first_bar_color,
                        last_bar_color=_kpi_last_bar_color)
    if view_mode == "전체 기간":
        with k4:
            # 평일/주말 CVR 격차
            weekday_df = base[~base["is_weekend"]]
            weekend_df = base[base["is_weekend"]]
            cvr_weekday = calc_cvr(weekday_df) or 0
            cvr_weekend = calc_cvr(weekend_df) or 0
            cvr_gap = cvr_weekday - cvr_weekend
            gap_sign = "+" if cvr_gap > 0 else ""
            arrow = "▲" if cvr_gap > 0 else "▼"
            with st.container(border=True):
                gap_desc = "평일 CVR 높음" if cvr_gap > 0 else "주말 CVR 높음"
                st.markdown(f"""
                <div style="padding: 4px 4px 0 4px; text-align: left;">
                    {get_kpi_title_html("평일 vs 주말 CVR 격차", color="#000")}
                    <div>
                        <span style="font-size:36px; font-weight:700; color:#000;">{gap_sign}{cvr_gap:.1f}</span>
                        <span style="font-size:20px; font-weight:400; color:#000;">%p {arrow}</span>
                    </div>
                    <div style="margin-top: 2px; min-height: 20px;">
                        <span style="font-size:12px; color:#000; font-weight:600;">평일 {cvr_weekday:.1f}%</span>
                        <span style="font-size:12px; color:#000;"> / </span>
                        <span style="font-size:12px; color:#000; font-weight:600;">주말 {cvr_weekend:.1f}%</span>
                        <span style="font-size:11px; color:#000; margin-left:6px;">({arrow} {gap_desc})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                wd_we_fig = make_weekday_weekend_mini_bar(base, cvr_weekday, cvr_weekend)
                st.plotly_chart(wd_we_fig, use_container_width=True,
                                config={"displayModeBar": False}, key="kpi_weekday_weekend")

# KPI 범례
_kpi_legend_weekend_label = "평시 값" if view_mode == "당일 vs 평시" else "주말"
st.markdown(f"""
<div style="
    background: {COLORS['legend_bg']}; border-radius: 8px;
    padding: 10px 20px;
    display: flex; align-items: center; gap: 28px;
    font-size: 13px; color: #444;
">
    <span><span style="display:inline-block; width:14px; height:14px; background:{COLORS['weekday']};
        border-radius:2px; vertical-align:middle; margin-right:6px;"></span>{"당일 값" if view_mode == "당일 vs 평시" else "일별 값 · 평일"}</span>
    <span><span style="display:inline-block; width:14px; height:14px; background:{COLORS['weekend']};
        border-radius:2px; vertical-align:middle; margin-right:6px;"></span>{_kpi_legend_weekend_label}</span>
    <span><span style="display:inline-block; width:28px; border-top:2px dashed #8A867F;
        vertical-align:middle; margin-right:6px;"></span>평균 KPI값</span>
</div>
""", unsafe_allow_html=True)

st.markdown(SECTION_GAP, unsafe_allow_html=True)

if view_mode == "전체 기간":
    # ════════════════════════════════════════════════════
    # 모델 결과 기반 운영 추천 (전체 기간 모드)
    # ════════════════════════════════════════════════════
    with st.container(border=True):
        _p1_thr_key = st.session_state.get("ml_m2_threshold_p1", "best_f1")
        _p1_fp = f"p1|all|{_p1_thr_key}"
        _p1_insight = get_cached_ml_insight_data(
            page_key="p1",
            fingerprint=_p1_fp,
            filtered=base,
            model_scores=model_scores,
            ad_master=ad_master,
            sched=sched,
            attr=attr,
            classification=classification,
            page_name="전체 오버뷰",
            threshold_key=_p1_thr_key,
            sched_agg_precomp=st.session_state.get("sched_agg"),
            early_click_precomp=st.session_state.get("early_click_df"),
            master_cols_precomp=st.session_state.get("master_cols"),
        )
        render_ml_recommendation_banner(_p1_insight, page_key="p1")

    st.markdown(SECTION_GAP, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 시간대 × 요일 히트맵 + 기본 출력 테이블 (전체 기간 모드)
# ════════════════════════════════════════════════════
if view_mode == "전체 기간":
    with st.container(border=True):
        hm1, hm2 = st.columns([1, 1])

        _weekday_options = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        _hour_options = list(range(24))

        with hm1:
            st.markdown(
                "#### 시간대 × 요일 CVR 히트맵\n\n"
                '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
                'CVR이 높은 요일·시간대를 파악해 광고 노출 전략을 최적화하세요.</div>'
                '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
                '색상 농도는 CVR 수준을 나타냅니다 (진할수록 높음)</div>',
                unsafe_allow_html=True,
            )
            heatmap_fig = _cached_compute("p1_heatmap", _p1_data_fp, make_heatmap, base)
            st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap_display")
            # 히트맵 아래 요일 × 시간대 선택
            _sel_wd_col, _sel_hr_col = st.columns(2)
            with _sel_wd_col:
                _sel_weekday = st.selectbox(
                    "요일", options=["전체"] + _weekday_options,
                    key="hm_sel_weekday", label_visibility="collapsed",
                )
            with _sel_hr_col:
                _sel_hour = st.selectbox(
                    "시간대", options=["전체"] + [f"{h}시" for h in _hour_options],
                    key="hm_sel_hour", label_visibility="collapsed",
                )
            st.markdown(
                '<div style="font-size:12px; color:#888; margin-top:4px;">'
                '요일·시간대를 선택하면 해당 구간의 광고가 표시됩니다</div>',
                unsafe_allow_html=True,
            )

        with hm2:
            _has_filter = (_sel_weekday != "전체") and (_sel_hour != "전체")

            if _has_filter:
                _sel_hour_int = int(_sel_hour.replace("시", ""))
                # 상단: 제목 + 전체 Top 10 복귀 버튼
                _hm_title_col, _hm_btn_col = st.columns([3, 1])
                with _hm_title_col:
                    st.markdown(
                        f"#### {_sel_weekday} {_sel_hour} CVR Top 10\n\n"
                        '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
                        'CVR 내림차순 · 클릭 30건 이상</div>',
                        unsafe_allow_html=True,
                    )
                with _hm_btn_col:
                    def _reset_hm_filters():
                        st.session_state["hm_sel_weekday"] = "전체"
                        st.session_state["hm_sel_hour"] = "전체"
                    st.button("전체 Top 10", key="hm_reset", on_click=_reset_hm_filters)
                # 해당 요일+시간 필터링
                _sel_weekday_short = _sel_weekday.replace("요일", "")
                filtered = base[(base["weekday_kr"] == _sel_weekday_short) & (base["rpt_time_time"] == _sel_hour_int)]
                if len(filtered) > 0:
                    ad_agg = filtered.groupby("ads_idx").agg(
                        ads_name=("ads_name", "first"),
                        analysis_ads_type_label=("analysis_ads_type_label", "first"),
                        final_media=("final_media", "first"),
                        clk=("rpt_time_clk", "sum"),
                        turn=("rpt_time_turn", "sum"),
                        acost=("rpt_time_acost", "sum"),
                        scost=("rpt_time_scost", "sum"),
                    ).reset_index()
                    ad_agg["cvr"] = np.where(ad_agg["clk"] > 0, ad_agg["turn"] / ad_agg["clk"] * 100, np.nan)
                    ad_agg["cpa"] = np.where(ad_agg["turn"] > 0, ad_agg["acost"] / ad_agg["turn"], np.nan)
                    ad_agg = ad_agg[ad_agg["clk"] >= 30].nlargest(10, "cvr")
                    render_ad_list_table_with_scores(ad_agg, model_scores, sort_col="cvr", ascending=False)
                else:
                    st.info("해당 시간대의 데이터가 없습니다.")
            else:
                st.markdown(
                    "#### 전체 기간 CVR Top 10\n\n"
                    '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
                    'CVR 상위 광고의 유형·매체 패턴에서 성공 요인을 확인하세요.</div>'
                    '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
                    'CVR 내림차순 · 클릭 30건 이상</div>',
                    unsafe_allow_html=True,
                )
                top_ads = ad_summary[ad_summary["clk"] >= 30].nlargest(10, "cvr")
                render_ad_list_table_with_scores(top_ads, model_scores, sort_col="cvr", ascending=False)

    st.markdown(SECTION_GAP, unsafe_allow_html=True)

if view_mode == "당일 vs 평시":
    # ════════════════════════════════════════════════════
    # 시간대별 CVR 추이 (당일 vs 평소 평균)
    # ════════════════════════════════════════════════════
    _cvr_trend_cached = _cached_compute("p1_cvr_trend", _p1_data_fp,
                                         lambda: make_hourly_cvr_trend(base, today))
    cvr_trend_fig, _merged_h = _cvr_trend_cached
    _today_df = _cached_compute("p1_today_df", _p1_data_fp,
                                 lambda: base[base["rpt_time_date"] == today])

    # 격차 최대 시점 계산 (차트에서 반환된 merged 재사용)
    _negative_gaps = _merged_h[_merged_h["gap"] < 0]
    if len(_negative_gaps) > 0:
        _max_gap_idx = _negative_gaps["gap"].idxmin()
    else:
        _max_gap_idx = _merged_h["gap"].idxmin()
    _max_gap_hour = int(_merged_h.loc[_max_gap_idx, "rpt_time_time"])
    _max_gap_val = _merged_h.loc[_max_gap_idx, "gap"]
    _max_today_cvr = _merged_h.loc[_max_gap_idx, "cvr_today"]
    _max_avg_cvr = _merged_h.loc[_max_gap_idx, "cvr_avg"]

    with st.container(border=True):
        _cvr_col_left, _cvr_col_right = st.columns([1, 1])
        with _cvr_col_left:
            st.markdown(
                "### 시간대별 CVR — 당일 vs 평시\n\n"
                '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
                '평시 대비 CVR이 급락한 시간대를 확인하고 이상 광고를 즉시 점검하세요.</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"당일 {today.strftime('%Y-%m-%d')}")
            cvr_trend_clicked = st.plotly_chart(cvr_trend_fig, use_container_width=True,
                                                 on_select="rerun", key="hourly_cvr")

        # 포인트 클릭 감지 (어떤 시간대 포인트든 클릭 가능)
        _cvr_marker_clicked = False
        _clicked_hour = _max_gap_hour
        if cvr_trend_clicked and cvr_trend_clicked.get("selection") and cvr_trend_clicked["selection"].get("points"):
            _cvr_pts = cvr_trend_clicked["selection"]["points"]
            if _cvr_pts:
                _cvr_marker_clicked = True
                _clicked_hour = int(str(_cvr_pts[0]["x"]).replace("시", ""))

        _clicked_row = _merged_h[_merged_h["rpt_time_time"] == _clicked_hour]
        if len(_clicked_row) > 0:
            _clicked_gap_val = _clicked_row["gap"].iloc[0]
            _clicked_today_cvr = _clicked_row["cvr_today"].iloc[0]
            _clicked_avg_cvr = _clicked_row["cvr_avg"].iloc[0]
        else:
            _clicked_gap_val, _clicked_today_cvr, _clicked_avg_cvr = _max_gap_val, _max_today_cvr, _max_avg_cvr

        with _cvr_col_right:
            if _cvr_marker_clicked:
                # 클릭한 시간대의 광고 리스트 표시
                st.markdown(f"### 🚨 {_clicked_hour}시 이상 광고 리스트")
                st.caption(f"격차 {_clicked_gap_val:+.1f}%p · 당일 CVR {_clicked_today_cvr:.1f}% · 평소 {_clicked_avg_cvr:.1f}%")
                st.caption("CVR 오름차순 · 클릭 30건 이상")

                _alert_df = _today_df[_today_df["rpt_time_time"] == _clicked_hour]
                if len(_alert_df) > 0:
                    _alert_agg = _alert_df.groupby("ads_idx").agg(
                        ads_name=("ads_name", "first"),
                        analysis_ads_type_label=("analysis_ads_type_label", "first"),
                        final_media=("final_media", "first"),
                        clk=("rpt_time_clk", "sum"),
                        turn=("rpt_time_turn", "sum"),
                        acost=("rpt_time_acost", "sum"),
                        scost=("rpt_time_scost", "sum"),
                    ).reset_index()
                    _alert_agg["cvr"] = np.where(_alert_agg["clk"] > 0,
                                                  _alert_agg["turn"] / _alert_agg["clk"] * 100, np.nan)
                    _alert_agg["cpa"] = np.where(_alert_agg["turn"] > 0,
                                                  _alert_agg["acost"] / _alert_agg["turn"], np.nan)
                    _alert_agg = _alert_agg[_alert_agg["clk"] >= 30]
                    render_ad_list_table_with_scores(_alert_agg, model_scores, sort_col="cvr", ascending=True)
                else:
                    st.info("해당 시간대의 데이터가 없습니다.")
            else:
                st.markdown(
                    "<div style='display:flex; align-items:center; justify-content:center; height:320px; "
                    "color:#999; font-size:14px;'>차트를 클릭하면 해당 시간대 이상 광고 리스트가 표시됩니다</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(SECTION_GAP, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # 차원별 TOP 3 요약 카드
    # ════════════════════════════════════════════════════
    def _make_top3(group_col, label, page_num):
        gk = _cached_compute(f"p1_gk_{group_col}", _p1_data_fp, calc_group_kpis, base, group_col)
        today_gk = _cached_compute(f"p1_gk_today_{group_col}", _p1_data_fp,
                                    calc_group_kpis, base[base["rpt_time_date"] == today], group_col)

        total_clk = gk["clk"].sum()

        items = []
        for _, row in gk.iterrows():
            name = row[group_col]
            today_row = today_gk[today_gk[group_col] == name]
            today_cvr = today_row["cvr"].values[0] if len(today_row) > 0 else row["cvr"]
            gap = (today_cvr - row["cvr"]) if pd.notna(today_cvr) and pd.notna(row["cvr"]) else 0
            share = row["clk"] / total_clk * 100 if total_clk > 0 else 0
            items.append({
                "name": name, "cvr": today_cvr, "gap": gap,
                "clk": row["clk"], "share": share,
            })

        # 오늘 CVR 내림차순 정렬 후 상위 3개
        items.sort(key=lambda x: x["cvr"] if pd.notna(x["cvr"]) else -1, reverse=True)
        top3 = items[:3]

        # 배지 판단 (정상/주의/위험)
        cvr_vals = [it["cvr"] for it in items if pd.notna(it["cvr"])]
        top3_cvr_vals = [it["cvr"] for it in top3 if pd.notna(it["cvr"])]
        all_avg_cvr = np.nanmean(cvr_vals) if cvr_vals else 0
        top3_avg_cvr = np.nanmean(top3_cvr_vals) if top3_cvr_vals else 0

        if top3_avg_cvr <= all_avg_cvr:
            badge = "위험"
        elif top3_avg_cvr <= all_avg_cvr * 1.1:
            badge = "주의"
        else:
            badge = "정상"

        return top3, len(gk), badge

    # 바로가기 버튼을 카드 박스 안에 연결하는 CSS
    st.markdown("""
    <style>
    .st-key-goto_type,
    .st-key-goto_media,
    .st-key-goto_cat {
        margin-top: -1rem;
    }
    .st-key-goto_type button,
    .st-key-goto_media button,
    .st-key-goto_cat button {
        background: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0 0 12px 12px !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        padding: 12px 20px !important;
        width: 100%;
        transition: background 0.15s;
    }
    .st-key-goto_type button:hover,
    .st-key-goto_media button:hover,
    .st-key-goto_cat button:hover {
        background: #333333 !important;
        color: #FFFFFF !important;
    }
    .st-key-goto_type button p,
    .st-key-goto_media button p,
    .st-key-goto_cat button p {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _scroll_to_top3 = st.session_state.pop("scroll_to_top3", False)

    st.html('<div id="top3-section"></div>')

    with st.container(border=True):
        st.markdown(
            get_heading_tooltip_html("차원별 Top3 - 당일 vs 평시", "배지 기준", [
                ("정상", COLORS["positive"], ": TOP3 평균 CVR이 전체 평균보다 10% 초과로 높음"),
                ("주의", COLORS["warning"], ": 전체 평균보다 높지만 10% 이하"),
                ("위험", COLORS["negative"], ": 전체 평균 이하"),
            ]),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
            '유형·매체·카테고리별 CVR 상위 3개를 평시와 비교해 당일의 성과 변화를 한눈에 파악하세요.</div>',
            unsafe_allow_html=True,
        )
        st.caption("CVR 내림차순")

        t1, t2, t3 = st.columns(3)
        with t1:
            items, cnt, badge = _make_top3("analysis_ads_type_label", "유형별", 2)
            render_top3_card("유형별", cnt, items, "02", badge=badge, button_label="유형별 페이지 ↗")
            if st.button("유형별 페이지 ↗", key="goto_type", use_container_width=True):
                st.switch_page("pages/02_유형별_운영_탐색.py")
        with t2:
            items, cnt, badge = _make_top3("final_media", "매체별", 3)
            render_top3_card("매체별", cnt, items, "03", badge=badge, button_label="매체별 페이지 ↗")
            if st.button("매체별 페이지 ↗", key="goto_media", use_container_width=True):
                st.switch_page("pages/03_매체별_운영_탐색.py")
        with t3:
            items, cnt, badge = _make_top3("category_name", "카테고리별", 4)
            render_top3_card("카테고리별", cnt, items, "04", badge=badge, button_label="카테고리별 페이지 ↗")
            if st.button("카테고리별 페이지 ↗", key="goto_cat", use_container_width=True):
                st.switch_page("pages/04_카테고리별_운영_탐색.py")

    if _scroll_to_top3:
        st.html("""<script>
        (function() {
            function tryScroll(attempts) {
                var el = document.getElementById("top3-section");
                if (el) {
                    el.scrollIntoView({behavior: "smooth", block: "start"});
                } else if (attempts < 25) {
                    setTimeout(function() { tryScroll(attempts + 1); }, 200);
                }
            }
            tryScroll(0);
        })();
        </script>""", unsafe_allow_javascript=True)

    st.markdown(SECTION_GAP, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# AI-agent 채팅
# ════════════════════════════════════════════════════
with st.container(border=False):
    render_chat_section(
        page_key="p1",
        ad_summary=ad_summary,
        kpis=kpis,
        page_name="전체 광고 현황 오버뷰",
        filters_desc="전체 기간, 필터 없음",
    )
