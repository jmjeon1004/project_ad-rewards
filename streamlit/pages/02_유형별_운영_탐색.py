"""Page 2: 유형별 운영 탐색"""
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
from src.metrics import calc_all_kpis, calc_daily_kpis, calc_change_rate, _cached_compute, _cached_filter
from src.components import (render_kpi_card, render_context_message,
                            render_ml_insight, render_chat_section)
from src.filters import render_page_header, render_filters, render_applied_chips
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
render_page_header("유형별 운영 탐색")

# ── 필터 ──
group_col = "analysis_ads_type_label"
group_items = sorted(base[group_col].dropna().unique().tolist())

period, selected_groups, (h_start, h_end), (c_start, c_end), _ = render_filters(
    page_key="p2",
    group_col=group_col,
    group_items=group_items,
    group_label="유형",
    data_date_min=base["rpt_time_date"].min(),
    data_date_max=base["rpt_time_date"].max(),
)

# ── 캐싱용 핑거프린트 ──
_filter_fp = f"p2|{period}|{sorted(selected_groups)}|{h_start}-{h_end}|{c_start}-{c_end}"
_group_fp = f"p2g|{sorted(selected_groups)}|{h_start}-{h_end}"

# ── 데이터 필터링 (session_state 캐싱) ──
def _apply_group_filter(df, groups=selected_groups, hs=h_start, he=h_end):
    gf = df[df[group_col].isin(groups)]
    return filter_by_hour(gf, hs, he)

group_filtered = _cached_filter("p2_group_filtered", _group_fp, base, _apply_group_filter)

def _apply_period_filter(df, p=period, t=today, cs=c_start, ce=c_end):
    return filter_by_period(df, p, t, cs, ce)

filtered = _cached_filter("p2_filtered", _filter_fp, group_filtered, _apply_period_filter)

# 컨텍스트 메시지
has_custom = c_start is not None and c_end is not None
render_context_message(period, has_custom,
                       data_date_min=base["rpt_time_date"].min(),
                       data_date_max=base["rpt_time_date"].max(),
                       custom_start=c_start, custom_end=c_end)

render_applied_chips(
    page_key="p2", filtered_count=len(filtered),
    group_items=group_items, group_label="유형",
)

if len(filtered) == 0:
    st.warning("선택한 필터 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ════════════════════════════════════════════════════
# KPI 카드 3종
# ════════════════════════════════════════════════════
kpis = _cached_compute("p2_kpis", _filter_fp, calc_all_kpis, filtered)
full_daily_kpis = _cached_compute("p2_daily_kpis", _group_fp, calc_daily_kpis, group_filtered)

# highlight_days 계산
if period == "최근 1일":
    _hl_days = 1
elif period == "최근 7일":
    _hl_days = 7
else:
    _hl_days = 0
_hl_start = c_start if period == "사용자 지정" else None
_hl_end = c_end if period == "사용자 지정" else None

# 이전 기간 대비
prev_start, prev_end = get_previous_period(today, period, c_start, c_end)
change_cvr, change_cpa, change_mr = None, None, None
if prev_start is not None:
    _prev_fp = f"p2_prev|{sorted(selected_groups)}|{h_start}-{h_end}|{prev_start}|{prev_end}"

    def _compute_prev_kpis(prev_df=base, groups=selected_groups, ps=prev_start, pe=prev_end, hs=h_start, he=h_end):
        prev_df = prev_df[prev_df[group_col].isin(groups)]
        prev_df = prev_df[(prev_df["rpt_time_date"] >= ps) & (prev_df["rpt_time_date"] <= pe)]
        prev_df = filter_by_hour(prev_df, hs, he)
        return calc_all_kpis(prev_df) if len(prev_df) > 0 else None

    prev_kpis = _cached_compute("p2_prev_kpis", _prev_fp, _compute_prev_kpis)
    if prev_kpis:
        change_cvr = calc_change_rate(kpis["cvr"], prev_kpis["cvr"])
        change_cpa = calc_change_rate(kpis["cpa"], prev_kpis["cpa"])
        change_mr = calc_change_rate(kpis["margin_rate"], prev_kpis["margin_rate"])

k1, k2, k3 = st.columns(3)
with k1:
    render_kpi_card("평균 CVR", fmt_pct(kpis["cvr"]), change=change_cvr,
                    daily_df=full_daily_kpis, kpi_col="cvr",
                    avg_val=kpis["cvr"], avg_label="평균 cvr값",
                    period=period, highlight_days=_hl_days, unit="%",
                    highlight_start=_hl_start, highlight_end=_hl_end)
with k2:
    render_kpi_card("평균 CPA", fmt_currency(kpis["cpa"]), change=change_cpa,
                    change_unit="원", daily_df=full_daily_kpis, kpi_col="cpa",
                    avg_val=kpis["cpa"], avg_label="평균 cpa값",
                    invert_color=True, period=period, highlight_days=_hl_days, unit="원",
                    highlight_start=_hl_start, highlight_end=_hl_end)
with k3:
    render_kpi_card("평균 마진율", fmt_pct(kpis["margin_rate"]), change=change_mr,
                    daily_df=full_daily_kpis, kpi_col="margin_rate",
                    avg_val=kpis["margin_rate"], avg_label="평균 마진율값",
                    period=period, highlight_days=_hl_days, unit="%",
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
# 이동 조치 필요 광고 (운영 우선순위 대시보드)
# ════════════════════════════════════════════════════
with st.container(border=True):
    # ── 설정 ──
    _OPD_PRIMARY_METRIC = {
        "설치형": "cpa", "감상형": "cpa", "클릭형": "cpa", "수행형": "cpa",
        "참여형": "completes", "구매형": "margin", "노출형": "margin", "기타": "cpa",
    }
    _OPD_METRIC_NAME     = {"cpa": "CPA", "margin": "마진율", "completes": "완료수", "cvr": "CVR"}
    _OPD_METRIC_UNIT     = {"cpa": "원", "margin": "%", "completes": "건", "cvr": "%"}
    _OPD_LOWER_IS_BETTER = {"cpa": True, "margin": False, "completes": False, "cvr": False}
    _OPD_TYPE_ORDER      = ["클릭형", "감상형", "수행형", "참여형", "구매형", "설치형", "노출형", "기타"]

    _OPD_MODE_CONFIG = {
        "urgent":  {"label": "즉시조치", "color": "#C62828"},
        "early":   {"label": "우선검토", "color": "#E65100"},
        "expand":  {"label": "매체확장", "color": "#1565C0"},
        "promote": {"label": "승급추진", "color": "#2E7D32"},
    }
    _OPD_GRADE_BADGE_STYLE = {
        "즉시조치": {"bg": "#C62828", "fg": "#ffffff"},
        "우선검토": {"bg": "#E65100", "fg": "#ffffff"},
        "매체확장": {"bg": "#1565C0", "fg": "#ffffff"},
        "승급추진": {"bg": "#2E7D32", "fg": "#ffffff"},
    }
    _OPD_CHART_CAPTION = {
        "urgent":  "유형별 즉시조치 비중",
        "early":   "유형별 우선검토 비중",
        "expand":  "유형별 매체확장 비중",
        "promote": "유형별 승급추진 비중",
    }

    # ── CSS ──
    st.markdown("""
    <style>
    .opd-ad-row {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 10px;
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 8px; margin-bottom: 6px;
    }
    .opd-name {
        flex: 1; font-weight: 500; font-size: 14px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .opd-metric-value { font-size: 15px; font-weight: 500; text-align: right; }
    .opd-metric-label { font-size: 11px; color: rgba(49,51,63,0.6); text-align: right; }
    .opd-badge { font-size: 11px; padding: 2px 8px; border-radius: 6px; white-space: nowrap; flex-shrink: 0; }
    .opd-type-header { font-size: 13px; font-weight: 500; margin-bottom: 6px; }
    .opd-type-subcaption { font-size: 12px; color: rgba(49,51,63,0.6); margin-left: 6px; }
    </style>
    """, unsafe_allow_html=True)

    # ── 데이터 준비 ──
    _opd_earn_col = "rpt_time_earn" if "rpt_time_earn" in filtered.columns else None
    _opd_agg_kwargs = dict(
        ads_name=("ads_name", "first"),
        type_label=(group_col, "first"),
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        acost=("rpt_time_acost", "sum"),
    )
    if _opd_earn_col:
        _opd_agg_kwargs["earn"] = (_opd_earn_col, "sum")
    _opd_agg = filtered.groupby("ads_idx").agg(**_opd_agg_kwargs).reset_index()
    if "earn" not in _opd_agg.columns:
        _opd_agg["earn"] = 0.0

    _opd_agg["cvr"] = np.where(
        _opd_agg["clk"] > 0, _opd_agg["turn"] / _opd_agg["clk"] * 100, np.nan
    )
    _opd_agg["cpa"] = np.where(
        _opd_agg["turn"] > 0, _opd_agg["acost"] / _opd_agg["turn"], np.nan
    )
    _opd_agg["margin"] = np.where(
        _opd_agg["acost"] > 0,
        (_opd_agg["acost"] - _opd_agg["earn"]) / _opd_agg["acost"] * 100, np.nan
    )
    _opd_agg["completes"] = _opd_agg["turn"].astype(float)

    if model_scores is not None and not model_scores.empty:
        _ms_cols = [c for c in ["ads_idx", "m1_grade", "m1_score", "m2_decision", "m2_proba"]
                    if c in model_scores.columns]
        _opd_agg = _opd_agg.merge(model_scores[_ms_cols], on="ads_idx", how="left")

    _opd_has_m1 = "m1_grade" in _opd_agg.columns
    _opd_has_m2 = "m2_decision" in _opd_agg.columns

    if _opd_has_m1:
        if "m1_score" in _opd_agg.columns:
            _opd_agg["opd_grade"] = np.where(
                _opd_agg["m1_grade"] == "D", "즉시조치",
                np.where(
                    (_opd_agg["m1_grade"] == "A") & (_opd_agg["m1_score"] >= 75),
                    "승급추진",
                    np.where(_opd_agg["m1_grade"].isin(["S", "A"]), "매체확장", None),
                ),
            )
        else:
            _opd_agg["opd_grade"] = np.where(
                _opd_agg["m1_grade"] == "D", "즉시조치",
                np.where(_opd_agg["m1_grade"].isin(["S", "A"]), "매체확장", None),
            )
    else:
        _opd_agg["opd_grade"] = None

    _opd_agg["opd_early"] = (
        _opd_agg["m2_decision"] == "decline_risk" if _opd_has_m2 else False
    )

    def _opd_mode_df(mode_key: str) -> pd.DataFrame:
        if mode_key == "urgent":
            return _opd_agg[_opd_agg["opd_grade"] == "즉시조치"]
        if mode_key == "early":
            return _opd_agg[_opd_agg["opd_early"] == True]
        if mode_key == "expand":
            return _opd_agg[_opd_agg["opd_grade"] == "매체확장"]
        if mode_key == "promote":
            return _opd_agg[_opd_agg["opd_grade"] == "승급추진"]
        return _opd_agg.iloc[0:0]

    _opd_type_totals  = _opd_agg.groupby("type_label")["ads_idx"].count().to_dict()
    _opd_grade_counts = {
        mk: _opd_mode_df(mk).groupby("type_label")["ads_idx"].count().to_dict()
        for mk in _OPD_MODE_CONFIG
    }
    _opd_mode_counts = {mk: len(_opd_mode_df(mk)) for mk in _OPD_MODE_CONFIG}

    # ── 헬퍼 ──
    def _opd_status_icon(value, peer_values: list, lower_is_better: bool) -> str:
        if pd.isna(value):
            return ""
        valid = [v for v in peer_values if not (isinstance(v, float) and pd.isna(v))]
        if len(valid) < 2:
            return ""
        pct = sum(1 for v in sorted(valid) if v < value) / (len(valid) - 1)
        if not lower_is_better:
            pct = 1 - pct
        if pct >= 0.66:
            return '<span style="color:#A32D2D; font-size:16px;">↓</span>'
        if pct >= 0.33:
            return '<span style="color:#854F0B; font-size:16px;">−</span>'
        return '<span style="color:#3B6D11; font-size:16px;">↑</span>'

    def _opd_grade_badge_html(grade: str) -> str:
        s = _OPD_GRADE_BADGE_STYLE.get(grade, {"bg": "#F1EFE8", "fg": "#5F5E5A"})
        return f'<span class="opd-badge" style="background:{s["bg"]}; color:{s["fg"]};">{grade}</span>'

    def _opd_get_metric(type_label: str, grade_label: str) -> str:
        if grade_label in ("매체확장", "승급추진"):
            return "cvr"
        return _OPD_PRIMARY_METRIC.get(type_label, "cpa")

    def _opd_fmt_value(val, metric_key: str) -> str:
        if pd.isna(val):
            return "-"
        if metric_key in ("cpa", "completes"):
            return f"{val:,.0f}"
        return f"{val:.1f}"

    # ── session state 초기화 ──
    if "p2_opd_mode" not in st.session_state:
        st.session_state["p2_opd_mode"] = "urgent"
    if "p2_opd_show_n" not in st.session_state:
        st.session_state["p2_opd_show_n"] = 5

    # ── 헤더 + 모드 버튼 ──
    st.markdown("### 이동 조치 필요 광고")

    _opd_btn_row = st.columns(len(_OPD_MODE_CONFIG))
    for _opd_col, (_opd_mk, _opd_cfg) in zip(_opd_btn_row, _OPD_MODE_CONFIG.items()):
        _opd_btn_lbl  = f"{_opd_cfg['label']} {_opd_mode_counts[_opd_mk]:,}건"
        _opd_btn_type = "primary" if st.session_state["p2_opd_mode"] == _opd_mk else "secondary"
        if _opd_col.button(
            _opd_btn_lbl, type=_opd_btn_type,
            use_container_width=True, key=f"p2_opd_btn_{_opd_mk}",
        ):
            st.session_state["p2_opd_mode"] = _opd_mk
            st.session_state["p2_opd_show_n"] = 5
            st.rerun()

    _opd_cur_mode  = st.session_state["p2_opd_mode"]
    _opd_cur_label = _OPD_MODE_CONFIG[_opd_cur_mode]["label"]
    _opd_cur_color = _OPD_MODE_CONFIG[_opd_cur_mode]["color"]

    # ── 본문: 좌측 리스트 / 우측 차트+필터 ──
    _opd_col_list, _opd_col_chart = st.columns([1.55, 1])

    with _opd_col_chart:
        st.caption(_OPD_CHART_CAPTION[_opd_cur_mode])

        _opd_counts = _opd_grade_counts[_opd_cur_mode]
        _opd_shares = sorted(
            [
                {"type": t, "count": c, "pct": c / _opd_type_totals.get(t, 1) * 100}
                for t, c in _opd_counts.items()
                if c > 0
            ],
            key=lambda s: s["pct"],
            reverse=True,
        )
        _opd_max_pct = max((s["pct"] for s in _opd_shares), default=1)

        if not _opd_shares:
            st.info(f"'{_opd_cur_label}' 조건의 광고가 없습니다.")
        else:
            for _opd_s in _opd_shares:
                _opd_bar_w = int(_opd_s["pct"] / _opd_max_pct * 100)
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;">'
                    f'<span>{_opd_s["type"]}</span>'
                    f'<span style="color:rgba(49,51,63,0.6);">'
                    f'{_opd_s["count"]}건 · {_opd_s["pct"]:.1f}%</span></div>'
                    f'<div style="background:#F0F0F0; border-radius:4px; height:8px; margin-bottom:8px;">'
                    f'<div style="background:{_opd_cur_color}; border-radius:4px; height:8px; width:{_opd_bar_w}%;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with _opd_col_list:
        _opd_list_df = _opd_mode_df(_opd_cur_mode).copy()
        _opd_show_n  = st.session_state["p2_opd_show_n"]

        _opd_grouped: dict[str, list] = {}
        for _, _r in _opd_list_df.iterrows():
            _opd_grouped.setdefault(_r["type_label"], []).append(_r)

        if _opd_list_df.empty:
            st.info("조건에 맞는 광고가 없습니다.")
        else:
            _opd_all_types = (
                [t for t in _OPD_TYPE_ORDER if t in _opd_grouped]
                + [t for t in _opd_grouped if t not in _OPD_TYPE_ORDER]
            )
            _opd_rendered = 0
            for _opd_at in _opd_all_types:
                if _opd_rendered >= _opd_show_n:
                    break
                _opd_ads = _opd_grouped.get(_opd_at)
                if not _opd_ads:
                    continue

                _opd_mk  = _opd_get_metric(_opd_at, _opd_cur_label)
                _opd_dir = "낮을수록 좋음" if _OPD_LOWER_IS_BETTER[_opd_mk] else "높을수록 좋음"

                st.markdown(
                    f'<div class="opd-type-header">{_opd_at}'
                    f'<span class="opd-type-subcaption">'
                    f'기준 지표: {_OPD_METRIC_NAME[_opd_mk]} ({_opd_dir})</span></div>',
                    unsafe_allow_html=True,
                )

                _opd_peers = [
                    float(ad[_opd_mk]) if pd.notna(ad.get(_opd_mk)) else float("nan")
                    for ad in _opd_ads
                ]

                for _opd_ad in _opd_ads:
                    if _opd_rendered >= _opd_show_n:
                        break
                    _opd_val     = _opd_ad.get(_opd_mk)
                    _opd_val_str = _opd_fmt_value(_opd_val, _opd_mk)
                    _opd_unit    = _OPD_METRIC_UNIT[_opd_mk]
                    _opd_icon    = _opd_status_icon(
                        float(_opd_val) if pd.notna(_opd_val) else float("nan"),
                        _opd_peers,
                        _OPD_LOWER_IS_BETTER[_opd_mk],
                    )
                    _opd_name = str(_opd_ad.get("ads_name") or f'#A-{int(_opd_ad["ads_idx"])}')

                    _opd_rc, _opd_dc = st.columns([8, 1])
                    with _opd_rc:
                        st.markdown(
                            f'<div class="opd-ad-row">'
                            f'{_opd_grade_badge_html(_opd_cur_label)}'
                            f'<span class="opd-name">{_opd_name}</span>'
                            f'<div>'
                            f'<div class="opd-metric-value">{_opd_val_str}{_opd_unit}</div>'
                            f'<div class="opd-metric-label">{_OPD_METRIC_NAME[_opd_mk]}</div>'
                            f'</div>'
                            f'{_opd_icon}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with _opd_dc:
                        if st.button(
                            "상세",
                            key=f"p2_opd_det_{int(_opd_ad['ads_idx'])}_{_opd_cur_mode}",
                            type="secondary",
                        ):
                            _prev = st.session_state.get("detail_ads_idx")
                            if _prev and _prev != int(_opd_ad["ads_idx"]):
                                st.session_state.pop(f"_detail_precomputed_{_prev}", None)
                                st.session_state.pop(f"_media_cvr_{_prev}", None)
                            st.session_state["detail_ads_idx"] = int(_opd_ad["ads_idx"])
                            st.session_state["detail_source"]  = "pages/02_유형별_운영_탐색.py"
                            st.session_state["detail_badge"]   = _opd_cur_label
                            st.session_state["detail_model"]   = (
                                "m2" if _opd_cur_mode in ("urgent", "early") else "m1"
                            )
                            st.switch_page("pages/05_광고_상세.py")
                    _opd_rendered += 1

            _opd_total = len(_opd_list_df)
            if _opd_rendered < _opd_total:
                if st.button(
                    f"+ 다음 5건 더보기 (총 {_opd_total}건 중 {_opd_rendered}건 표시)",
                    use_container_width=True, key="p2_opd_more",
                ):
                    st.session_state["p2_opd_show_n"] = _opd_show_n + 5
                    st.rerun()

# ════════════════════════════════════════════════════
# ML 인사이트 (기회발굴 + 손실방어)
# ════════════════════════════════════════════════════
with st.container(border=True):
    _p2_thr_key = st.session_state.get("ml_m2_threshold_p2", "best_f1")
    _p2_fp = f"p2|{period}|{sorted(selected_groups)}|{h_start}-{h_end}|{c_start}-{c_end}|{_p2_thr_key}"
    _p2_insight = get_cached_ml_insight_data(
        page_key="p2",
        fingerprint=_p2_fp,
        filtered=filtered,
        model_scores=model_scores,
        ad_master=ad_master,
        sched=sched,
        attr=attr,
        classification=classification,
        page_name="유형별 운영 탐색",
        threshold_key=_p2_thr_key,
        opp_top_n=1,
        risk_top_n=1,
        sched_agg_precomp=st.session_state.get("sched_agg"),
        early_click_precomp=st.session_state.get("early_click_df"),
        master_cols_precomp=st.session_state.get("master_cols"),
    )
    render_ml_insight(_p2_insight, page_key="p2")

# ── AI-agent 채팅 ──
from src.preprocessing import build_ad_summary as _build_ad_summary

def _build_chat_summary():
    summary = _build_ad_summary(filtered)
    if model_scores is not None:
        _ms_cols = [c for c in ['ads_idx', 'm1_score', 'm1_grade', 'm2_proba', 'm2_decision'] if c in model_scores.columns]
        summary = summary.merge(model_scores[_ms_cols], on='ads_idx', how='left')
    return summary

_p2_ad_summary = _cached_compute("p2_ad_summary", _filter_fp, _build_chat_summary)

render_chat_section(
    page_key="p2",
    ad_summary=_p2_ad_summary,
    kpis=kpis,
    page_name="유형별 운영 탐색",
    filters_desc=f"기간: {period} / 유형: {', '.join(selected_groups)} / 시간: {h_start}~{h_end}시",
)
