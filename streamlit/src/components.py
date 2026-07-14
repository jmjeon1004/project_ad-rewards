"""KPI 카드, 배너, 포맷 함수 등 공통 UI 컴포넌트"""
import html as _html
import colorsys
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from src.config import (COLORS, fmt_number, fmt_pct, fmt_currency, fmt_won_man,
                        ML_GRADE_COLORS, ML_GRADE_LABELS, RISK_COLORS, RISK_LABELS,
                        DEPENDENCY_RED, DEPENDENCY_YELLOW)
from src.charts import make_mini_sparkline, make_acost_rank_bar


def _desaturate(hex_color: str, factor: float = 0.5) -> str:
    """hex 색상의 채도를 factor만큼 낮춘 hex 색상을 반환한다 (0=무채색, 1=원본)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s * factor)
    return f"#{int(r2 * 255):02x}{int(g2 * 255):02x}{int(b2 * 255):02x}"

# page_key → 소스 페이지 파일 매핑 (상세 페이지에서 돌아가기용)
_PAGE_KEY_TO_FILE = {
    "p1": "pages/01_전체_광고_현황_오버뷰.py",
    "p2": "pages/02_유형별_운영_탐색.py",
    "p3": "pages/03_매체별_운영_탐색.py",
    "p4": "pages/04_카테고리별_운영_탐색.py",
}

# ---------------------------------------------------------------------------
# KPI 정의 정보 (ℹ 툴팁용)
# ---------------------------------------------------------------------------
KPI_INFO = {
    "총 클릭수": {
        "definition": "총 클릭수",
        "description": "선택 기간 내 광고가 받은 전체 클릭 횟수",
        "formula": "",
    },
    "평균 CVR": {
        "definition": "완료율 (Conversion Rate)",
        "description": "클릭 대비 전환(참여완료) 비율",
        "formula": "총 완료수 / 총 클릭수 × 100",
    },
    "평균 CPA": {
        "definition": "완료당 비용 (Cost Per Action)",
        "description": "완료 1건을 발생시키는 데 소요된 평균 광고비",
        "formula": "총 광고비 / 총 완료수",
    },
    "평균 마진율": {
        "definition": "마진율",
        "description": "광고비 대비 수익을 제외한 마진 비율",
        "formula": "(총 광고비 − 총 정산액) / 총 광고비 × 100",
    },
    "총 소진 광고비": {
        "definition": "소진 광고비",
        "description": "선택 기간 내 광고에 실제 집행된 총 광고비",
        "formula": "sum(acost)",
    },
    "총 완료수": {
        "definition": "총 완료수",
        "description": "선택 기간 내 완료된 전체 광고수",
        "formula": "",
    },
    "평일 vs 주말 CVR 격차": {
        "definition": "평일/주말 CVR 격차",
        "description": "평일과 주말의 완료율 차이",
        "formula": "평일 CVR − 주말 CVR (%p)",
    },
}

_KPI_TOOLTIP_CSS = """
<style>
.kpi-title-wrap{display:inline-flex;align-items:center;gap:5px;}
.kpi-info-icon{display:inline-flex;align-items:center;justify-content:center;
  width:16px;height:16px;border-radius:50%;background:#e0e0e0;color:#666;
  font-size:10px;font-weight:700;cursor:default;position:relative;flex-shrink:0;}
.kpi-info-icon .kpi-tooltip{visibility:hidden;opacity:0;position:absolute;
  left:0;top:22px;background:#333;
  color:#fff;font-size:11px;font-weight:400;line-height:1.6;padding:10px 12px;
  border-radius:8px;z-index:9999;white-space:normal;width:max-content;pointer-events:none;
  transition:opacity .15s;box-shadow:0 2px 8px rgba(0,0,0,.25);}
.kpi-info-icon .kpi-tooltip::before{content:'';position:absolute;
  top:-5px;left:6px;border-left:5px solid transparent;
  border-right:5px solid transparent;border-bottom:5px solid #333;}
.kpi-info-icon:hover .kpi-tooltip{visibility:visible;opacity:1;}
.kpi-tooltip b{color:#90caf9;}
</style>
"""


def get_kpi_title_html(title: str, color: str = "#666") -> str:
    """KPI 타이틀 + ℹ 툴팁 HTML을 반환한다. 외부 페이지에서도 import 가능."""
    info = KPI_INFO.get(title)
    if not info:
        return f'<div style="color:{color};font-size:14px;font-weight:500;margin-bottom:6px;">{_html.escape(title)}</div>'
    formula_line = (
        f"<div style='white-space:nowrap;'><b>계산식:</b> {_html.escape(info['formula'])}</div>"
        if info["formula"] else ""
    )
    tooltip_body = (
        f"<div style='white-space:nowrap;'><b>{_html.escape(info['definition'])}</b></div>"
        f"<div style='white-space:nowrap;'>{_html.escape(info['description'])}</div>"
        f"{formula_line}"
    )
    return (
        _KPI_TOOLTIP_CSS
        + f'<div style="color:{color};font-size:14px;font-weight:500;margin-bottom:6px;">'
        f'<span class="kpi-title-wrap">{_html.escape(title)}'
        f'<span class="kpi-info-icon">i<span class="kpi-tooltip">{tooltip_body}</span></span>'
        f'</span></div>'
    )


def get_heading_tooltip_html(
    title: str, tooltip_header: str, tooltip_lines: list[str | tuple[str, str]], heading_tag: str = "h3"
) -> str:
    """제목(헤딩 태그) + ⓘ 호버 툴팁 HTML (KPI 카드와 동일한 스타일: 굵은 헤더 + 본문 줄).

    tooltip_lines 각 항목은 문자열 또는 (강조 단어, 색상, 나머지 텍스트) 튜플.
    튜플인 경우 강조 단어만 색상이 적용되고 나머지 텍스트는 기본(흰색)으로 표시된다.
    """
    def _line_html(line):
        if isinstance(line, tuple):
            word, color, rest = line
            badge = (
                f"<span style='background:{color}; color:white; border-radius:4px; "
                f"padding:2px 10px; font-size:11px; font-weight:600;'>{_html.escape(word)}</span>"
            )
            return (
                f"<div style='white-space:nowrap; display:flex; align-items:center; gap:6px; margin:4px 0;'>"
                f"{badge}<span>{_html.escape(rest)}</span></div>"
            )
        return f"<div style='white-space:nowrap;'>{_html.escape(line)}</div>"

    body = (
        f"<div style='white-space:nowrap;'><b>{_html.escape(tooltip_header)}</b></div>"
        + "".join(_line_html(line) for line in tooltip_lines)
    )
    return (
        _KPI_TOOLTIP_CSS
        + f'<{heading_tag} style="display:flex; align-items:center; gap:6px;">{_html.escape(title)}&nbsp;&nbsp;'
        f'<span class="kpi-info-icon">i<span class="kpi-tooltip">{body}</span></span>'
        f'</{heading_tag}>'
    )


def render_kpi_card(
    title: str,
    value: str,
    change: float | None = None,
    change_unit: str = "%p",
    daily_df: pd.DataFrame | None = None,
    kpi_col: str = "cvr",
    avg_val: float | None = None,
    avg_label: str = "",
    invert_color: bool = False,
    period: str = "전체",
    agg_method: str = "mean",
    highlight_days: int = 0,
    unit: str = "",
    highlight_start=None,
    highlight_end=None,
    first_bar_color: str | None = None,
    last_bar_color: str | None = None,
):
    """KPI 카드 렌더링 (막대 그래프 포함, 통합 카드 디자인)

    Parameters
    ----------
    agg_method : "sum" for count metrics (clk), "mean" for rate metrics (cvr, margin_rate)
    highlight_days : 최근 N일을 강조색으로 표시 (0이면 전체 강조)
    unit : 값 뒤에 표시할 단위 (%, 원, x 등). 설정 시 value에서 단위를 분리해 작게 표시.
    first_bar_color : 지정 시 첫번째(가장 왼쪽) 막대 색상을 이 값으로 덮어씀
    last_bar_color : 지정 시 마지막(가장 오른쪽) 막대 색상을 이 값으로 덮어씀
    """
    # 증감 표시
    change_html = ""
    if change is not None and period != "전체":
        if change > 0:
            color = COLORS["negative"] if invert_color else COLORS["positive"]
            arrow = "▲"
            sign = "+"
        elif change < 0:
            color = COLORS["positive"] if invert_color else COLORS["negative"]
            arrow = "▼"
            sign = ""
        else:
            color = COLORS["neutral"]
            arrow = "─"
            sign = ""
        change_html = f'<span style="color:{color}; font-size:13px;">{sign}{change:.1f}{change_unit} {arrow}</span>'

    # 값과 단위 분리 표시
    if unit:
        display_value = value.replace(unit, "").strip()
        value_html = (
            f'<span style="font-size: 36px; font-weight: 700; color: #222;">{display_value}</span>'
            f'<span style="font-size: 20px; font-weight: 400; color: #222;">{unit}</span>'
        )
    else:
        value_html = f'<span style="font-size: 36px; font-weight: 700; color: #222;">{value}</span>'

    with st.container(border=True):
        title_html = get_kpi_title_html(title)
        st.markdown(f"""
        <div style="padding: 4px 4px 0 4px; text-align: left;">
            {title_html}
            <div>{value_html}</div>
            <div style="margin-top: 2px; min-height: 20px;">{change_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if daily_df is not None and len(daily_df) > 0:
            fig = _make_mini_bar(daily_df, kpi_col, avg_val, avg_label,
                                 agg_method=agg_method, highlight_days=highlight_days,
                                 highlight_start=highlight_start, highlight_end=highlight_end,
                                 first_bar_color=first_bar_color, last_bar_color=last_bar_color)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"kpi_{title}_{kpi_col}")


def _make_mini_bar(daily_df: pd.DataFrame, kpi_col: str, avg_val=None, avg_label="",
                   agg_method: str = "mean", highlight_days: int = 0,
                   highlight_start=None, highlight_end=None,
                   first_bar_color: str | None = None,
                   last_bar_color: str | None = None) -> go.Figure:
    """KPI 카드 하단 미니 막대 그래프 (30개 bin 고정)

    Parameters
    ----------
    agg_method : "sum" for count metrics (clk), "mean" for rate metrics (cvr, margin_rate)
    highlight_days : 최근 N일을 강조색으로 표시 (0이면 전체 강조)
    highlight_start, highlight_end : 사용자 지정 날짜 범위 — 설정 시 highlight_days 대신 사용
    """
    NUM_BINS = 30
    df = daily_df.sort_values("rpt_time_date").copy()
    n = len(df)

    # 사용자 지정 날짜 범위로 강조할 행 인덱스 집합 계산
    use_date_range = highlight_start is not None and highlight_end is not None
    if use_date_range:
        hl_s = pd.Timestamp(highlight_start)
        hl_e = pd.Timestamp(highlight_end)

    if n > NUM_BINS:
        bin_size = n / NUM_BINS
        df = df.reset_index(drop=True)
        df["bin"] = np.minimum((df.index / bin_size).astype(int), NUM_BINS - 1)
        df_binned = df.groupby("bin").agg({kpi_col: agg_method}).reset_index()
        vals = df_binned[kpi_col].fillna(0).values[:NUM_BINS]
        pad = 0

        if use_date_range:
            # 날짜 범위에 속하는 행들의 bin 범위 계산
            mask = (df["rpt_time_date"] >= hl_s) & (df["rpt_time_date"] <= hl_e)
            hl_rows = df[mask]
            if len(hl_rows) > 0:
                hl_bin_start = int(hl_rows["bin"].min())
                hl_bin_end = int(hl_rows["bin"].max())
            else:
                hl_bin_start = hl_bin_end = -1  # 해당 없음
        elif highlight_days > 0:
            first_hl_row = max(n - highlight_days, 0)
            first_hl_bin = int(np.minimum(int(first_hl_row / bin_size), NUM_BINS - 1))
            hl_bin_start = first_hl_bin
            hl_bin_end = NUM_BINS - 1
        else:
            hl_bin_start = 0
            hl_bin_end = NUM_BINS - 1
    else:
        raw_vals = df[kpi_col].fillna(0).values
        pad = NUM_BINS - len(raw_vals)
        vals = np.concatenate([np.full(pad, np.nan), raw_vals])

        if use_date_range:
            # 날짜 범위에 속하는 행의 원본 인덱스 → 우측 정렬 bar 인덱스
            df = df.reset_index(drop=True)
            mask = (df["rpt_time_date"] >= hl_s) & (df["rpt_time_date"] <= hl_e)
            hl_indices = df[mask].index
            if len(hl_indices) > 0:
                hl_bin_start = pad + int(hl_indices.min())
                hl_bin_end = pad + int(hl_indices.max())
            else:
                hl_bin_start = hl_bin_end = -1
        elif highlight_days > 0:
            hl_bin_start = NUM_BINS - min(highlight_days, len(raw_vals))
            hl_bin_end = NUM_BINS - 1
        else:
            hl_bin_start = pad
            hl_bin_end = NUM_BINS - 1

    # 막대 색상: 강조 vs 비강조
    colors = []
    for i in range(NUM_BINS):
        if np.isnan(vals[i]) if isinstance(vals[i], float) else False:
            colors.append("rgba(0,0,0,0)")
        elif hl_bin_start <= i <= hl_bin_end:
            colors.append(COLORS["bar_highlight"])
        else:
            colors.append(COLORS["bar_dim"])

    if first_bar_color is not None:
        for i in range(NUM_BINS):
            if colors[i] != "rgba(0,0,0,0)":
                colors[i] = first_bar_color
                break

    if last_bar_color is not None:
        for i in range(NUM_BINS - 1, -1, -1):
            if colors[i] != "rgba(0,0,0,0)":
                colors[i] = last_bar_color
                break

    x = list(range(NUM_BINS))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=vals,
        marker_color=colors,
        marker_line_width=0,
        showlegend=False,
        hoverinfo="skip",
    ))

    # 평균 기준선 (진한 점선, 라벨 없음)
    if avg_val is not None:
        fig.add_hline(
            y=avg_val, line_dash="dot", line_color=COLORS["avg_line"], line_width=3,
        )

    fig.update_layout(
        height=80, margin=dict(l=0, r=0, t=4, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.15,
    )
    return fig


def render_status_banner(normal_pct: float, urgent: int, caution: int,
                         model_coverage_pct: float = None, unscored_count: int = 0):
    """운영 상태 배너 — 품질 점수 등급 기반 (S/A/B 정상, C 주의, D 긴급)"""
    if normal_pct >= 80:
        bg = "#E8F5E9"
        icon = "✓"
        border_c = "#A5D6A7"
        label = "정상 운영 중"
    elif normal_pct >= 60:
        bg = "#FFF8E1"
        icon = "⚠"
        border_c = "#FFE082"
        label = "검토 권장"
    else:
        bg = "#FFEBEE"
        icon = "🚨"
        border_c = "#EF9A9A"
        label = "즉시 검토 필요"

    sub_html = (
        f'<div style="font-size: 12px; color: #888; margin-top: 2px;">'
        f'ML 평가 대상 기준 · 미평가 {unscored_count}건 제외'
    )
    if model_coverage_pct is not None:
        sub_html += f' (커버리지 {model_coverage_pct:.0f}%)'
    sub_html += '</div>'

    st.markdown(f"""
    <div style="
        background: {bg}; border: 1px solid {border_c}; border-radius: 10px;
        padding: 14px 20px; display: flex; align-items: center; gap: 16px;
    ">
        <span style="font-size: 20px;">{icon}</span>
        <div>
            <div style="font-weight: 600; font-size: 15px;">{label} — 긴급 {urgent} · 주의 {caution}</div>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dependency_banner(dep_pct: float):
    """TOP 10 의존도 경고 배너"""
    if dep_pct >= DEPENDENCY_RED:
        bg = "#FFEBEE"
        border_c = "#EF9A9A"
        icon = "⚠"
    elif dep_pct >= DEPENDENCY_YELLOW:
        bg = "#FFF8E1"
        border_c = "#FFE082"
        icon = "⚠"
    else:
        bg = "#E8F5E9"
        border_c = "#A5D6A7"
        icon = "🟢"

    st.markdown(f"""
    <div style="
        background: {bg}; border: 1px solid {border_c}; border-radius: 10px;
        padding: 14px 20px;
    ">
        <div style="font-weight: 600; font-size: 15px;">{icon} TOP 10 의존도 {dep_pct:.0f}%</div>
        <div style="font-size: 12px; color: #666; margin-top: 2px;">
            완료수 상위 10개 광고의 완료 점유율
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_context_message(period: str, has_custom_date: bool = False,
                           data_date_min: pd.Timestamp | None = None,
                           data_date_max: pd.Timestamp | None = None,
                           custom_start=None, custom_end=None):
    """컨텍스트 안내 메시지 — 선택 기간 대비 이전 구간 데이터 충분 여부 판단"""
    if period == "전체":
        st.info("ℹ 전체 기간에서는 이전 기간 대비 비교가 비활성화됩니다.")
        return
    elif period == "사용자 지정" and not has_custom_date:
        st.warning("⚠ 사용자 지정 기간을 달력에서 선택해주세요.")
        return

    # 선택 기간 일수 계산
    if period == "최근 1일":
        selected_days = 1
    elif period == "최근 7일":
        selected_days = 7
    elif period == "사용자 지정" and custom_start and custom_end:
        selected_days = (pd.Timestamp(custom_end) - pd.Timestamp(custom_start)).days + 1
    else:
        selected_days = 0

    # 이전 비교 구간에 필요한 데이터 존재 여부 확인
    if selected_days > 0 and data_date_min is not None and data_date_max is not None:
        # 현재 기간 시작일 계산
        if period == "최근 1일":
            cur_start = data_date_max
        elif period == "최근 7일":
            cur_start = data_date_max - pd.Timedelta(days=6)
        elif period == "사용자 지정":
            cur_start = pd.Timestamp(custom_start)
        else:
            cur_start = data_date_min

        # 이전 비교 구간: cur_start 이전으로 selected_days만큼 필요
        available_prev_days = (cur_start - data_date_min).days

        if available_prev_days >= selected_days:
            st.success("✅ 비교 기간이 충분히 확보되었습니다. 신뢰도 있는 비교가 가능합니다.")
        else:
            st.error(
                f"🚨 선택 기간({selected_days}일)에 비해 비교 가능 기간이 부족합니다. "
                f"더 짧은 기간을 선택하거나 '전체 기간'을 선택해주세요. "
                f"비교 차트는 비활성화되었습니다."
            )
    else:
        st.success("✅ 비교 기간이 충분히 확보되었습니다. 신뢰도 있는 비교가 가능합니다.")


def render_top3_card(title: str, count: int, items: list[dict], page_link: str, badge: str = "정상",
                     button_label: str = ""):
    """차원별 TOP 3 카드 — 프로그레스 바 + 점유율 포함"""
    # 배지 색상
    badge_colors = {"정상": COLORS["positive"], "주의": COLORS["warning"], "위험": COLORS["negative"]}
    bc = badge_colors.get(badge, COLORS["neutral"])
    badge_html = (
        f'<span style="background:{bc}; color:white; border-radius:4px; '
        f'padding:2px 10px; font-size:11px; font-weight:600;">{badge}</span>'
    )

    rows_html = ""
    for i, item in enumerate(items[:3], 1):
        cvr = item.get("cvr", 0)
        gap = item.get("gap", 0)
        clk = item.get("clk", 0)
        share = item.get("share", 0)

        if gap > 0:
            gap_html = f'<span style="color:{COLORS["positive"]}; font-size:13px; font-weight:600;">▲ +{gap:.1f}%p</span>'
        elif gap < 0:
            gap_html = f'<span style="color:{COLORS["negative"]}; font-size:13px; font-weight:600;">▼ {gap:.1f}%p</span>'
        else:
            gap_html = f'<span style="color:{COLORS["neutral"]}; font-size:13px;">— {gap:.1f}%p</span>'

        # 프로그레스 바 너비: CVR 기준 (최대 100%)
        bar_width = min(cvr if pd.notna(cvr) else 0, 100)

        rows_html += f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 14px; color: #222;">{i}. {item['name']}</span>
                <span style="font-size: 14px;">
                    <span style="font-weight: 700; color: #222;">{fmt_pct(cvr)}</span>
                    &nbsp;{gap_html}
                </span>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #222222; border-radius: 4px; height: 8px; margin-bottom: 4px;">
                <div style="background: {COLORS['weekend']}; border-radius: 3px 0 0 3px; height: 100%; width: {bar_width}%;"></div>
            </div>
            <div style="font-size: 12px; color: #999;">
                클릭 {fmt_number(clk)} · 점유율 {share:.0f}%
            </div>
        </div>
        """

    # button_label이 있으면 아래에 st.button이 연결되므로 하단 border 제거
    if button_label:
        border_radius = "12px 12px 0 0"
        border_css = f"border: 1px solid {COLORS['border']}; border-bottom: none;"
    else:
        border_radius = "12px"
        border_css = f"border: 1px solid {COLORS['border']};"

    st.html(f"""
    <div style="
        background: #FFFFFF; {border_css}
        border-radius: {border_radius}; padding: 18px 20px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;
                    margin-bottom: 14px;">
            <span style="font-weight: 700; font-size: 14px; color: #222;">
                {title}
                <span style="font-weight: 400; color: #888; font-size: 12px; margin-left: 4px;">
                    전체 {count}개 중 상위 3개
                </span>
            </span>
            {badge_html}
        </div>
        {rows_html}
    </div>
    """)


def render_alert_card(icon: str, title: str, items: list[str], color: str, detail: str = ""):
    """운영 알림 카드"""
    color_map = {
        "red": ("#FFEBEE", "#C62828"),
        "yellow": ("#FFF8E1", "#F57F17"),
        "green": ("#E8F5E9", "#2E7D32"),
        "blue": ("#E3F2FD", "#1565C0"),
    }
    bg, text_c = color_map.get(color, ("#F5F5F5", "#333"))

    if len(items) > 3:
        half = (len(items) + 1) // 2
        col1 = items[:half]
        col2 = items[half:]
        items_html = (
            '<div style="display:flex; gap:8px; margin-top:8px;">'
            '<div style="flex:1; font-size:12px; color:#444;">'
            + "<br>".join(col1)
            + '</div><div style="flex:1; font-size:12px; color:#444;">'
            + "<br>".join(col2)
            + '</div></div>'
        )
    else:
        items_html = (
            f'<div style="font-size:12px; margin-top:8px; color:#444;">'
            + "<br>".join(items)
            + '</div>'
        )

    st.markdown(f"""
    <div style="
        background: {bg}; border-radius: 10px; padding: 14px;
        border-left: 4px solid {text_c}; height: 160px;
        display: flex; flex-direction: column;
    ">
        <div style="font-weight: 600; color: {text_c}; font-size: 13px;">
            {icon} {title}
        </div>
        {items_html}
        <div style="font-size: 11px; margin-top: auto; color: #777;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


def render_media_highlight_card(icon: str, title: str, media_name: str, color: str,
                                 mode: str = "trend",
                                 quote: str | None = None,
                                 daily_df: pd.DataFrame | None = None, kpi_col: str | None = None,
                                 badge_text: str | None = None, button_label: str | None = None,
                                 period_label: str | None = None,
                                 group_kpis: pd.DataFrame | None = None, group_col: str | None = None,
                                 spend_insight: str | None = None, spend_action: str | None = None):
    """매체 인사이트 하이라이트 카드 (최우수/비효율 공용).

    mode="trend": 일별 스파크라인 + 안정성 배지 + CTA.
    mode="spend": 추이 데이터가 부족한 매체용 대체 뷰 — 전체 매체 광고비 순위 차트 + 인사이트.
    """
    theme_map = {
        "green": dict(border="#A5D6A7", title_c="#2E7D32", line="#2E7D32",
                      badge_bg="#E8F5E9", badge_c="#2E7D32",
                      btn_bg="#FFFFFF", btn_border="#2E7D32", btn_c="#2E7D32"),
        "red": dict(border="#EF9A9A", title_c="#C62828", line="#C62828",
                    badge_bg="#FFEBEE", badge_c="#C62828",
                    btn_bg="#C62828", btn_border="#C62828", btn_c="#FFFFFF"),
    }
    t = theme_map.get(color, theme_map["green"])
    card_key = f"media_card_{abs(hash((title, media_name, kpi_col)))}"

    st.markdown(f"""
    <style>
    div.st-key-{card_key} {{
        border: 1.5px solid {t['border']} !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container(border=True, key=card_key):
        subtitle_html = (
            f'<div style="font-size: 12px; color: #777; margin-top: 4px;">{period_label}</div>'
            if mode != "trend" else ""
        )
        st.markdown(f"""
        <div style="font-weight: 700; font-size: 14px; color: {t['title_c']};">{icon} {title}</div>
        {subtitle_html}
        """, unsafe_allow_html=True)

        if mode == "trend":
            n_valid = int(daily_df[kpi_col].notna().sum()) if daily_df is not None else 0
            st.markdown(
                "<div style='font-size:11px; color:#999; margin:8px 0 2px 0;'>마진율 추이</div>",
                unsafe_allow_html=True,
            )
            if n_valid >= 2:
                fig = make_mini_sparkline(daily_df, kpi_col, t["line"])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"sparkline_{title}_{kpi_col}")
            else:
                st.markdown(
                    "<div style='height:170px; display:flex; align-items:center; justify-content:center;"
                    "color:#999; font-size:12px;'>일별 추이를 표시할 데이터가 부족합니다</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"""
            <div style="min-height: 96px; box-sizing: border-box; display: flex;
                        flex-direction: column; justify-content: center; margin-top: 8px;">
                <div style="background: {t['badge_bg']}; color: {t['badge_c']}; font-size: 12px; font-weight: 600;
                            border-radius: 8px; padding: 8px 12px; text-align: center;">{badge_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-size:11px; color:#999; margin:8px 0 2px 0;'>전체 매체 광고비 대비 위치</div>",
                unsafe_allow_html=True,
            )
            fig = make_acost_rank_bar(group_kpis, group_col, media_name, t["line"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"acost_rank_{title}")
            st.markdown(f"""
            <div style="min-height: 96px; box-sizing: border-box; display: flex;
                        flex-direction: column; justify-content: center; margin-top: 8px;">
                <div style="background: #FFF8E1; border-radius: 8px; padding: 10px 12px;
                            border-left: 4px solid #F57F17;">
                    <div style="font-size: 12px; font-weight: 700; color: #E65100;">💡 {spend_insight}</div>
                    <div style="font-size: 11px; color: #555; margin-top: 4px;">{spend_action}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_media_price_coop_card(title: str, subtitle: str, rows: pd.DataFrame):
    """단가 협업 카드 — 마진율 평균비 기준 매체 비교 테이블."""
    rows_html = ""
    n = len(rows)
    for i, r in rows.iterrows():
        ratio = r["avg_ratio"]
        if pd.isna(ratio):
            dot, highlight = "#9CA3AF", False
        elif i == 0:
            dot, highlight = "#2E7D32", False
        elif i == n - 1:
            dot, highlight = "#F57F17", True
        else:
            dot, highlight = "#9CA3AF", False
        row_bg = "background:#FFF3E0;" if highlight else ""
        ratio_c = "#2E7D32" if pd.notna(ratio) and ratio >= 0 else "#E65100"
        ratio_txt = f"{ratio:+.0f}%" if pd.notna(ratio) else "-"
        rows_html += (
            f'<div style="display:flex; align-items:center; padding:10px 4px; {row_bg}'
            f'border-bottom:1px solid #F0EEE8; font-size:13px;">'
            f'<div style="flex:2; color:#333;">{r["media"]}</div>'
            f'<div style="flex:1.5; color:#333;">{fmt_pct(r["margin_rate"])}</div>'
            f'<div style="flex:1.5; color:#333;">{fmt_currency(r["cpa"])}</div>'
            f'<div style="flex:1.5; color:{ratio_c}; font-weight:600;">{ratio_txt}</div>'
            f'<div style="flex:0.4; text-align:right;">'
            f'<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{dot};"></span>'
            f'</div></div>'
        )

    st.markdown(f"""
    <div style="border: 1.5px solid {COLORS['border']}; border-radius: 12px; padding: 16px 18px;">
        <div style="font-size:12px; color:#777;">
            "평균 대비 뭐가 아쉬운가, 어디부터 손볼까" — 단일 수치보다 그룹 내 상대 비교가 먼저 필요
        </div>
        <div style="display:flex; padding:8px 4px 6px 4px; font-size:11px; color:#999;
                    border-bottom:1px solid #E8E4DC;">
            <div style="flex:2;">매체</div><div style="flex:1.5;">마진율</div>
            <div style="flex:1.5;">CPA</div><div style="flex:1.5;">평균비</div><div style="flex:0.4;"></div>
        </div>
        {rows_html}
        <div style="display:flex; gap:10px; margin-top:14px;">
            <div style="flex:1; border:1.5px solid {COLORS['border']}; border-radius:8px; padding:8px 12px;
                        text-align:center; font-size:13px; font-weight:600; color:#333;">우선순위 정렬 ↗</div>
            <div style="flex:1; border:1.5px solid {COLORS['border']}; border-radius:8px; padding:8px 12px;
                        text-align:center; font-size:13px; font-weight:600; color:#333;">개별 매체 드릴다운 ↗</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_media_material_card(media_name: str, quote: str, margin_val: float, margin_caption: str,
                                cvr_val: float, cvr_caption: str, insight_text: str, button_label: str,
                                insight_action: str | None = None):
    """소재 개선 카드 — 마진율/CVR 비교 KPI + 인사이트 텍스트 + CTA."""
    st.markdown(f"""
    <div style="border: 1.5px solid #B39DDB; border-radius: 12px; padding: 16px 18px;">
        <div style="font-weight:700; font-size:14px; color:#5E35B1;">☼ 소재 개선·{media_name}</div>
        <div style="display:flex; gap:16px; margin-top:14px;">
            <div style="flex:1; text-align:center; background:#F5F3FB; border-radius:10px; padding:14px 8px;">
                <div style="font-size:12px; color:#888;">마진율</div>
                <div style="font-size:26px; font-weight:700; color:#5E35B1;">{fmt_pct(margin_val)}</div>
                <div style="font-size:11px; color:#2E7D32; margin-top:2px;">{margin_caption}</div>
            </div>
            <div style="flex:1; text-align:center; background:#F5F3FB; border-radius:10px; padding:14px 8px;">
                <div style="font-size:12px; color:#888;">CVR</div>
                <div style="font-size:26px; font-weight:700; color:#5E35B1;">{fmt_pct(cvr_val)}</div>
                <div style="font-size:11px; color:#E65100; margin-top:2px;">{cvr_caption}</div>
            </div>
        </div>
        <div style="background:#F5F3FB; border-radius:8px; padding:10px 12px; margin-top:12px;
                    border-left: 4px solid #7C4DFF;">
            <div style="font-size:12px; font-weight:700; color:#4527A0;">💡 {insight_text}</div>
            {f'<div style="font-size:11px; color:#555; margin-top:4px;">{insight_action}</div>' if insight_action else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ad_list_table(ads_df: pd.DataFrame, sort_col: str = "cvr",
                          ascending: bool = False, max_rows: int = 10):
    """광고 리스트 테이블 렌더링"""
    if ads_df.empty:
        st.info("해당 조건의 광고가 없습니다.")
        return

    display_df = ads_df.sort_values(sort_col, ascending=ascending).head(max_rows).copy()

    # 표시용 컬럼 정리
    col_map = {
        "ads_name": "광고명",
        "analysis_ads_type_label": "광고 유형",
        "final_media": "매체",
        "clk": "클릭수",
        "turn": "완료수",
        "cvr": "CVR",
        "cpa": "CPA",
    }

    available = [c for c in col_map if c in display_df.columns]
    show_df = display_df[available].rename(columns=col_map)

    # 첫 번째 열에 순위 컬럼 삽입
    show_df.insert(0, "순위", range(1, len(show_df) + 1))

    if "CVR" in show_df.columns:
        show_df["CVR"] = show_df["CVR"].apply(lambda x: fmt_pct(x) if pd.notna(x) else "-")
    if "CPA" in show_df.columns:
        show_df["CPA"] = show_df["CPA"].apply(lambda x: fmt_currency(x) if pd.notna(x) else "-")
    if "클릭수" in show_df.columns:
        show_df["클릭수"] = show_df["클릭수"].apply(lambda x: f"{x:,}")
    if "완료수" in show_df.columns:
        show_df["완료수"] = show_df["완료수"].apply(lambda x: f"{x:,}")

    # HTML 테이블 (헤더 행 강조)
    html = '<div style="overflow-x:auto;">'
    html += '<table style="width:100%; border-collapse:collapse; font-size:12px;">'
    html += '<tr>'
    for col in show_df.columns:
        html += (
            f'<th style="background:#264653; color:#fff; padding:8px 6px;'
            f' text-align:center; font-weight:600; white-space:nowrap;">{col}</th>'
        )
    html += '</tr>'
    for i, (_, row) in enumerate(show_df.iterrows()):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        html += f'<tr style="background:{bg};">'
        for col in show_df.columns:
            val = row[col]
            align = "center" if col == "순위" else "left" if col == "광고명" else "right"
            html += (
                f'<td style="padding:6px; text-align:{align};'
                f' border-bottom:1px solid #eee; white-space:nowrap;">{val}</td>'
            )
        html += '</tr>'
    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# ML 모델 관련 컴포넌트
# =====================================================================

def _grade_badge_html(grade: str) -> str:
    """S/A/B/C/D 등급 배지 HTML (인라인)."""
    color = ML_GRADE_COLORS.get(grade, "#757575")
    return (
        f'<span style="display:inline-block; padding:2px 8px; border-radius:4px;'
        f' background:{color}; color:#fff; font-size:11px; font-weight:700;'
        f' letter-spacing:0.5px;">{grade}</span>'
    )


def _risk_badge_html(decision: str) -> str:
    """부진위험/정상/판단보류 배지 HTML (인라인)."""
    color = RISK_COLORS.get(decision, "#757575")
    label = RISK_LABELS.get(decision, decision)
    return (
        f'<span style="display:inline-block; padding:2px 8px; border-radius:4px;'
        f' background:{color}; color:#fff; font-size:11px; font-weight:600;'
        f' white-space:nowrap;">'
        f'{label}</span>'
    )


def render_grade_distribution(scores_df: pd.DataFrame):
    """등급 분포 가로 막대 렌더링."""
    if scores_df.empty or 'm1_grade' not in scores_df.columns:
        return

    grade_order = ['S', 'A', 'B', 'C', 'D']
    counts = scores_df['m1_grade'].value_counts()
    total = counts.sum()

    bars_html = ""
    for g in grade_order:
        cnt = counts.get(g, 0)
        pct = cnt / total * 100 if total > 0 else 0
        color = ML_GRADE_COLORS.get(g, "#757575")
        if pct > 0:
            bars_html += (
                f'<div style="display:inline-block; width:{pct}%; background:{color};'
                f' height:24px; position:relative;" title="{g}: {cnt}건 ({pct:.1f}%)">'
                f'<span style="color:#fff; font-size:10px; font-weight:700;'
                f' position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);">'
                f'{g} {cnt}</span></div>'
            )

    legend_html = " ".join(
        f'<span style="font-size:11px;">'
        f'<span style="display:inline-block; width:10px; height:10px; border-radius:2px;'
        f' background:{ML_GRADE_COLORS[g]}; margin-right:3px;"></span>'
        f'{g}({ML_GRADE_LABELS[g]}) {counts.get(g, 0)}건</span>'
        for g in grade_order
    )

    st.markdown(f"""
    <div style="margin:8px 0;">
        <div style="font-size:13px; font-weight:600; margin-bottom:6px;">
            ML 품질 등급 분포 <span style="font-weight:400; color:#888;">({total}건)</span>
        </div>
        <div style="display:flex; width:100%; border-radius:6px; overflow:hidden;
                    height:24px; background:#eee;">{bars_html}</div>
        <div style="margin-top:4px;">{legend_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_summary(scores_df: pd.DataFrame):
    """조기부진 위험 요약 배너."""
    if scores_df.empty or 'm2_decision' not in scores_df.columns:
        return

    counts = scores_df['m2_decision'].value_counts()
    decline = counts.get('decline_risk', 0)
    normal = counts.get('normal', 0)
    review = counts.get('rule_based_review', 0)
    total = decline + normal + review

    if total == 0:
        return

    decline_pct = decline / total * 100

    if decline > 0:
        border_color = RISK_COLORS['decline_risk']
        icon = "&#9888;"
    else:
        border_color = RISK_COLORS['normal']
        icon = "&#10003;"

    st.markdown(f"""
    <div style="
        background:#fff; border-left:4px solid {border_color};
        padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0;
        box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:14px; font-weight:600; margin-bottom:6px;">
            {icon} 조기부진 예측 (D+3)
        </div>
        <div style="font-size:12px; color:#555;">
            <span style="color:{RISK_COLORS['decline_risk']}; font-weight:700;">
                부진위험 {decline}건</span> ({decline_pct:.1f}%)
            &nbsp;·&nbsp; 정상 {normal}건
            &nbsp;·&nbsp; 판단보류 {review}건
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ad_list_table_with_scores(
    ads_df: pd.DataFrame,
    scores_df: pd.DataFrame | None = None,
    sort_col: str = "cvr",
    ascending: bool = False,
    max_rows: int = 10,
    max_height: int | None = None,
    hide_cols: tuple[str, ...] = (),
):
    """모델 등급/위험도 포함 광고 리스트 테이블."""
    if ads_df.empty:
        st.info("해당 조건의 광고가 없습니다.")
        return

    df = ads_df.copy()
    if scores_df is not None and not scores_df.empty:
        df = df.merge(scores_df[['ads_idx', 'm1_grade', 'm2_decision']],
                       on='ads_idx', how='left')

    display_df = df.sort_values(sort_col, ascending=ascending).head(max_rows).copy()

    show_ads_type = "광고 유형" not in hide_cols
    show_media = "매체" not in hide_cols

    # HTML 테이블
    _height_style = f"max-height:{max_height}px; overflow-y:auto;" if max_height else ""
    html = f'<div style="{_height_style}">'
    html += '<table style="width:100%; border-collapse:collapse; font-size:11px; table-layout:auto;">'

    # 헤더
    cols = [("순위", "center"), ("광고명", "left")]
    if show_ads_type:
        cols.append(("광고 유형", "center"))
    if show_media:
        cols.append(("매체", "center"))
    cols += [("클릭수", "right"), ("완료수", "right"), ("CVR", "right"), ("CPA", "right")]
    if 'm1_grade' in display_df.columns:
        cols.append(("등급", "center"))
    if 'm2_decision' in display_df.columns:
        cols.append(("부진예측", "center"))

    html += '<tr>'
    for col_name, _ in cols:
        html += (
            f'<th style="background:#264653; color:#fff; padding:6px 4px;'
            f' text-align:center; font-weight:600; white-space:nowrap;">{col_name}</th>'
        )
    html += '</tr>'

    # 데이터 행
    for i, (_, row) in enumerate(display_df.iterrows()):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        html += f'<tr style="background:{bg};">'

        # 순위
        html += f'<td style="padding:6px; text-align:center; border-bottom:1px solid #eee; white-space:nowrap;">{i+1}</td>'
        # 광고명
        name = row.get('ads_name', '-')
        html += f'<td style="padding:6px 4px; text-align:left; border-bottom:1px solid #eee; word-break:break-all;">{name}</td>'
        # 광고 유형
        if show_ads_type:
            html += f'<td style="padding:6px; text-align:center; border-bottom:1px solid #eee; white-space:nowrap;">{row.get("analysis_ads_type_label", "-")}</td>'
        # 매체
        if show_media:
            html += f'<td style="padding:6px; text-align:center; border-bottom:1px solid #eee; white-space:nowrap;">{row.get("final_media", "-")}</td>'
        # 클릭수
        clk = row.get('clk', 0)
        html += f'<td style="padding:6px; text-align:right; border-bottom:1px solid #eee; white-space:nowrap;">{clk:,.0f}</td>'
        # 완료수
        turn = row.get('turn', 0)
        html += f'<td style="padding:6px; text-align:right; border-bottom:1px solid #eee; white-space:nowrap;">{turn:,.0f}</td>'
        # CVR
        cvr = row.get('cvr', None)
        cvr_str = fmt_pct(cvr) if pd.notna(cvr) else "-"
        html += f'<td style="padding:6px; text-align:right; border-bottom:1px solid #eee; white-space:nowrap;">{cvr_str}</td>'
        # CPA
        cpa = row.get('cpa', None)
        cpa_str = fmt_currency(cpa) if pd.notna(cpa) else "-"
        html += f'<td style="padding:6px; text-align:right; border-bottom:1px solid #eee; white-space:nowrap;">{cpa_str}</td>'
        # 등급
        if 'm1_grade' in display_df.columns:
            grade = row.get('m1_grade', '')
            badge = _grade_badge_html(grade) if pd.notna(grade) and grade else '-'
            html += f'<td style="padding:6px; text-align:center; border-bottom:1px solid #eee; white-space:nowrap;">{badge}</td>'

        # 부진예측
        if 'm2_decision' in display_df.columns:
            dec = row.get('m2_decision', '')
            badge = _risk_badge_html(dec) if pd.notna(dec) and dec else '-'
            html += f'<td style="padding:6px; text-align:center; border-bottom:1px solid #eee; white-space:nowrap;">{badge}</td>'

        html += '</tr>'

    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 그룹(매체/유형/카테고리)별 볼륨·효율·수익성 요약 테이블
# ════════════════════════════════════════════════════

def render_media_summary_table(
    group_kpis: pd.DataFrame,
    group_col: str,
    ads_df: pd.DataFrame | None = None,
    model_scores: pd.DataFrame | None = None,
    group_label: str = "매체",
    widget_key_prefix: str = "summary",
):
    """그룹별 볼륨(완료수)·효율(CVR/CPA)·수익성(마진율/정산) 요약 테이블 + 인사이트."""
    if group_kpis.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    max_margin = group_kpis["margin_rate"].clip(lower=0).max()
    max_margin = max_margin if pd.notna(max_margin) and max_margin > 0 else 1.0

    with st.expander(f"{group_label}별 성과 요약"):
        sort_key = st.pills(
            "정렬 기준", ["완료수순", "마진율순", "CPA순"],
            selection_mode="single", default="완료수순",
            key=f"{widget_key_prefix}_sort", label_visibility="collapsed",
        ) or "완료수순"

        sort_map = {
            "완료수순": ("turn", False),
            "마진율순": ("margin_rate", False),
            "CPA순": ("cpa", True),
        }
        sort_col, ascending = sort_map.get(sort_key, ("turn", False))
        df = group_kpis.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

        cols = [(group_label, "left"), ("완료수", "right"), ("CVR", "right"), ("CPA", "right"),
                ("광고비", "right"), ("마진율", "left")]

        html = '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
        html += '<tr>'
        for name, align in cols:
            html += (f'<th style="padding:8px 10px; text-align:{align}; border-bottom:2px solid #E8E4DC;'
                      f' color:#555; font-weight:600;">{name}</th>')
        html += '</tr>'

        for _, row in df.iterrows():
            margin_rate = row.get("margin_rate")
            bar_pct = min(100, max(0, margin_rate / max_margin * 100)) if pd.notna(margin_rate) else 0

            html += '<tr style="border-bottom:1px solid #F0EEE8;">'
            html += f'<td style="padding:8px 10px; font-weight:600;">{row[group_col]}</td>'
            html += f'<td style="padding:8px 10px; text-align:right;">{fmt_number(row.get("turn"))}</td>'
            html += f'<td style="padding:8px 10px; text-align:right;">{fmt_pct(row.get("cvr")) if pd.notna(row.get("cvr")) else "-"}</td>'
            html += f'<td style="padding:8px 10px; text-align:right;">{fmt_currency(row.get("cpa")) if pd.notna(row.get("cpa")) else "-"}</td>'
            html += f'<td style="padding:8px 10px; text-align:right;">{fmt_won_man(row.get("acost"))}</td>'
            html += (
                '<td style="padding:8px 10px;">'
                '<div style="display:flex; align-items:center; gap:8px;">'
                f'<div style="flex:1; max-width:120px; background:#EEEAE2; border-radius:4px; height:10px;">'
                f'<div style="width:{bar_pct:.0f}%; background:{COLORS["primary"]}; height:10px; border-radius:4px;"></div></div>'
                f'<span style="white-space:nowrap; font-size:12px;">'
                f'{fmt_pct(margin_rate) if pd.notna(margin_rate) else "-"}</span>'
                '</div></td>'
            )
            html += '</tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 모델 결과 기반 운영 추천 배너 (전체 기간 모드, P1 전용)
# ════════════════════════════════════════════════════

def render_ml_recommendation_banner(insight_data: dict, page_key: str):
    """
    "모델 결과 기반 운영 추천" 통합 배너.
    품질등급(기회) + 조기부진(위험) 배지 4종 카운트를 한 줄에 표시하고,
    각 배지별 대표 광고 1건씩(최대 4건)을 하나의 리스트로 통합 렌더링.
    """
    from src.config import ML_GRADE_COLORS, OPPORTUNITY_BADGE_COLORS, RISK_ACTION_BADGE_COLORS

    d = insight_data

    def _pill(text: str, color: str) -> str:
        return (
            f'<span style="display:inline-block; background:{color}; color:#fff; padding:2px 10px;'
            f' border-radius:4px; font-size:11px; font-weight:700; white-space:nowrap;">{text}</span>'
        )

    _tooltip_body = (
        "<div style='white-space:nowrap;'><b>선별 기준</b></div>"
        "<div style='white-space:nowrap; margin:4px 0 10px 0;'>"
        "광고 품질 등급과 조기부진 예측 모델의 결과를 종합해 도출한 운영 추천</div>"
        "<div style='display:flex; align-items:center; gap:6px; margin:4px 0; white-space:nowrap;'>"
        f"{_pill('즉시조치', RISK_ACTION_BADGE_COLORS.get('즉시조치', '#C62828'))}"
        f"{_pill('우선검토', RISK_ACTION_BADGE_COLORS.get('우선검토', '#E65100'))}"
        "<span>: 조기부진 예측 모델</span></div>"
        "<div style='display:flex; align-items:center; gap:6px; margin:4px 0; white-space:nowrap;'>"
        f"{_pill('매체확장', OPPORTUNITY_BADGE_COLORS.get('매체확장', '#1565C0'))}"
        f"{_pill('승급추진', OPPORTUNITY_BADGE_COLORS.get('승급추진', '#2E7D32'))}"
        "<span>: 품질 등급 모델</span></div>"
    )

    st.markdown(
        _KPI_TOOLTIP_CSS
        + '<h4 style="display:flex; align-items:center; gap:6px;">🎯 운영 조치 필요 광고&nbsp;&nbsp;'
        f'<span class="kpi-info-icon">i<span class="kpi-tooltip">{_tooltip_body}</span></span>'
        '</h4>'
        '<div style="font-size:12px; color:#888; margin-top:-4px; margin-bottom:8px;">'
        '카드를 클릭하면 해당 항목의 광고 리스트를 아래에서 확인할 수 있습니다</div>',
        unsafe_allow_html=True,
    )

    # ── 4개 배지 통계 박스 (클릭 시 하단 리스트 필터링) ──
    _filter_key = f"_ml_rec_badge_filter_{page_key}"
    active_filter = st.session_state.get(_filter_key)

    stat_boxes = [
        ("즉시조치", "즉시조치", d.get("risk_immediate_count", 0), RISK_ACTION_BADGE_COLORS.get("즉시조치", "#C62828"), "#FFEBEE"),
        ("우선검토", "우선검토", d.get("risk_review_count", 0), RISK_ACTION_BADGE_COLORS.get("우선검토", "#E65100"), "#FFF3E0"),
        ("매체 다각화 추천", "매체확장", d.get("single_media_count", 0), OPPORTUNITY_BADGE_COLORS.get("매체확장", "#1565C0"), "#E3F2FD"),
        ("S등급 진입유력", "승급추진", d.get("trend_up_count", 0), "#1B5E20", "#E8F5E9"),
    ]
    stat_cols = st.columns(4)
    for i, (col, (label, badge_val, cnt, color, bg)) in enumerate(zip(stat_cols, stat_boxes)):
        with col:
            box_key = f"ml_statbox_{page_key}_{i}"
            is_selected = active_filter == badge_val
            ring = f"box-shadow:0 0 0 2px {color};" if is_selected else ""
            st.markdown(f"""
            <style>
            .st-key-{box_key} {{ position:relative; }}
            .st-key-{box_key}_btn {{
                position:absolute !important; inset:0; z-index:2;
            }}
            .st-key-{box_key}_btn div[data-testid="stButton"],
            .st-key-{box_key}_btn button {{
                width:100%; height:100%;
            }}
            .st-key-{box_key}_btn button {{
                opacity:0; cursor:pointer;
            }}
            </style>
            """, unsafe_allow_html=True)
            with st.container(key=box_key):
                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {color}; border-radius:10px;
                            padding:14px 12px; text-align:center; position:relative; {ring}">
                    <div style="font-size:12px; color:{color}; font-weight:600;">{label}</div>
                    <div style="font-size:20px; font-weight:800; color:{color}; margin-top:4px;">{cnt:,}건</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(" ", key=f"{box_key}_btn", use_container_width=True):
                    st.session_state[_filter_key] = None if is_selected else badge_val
                    st.rerun()

    # ── 통합 리스트: 기회 광고 + 위험 광고 (즉시조치 → 우선검토 → 매체확장 → 승급추진 순) ──
    _badge_priority = {"즉시조치": 0, "우선검토": 1, "매체확장": 2, "승급추진": 3}
    combined = list(d.get("opportunity_ads", [])) + list(d.get("risk_ads", []))
    combined.sort(key=lambda a: _badge_priority.get(a.get("badge"), 99))
    if active_filter:
        combined = [a for a in combined if a.get("badge") == active_filter][:4]
    else:
        _seen_badges = set()
        _top1 = []
        for a in combined:
            b = a.get("badge")
            if b not in _seen_badges:
                _seen_badges.add(b)
                _top1.append(a)
        combined = _top1
    if combined:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

        for ad in combined:
            is_opportunity = "m1_grade" in ad
            _ad_anchor = f"ad-{page_key}-rec-{ad['ads_idx']}"

            _cvr_tooltip_body = (
                "<div style='white-space:nowrap;'><b>규모 지표</b></div>"
                f"<div style='white-space:nowrap;'>클릭수 {ad.get('clk', 0):,}건, "
                f"완료수 {ad.get('turn', 0):,}건</div>"
            )
            _cvr_icon = (
                f'<span class="kpi-info-icon" style="width:14px; height:14px; font-size:9px;">i'
                f'<span class="kpi-tooltip">{_cvr_tooltip_body}</span></span>'
            )

            _value_text_style = "font-size:13px; font-weight:700;"

            if is_opportunity:
                badge_color = OPPORTUNITY_BADGE_COLORS.get(ad["badge"], "#666")
                grade_color = ML_GRADE_COLORS.get(ad["m1_grade"], "#757575")
                text_color = _desaturate(grade_color, 0.45)
                pill_html = f'<span style="{_value_text_style} color:{text_color};">{ad["m1_grade"]}등급</span>'
                desc_parts = []
                if ad.get("cvr") is not None:
                    desc_parts.append(f'CVR {ad["cvr"]:.1f}%&nbsp;&nbsp;{_cvr_icon}')
            else:
                badge_color = RISK_ACTION_BADGE_COLORS.get(ad["badge"], "#E65100")
                proba = ad["m2_proba"]
                text_color = _desaturate(badge_color, 0.45)
                pill_html = f'<span style="{_value_text_style} color:{text_color};">부진확률 {proba * 100:.0f}%</span>'
                desc_parts = []
                if ad.get("cvr") is not None:
                    desc_parts.append(f'CVR {ad["cvr"]:.1f}%&nbsp;&nbsp;{_cvr_icon}')
            desc_line = " · ".join(desc_parts)

            card_col, btn_col = st.columns([11, 1])
            with card_col:
                st.markdown(f"""
                <div id="{_ad_anchor}" style="background:#fff; border:1px solid #E8E4DC; border-radius:10px;
                            padding:12px 16px; margin-bottom:8px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1; display:flex; align-items:center; gap:8px;">
                        <span style="background:{badge_color}; color:#fff; padding:2px 10px;
                                     border-radius:4px; font-size:11px; font-weight:600;">{ad["badge"]}</span>
                        <span style="font-weight:700; font-size:14px; max-width:300px; overflow:hidden;
                                     text-overflow:ellipsis; white-space:nowrap;">{ad["ads_name"] or f'#A-{ad["ads_idx"]}'}</span>
                        <span style="font-size:12px; color:#555;">{desc_line}</span>
                    </div>
                    <div style="display:flex; align-items:center;">{pill_html}</div>
                </div>
                """, unsafe_allow_html=True)
            with btn_col:
                _btn_key = f"{page_key}_rec_{ad['ads_idx']}"
                st.markdown(
                    f'<style>.st-key-{_btn_key} {{ margin-top:-12px; }}</style>',
                    unsafe_allow_html=True,
                )
                if st.button("상세", key=_btn_key, type="secondary"):
                    _prev = st.session_state.get("detail_ads_idx")
                    if _prev and _prev != ad["ads_idx"]:
                        st.session_state.pop(f"_detail_precomputed_{_prev}", None)
                        st.session_state.pop(f"_media_cvr_{_prev}", None)
                    st.session_state["detail_ads_idx"] = ad["ads_idx"]
                    st.session_state["detail_source"] = _PAGE_KEY_TO_FILE.get(page_key, "pages/01_전체_광고_현황_오버뷰.py")
                    st.session_state["detail_badge"] = ad["badge"]
                    st.session_state["detail_model"] = "m1" if is_opportunity else "m2"
                    st.session_state["scroll_to_ad"] = _ad_anchor
                    st.switch_page("pages/05_광고_상세.py")


# ════════════════════════════════════════════════════
# ML 인사이트 섹션 (기회발굴 + 손실방어)
# ════════════════════════════════════════════════════

def render_ml_insight(insight_data: dict, page_key: str):
    """
    ML 인사이트 컨테이너 전체 렌더링.
    - 품질 등급 탭 (기회발굴)
    - 조기부진 탭 (손실방어)
    @st.fragment로 감싸서 탭 전환 시 이 섹션만 부분 리렌더링.
    """
    from src.config import (
        ML_INSIGHT_BG, ML_INSIGHT_HEADER_BG,
        ML_GRADE_COLORS, ML_GRADE_LABELS,
        RISK_COLORS, RISK_LABELS,
        OPPORTUNITY_BADGE_COLORS, RISK_ACTION_BADGE_COLORS,
    )

    # 탭 세션 초기화 (fragment 밖에서 1회)
    tab_key = f"ml_tab_{page_key}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = "grade"

    # ── CSS는 fragment 밖에서 1회만 삽입 ──
    _ml_key = f"ml_insight_{page_key}"
    st.markdown(f"""
    <style>
    div[data-testid="element-container"]:has(> div[data-testid="stVerticalBlockBorderWrapper"] > div[data-key="{_ml_key}"]) > div > div > div[data-testid="stVerticalBlock"] {{
        background: {ML_INSIGHT_BG};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0 0 0;
    }}
    </style>
    """, unsafe_allow_html=True)

    @st.fragment
    def _ml_insight_fragment():
        d = insight_data

        with st.container(key=_ml_key):

            # ── 헤더: 현재 화면 정보 ──
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding-top:2px;">
                <span style="font-size:13px; color:#555;">
                    현재 화면 : <b>{d['page_name']}</b> → 운영 중 광고 <b>{d['active_count']:,}</b>건
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── 탭 전환 컨트롤 (세그먼트 컨트롤) ──
            st.segmented_control(
                "탭 선택", ["grade", "risk"],
                format_func=lambda v: "품질등급" if v == "grade" else "조기부진",
                key=tab_key, label_visibility="collapsed",
            )
            if st.session_state[tab_key] is None:
                st.session_state[tab_key] = "grade"

            # ── 탭 내용 ──
            if st.session_state[tab_key] == "grade":
                _render_grade_tab(d, page_key)
            else:
                _render_risk_tab(d, page_key)

    _ml_insight_fragment()

    # ── 상세에서 돌아온 경우 스크롤 복원 (fragment 밖 — full rerun 시에만 실행) ──
    _scroll_target = st.session_state.pop("scroll_to_ad", None)
    if _scroll_target:
        st.html(f"""
        <script>
        (function() {{
            function tryScroll(attempts) {{
                var el = document.getElementById("{_scroll_target}");
                if (el) {{
                    el.scrollIntoView({{behavior: "smooth", block: "center"}});
                }} else if (attempts < 25) {{
                    setTimeout(function() {{ tryScroll(attempts + 1); }}, 200);
                }}
            }}
            tryScroll(0);
        }})();
        </script>
        """, unsafe_allow_javascript=True)


def _render_grade_tab(d: dict, page_key: str):
    """품질 등급 탭 (기회발굴도구) 내용 렌더링."""
    from src.config import ML_GRADE_COLORS, OPPORTUNITY_BADGE_COLORS

    opp_count = d["opportunity_count"]
    media_c = d["single_media_count"]
    trend_c = d["trend_up_count"]

    # ── Summary 배너 ──
    st.markdown(f"""
    <div style="background:#E8F5E9; border-radius:10px; padding:14px 18px; margin:10px 0;
                display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:15px; font-weight:700; color:#1B5E20;">
                {opp_count}건의 광고에서 추가 성장 기회가 포착됐어요
            </div>
            <div style="font-size:12px; color:#555; margin-top:4px;">
                매체 다각화 추천 {media_c}건 · S등급 진입 유력 {trend_c}건
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 기회 광고 리스트 ──
    opp_ads = d["opportunity_ads"]
    if opp_ads:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin:12px 0 -10px 0; padding-right:12px;">
            <span style="font-size:14px; font-weight:700;">기회 광고</span>
            <span style="font-size:11px; color:#888;">상위 {len(opp_ads)}건 표시</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(_KPI_TOOLTIP_CSS, unsafe_allow_html=True)

        for ad in opp_ads:
            badge_color = OPPORTUNITY_BADGE_COLORS.get(ad["badge"], "#666")
            grade_text_color = _desaturate(ML_GRADE_COLORS.get(ad["m1_grade"], "#757575"), 0.45)
            cvr_str = f'CVR {ad["cvr"]:.1f}%' if ad.get("cvr") is not None else "-"
            clk = ad.get("clk", 0) or 0
            turn = ad.get("turn", 0) or 0

            _cvr_tooltip_body = (
                "<div style='white-space:nowrap;'><b>규모 지표</b></div>"
                f"<div style='white-space:nowrap;'>클릭수 {clk:,}건, 완료수 {turn:,}건</div>"
            )
            _cvr_icon = (
                f'<span class="kpi-info-icon" style="width:14px; height:14px; font-size:9px;">i'
                f'<span class="kpi-tooltip">{_cvr_tooltip_body}</span></span>'
            )

            _ad_anchor = f"ad-{page_key}-gd-{ad['ads_idx']}"
            card_col, btn_col = st.columns([11, 1])
            with card_col:
                st.markdown(f"""
                <div id="{_ad_anchor}" style="background:#fff; border:1px solid #E8E4DC; border-radius:10px;
                            padding:12px 16px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:10px; flex:1; min-width:0;">
                        <span style="background:{badge_color}; color:#fff; padding:2px 10px;
                                     border-radius:4px; font-size:11px; font-weight:600; white-space:nowrap;">{ad["badge"]}</span>
                        <span style="font-weight:700; font-size:14px; overflow:hidden;
                                     text-overflow:ellipsis; white-space:nowrap;">{ad["ads_name"] or f'#A-{ad["ads_idx"]}'}</span>
                        <span style="font-size:12px; color:#555; white-space:nowrap; display:inline-flex; align-items:center; gap:6px;">{cvr_str}{_cvr_icon}</span>
                    </div>
                    <div style="font-weight:700; font-size:13px; color:{grade_text_color}; white-space:nowrap; margin-left:12px;">{ad["m1_grade"]}등급</div>
                </div>
                """, unsafe_allow_html=True)
            with btn_col:
                _btn_key = f"{page_key}_gd_{ad['ads_idx']}"
                st.markdown(
                    f'<style>.st-key-{_btn_key} {{ margin-top:-12px; }}</style>',
                    unsafe_allow_html=True,
                )
                if st.button("상세", key=_btn_key,
                             type="secondary"):
                    # 이전 상세 캐시 무효화
                    _prev = st.session_state.get("detail_ads_idx")
                    if _prev and _prev != ad["ads_idx"]:
                        st.session_state.pop(f"_detail_precomputed_{_prev}", None)
                        st.session_state.pop(f"_media_cvr_{_prev}", None)
                    st.session_state["detail_ads_idx"] = ad["ads_idx"]
                    st.session_state["detail_source"] = _PAGE_KEY_TO_FILE.get(page_key, "pages/01_전체_광고_현황_오버뷰.py")
                    st.session_state["detail_badge"] = ad["badge"]
                    st.session_state["detail_model"] = "m1"
                    st.session_state["scroll_to_ad"] = _ad_anchor
                    st.switch_page("pages/05_광고_상세.py")


def _render_risk_tab(d: dict, page_key: str):
    """조기부진 탭 (손실방어) 내용 렌더링."""
    from src.config import RISK_ACTION_BADGE_COLORS

    decline_n = d["risk_counts"]["decline_risk"]
    risk_pct = d["risk_pct"]

    # ── Summary 배너 ──
    banner_bg = "#FFF3E0" if decline_n > 0 else "#E8F5E9"
    banner_color = "#E65100" if decline_n > 0 else "#1B5E20"
    st.markdown(f"""
    <div style="background:{banner_bg}; border-radius:10px; padding:14px 18px; margin:10px 0;
                display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:15px; font-weight:700; color:{banner_color};">
                {decline_n}건이 D+3 시점 부진 위험 — 전체의 {risk_pct}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 즉시 검토 필요 광고 리스트 ──
    risk_ads = d["risk_ads"]
    if risk_ads:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin:12px 0 6px 0; padding-right:12px;">
            <span style="font-size:14px; font-weight:700;">즉시 검토 필요 광고</span>
            <span style="font-size:11px; color:#888;">상위 {len(risk_ads)}건 표시</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(_KPI_TOOLTIP_CSS, unsafe_allow_html=True)

        for ad in risk_ads:
            badge_color = RISK_ACTION_BADGE_COLORS.get(ad["badge"], "#E65100")
            proba = ad["m2_proba"]
            text_color = _desaturate(badge_color, 0.45)
            pill_html = f'<span style="font-size:13px; font-weight:700; color:{text_color};">부진확률 {proba * 100:.0f}%</span>'

            cvr_str = f'CVR {ad["cvr"]:.1f}%' if ad.get("cvr") is not None else "-"
            clk = ad.get("clk", 0) or 0
            turn = ad.get("turn", 0) or 0
            _cvr_tooltip_body = (
                "<div style='white-space:nowrap;'><b>규모 지표</b></div>"
                f"<div style='white-space:nowrap;'>클릭수 {clk:,}건, 완료수 {turn:,}건</div>"
            )
            _cvr_icon = (
                f'<span class="kpi-info-icon" style="width:14px; height:14px; font-size:9px;">i'
                f'<span class="kpi-tooltip">{_cvr_tooltip_body}</span></span>'
            )

            _ad_anchor = f"ad-{page_key}-rk-{ad['ads_idx']}"
            card_col, btn_col = st.columns([11, 1])
            with card_col:
                st.markdown(f"""
                <div id="{_ad_anchor}" style="background:#fff; border:1px solid #E8E4DC; border-radius:10px;
                            padding:12px 16px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:10px; flex:1; min-width:0;">
                        <span style="background:{badge_color}; color:#fff; padding:2px 10px;
                                     border-radius:4px; font-size:11px; font-weight:600; white-space:nowrap;">{ad["badge"]}</span>
                        <span style="font-weight:700; font-size:14px; overflow:hidden;
                                     text-overflow:ellipsis; white-space:nowrap;">{ad["ads_name"] or f'#A-{ad["ads_idx"]}'}</span>
                        <span style="font-size:12px; color:#555; white-space:nowrap; display:inline-flex; align-items:center; gap:6px;">{cvr_str}{_cvr_icon}</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-left:12px;">{pill_html}</div>
                </div>
                """, unsafe_allow_html=True)
            with btn_col:
                _btn_key = f"{page_key}_rk_{ad['ads_idx']}"
                st.markdown(
                    f'<style>.st-key-{_btn_key} {{ margin-top:-12px; }}</style>',
                    unsafe_allow_html=True,
                )
                if st.button("상세", key=_btn_key,
                             type="secondary"):
                    # 이전 상세 캐시 무효화
                    _prev = st.session_state.get("detail_ads_idx")
                    if _prev and _prev != ad["ads_idx"]:
                        st.session_state.pop(f"_detail_precomputed_{_prev}", None)
                        st.session_state.pop(f"_media_cvr_{_prev}", None)
                    st.session_state["detail_ads_idx"] = ad["ads_idx"]
                    st.session_state["detail_source"] = _PAGE_KEY_TO_FILE.get(page_key, "pages/01_전체_광고_현황_오버뷰.py")
                    st.session_state["detail_badge"] = ad["badge"]
                    st.session_state["detail_model"] = "m2"
                    st.session_state["scroll_to_ad"] = _ad_anchor
                    st.switch_page("pages/05_광고_상세.py")
    elif decline_n == 0:
        st.markdown("""
        <div style="text-align:center; padding:20px; color:#2E7D32; font-size:13px;">
            현재 필터 조건에서 부진 위험 광고가 없습니다.
        </div>
        """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════
# AI Agent 채팅 섹션
# ════════════════════════════════════════════════════════════════════════

_CHAT_CSS = """
<style>
.chat-container {
    background: #2B2D42;
    border-radius: 12px;
    padding: 24px 20px 16px;
    margin-top: 8px;
}
.chat-header {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 2px;
}
.chat-subtitle {
    color: #B0B3C1;
    font-size: 13px;
    margin-bottom: 16px;
}
.chat-bubble-user {
    background: #4A7C59;
    color: #fff;
    padding: 10px 14px;
    border-radius: 12px 12px 2px 12px;
    margin: 6px 0;
    max-width: 75%;
    margin-left: auto;
    text-align: right;
    font-size: 14px;
    word-break: break-word;
}
.chat-bubble-ai {
    background: #FFFFFF;
    color: #333333;
    padding: 12px 14px;
    border-radius: 12px 12px 12px 2px;
    margin: 6px 0;
    max-width: 85%;
    font-size: 14px;
    line-height: 1.6;
    word-break: break-word;
}
.chat-bubble-ai .ev-badge {
    display: inline-block;
    background: #E8E4DC;
    border-radius: 6px;
    padding: 3px 8px;
    margin: 2px 4px 2px 0;
    font-size: 12px;
    color: #444444;
}
.chat-bubble-ai .action-item {
    color: #2E7D32;
}
.chat-bubble-ai .section-label {
    font-weight: 600;
    color: #555555;
    font-size: 12px;
    margin-top: 8px;
    margin-bottom: 2px;
}
.chat-bubble-ai .source-caption {
    display: block;
    margin-top: 8px;
    font-size: 11px;
    color: #999999;
}
.chat-messages {
    max-height: 420px;
    overflow-y: auto;
    padding-right: 4px;
    margin-bottom: 12px;
}
/* AI Agent 입력칸 테두리 */
[data-testid="stTextInput"] input {
    border: 1px solid #E8E4DC !important;
    border-radius: 8px;
}
[data-testid="stTextInput"] input:focus {
    border-color: #E8E4DC !important;
    box-shadow: 0 0 0 1px #E8E4DC !important;
}
</style>
"""


def _render_ai_bubble(response: dict) -> str:
    """AI 응답 dict를 HTML 문자열로 변환한다."""
    _esc = _html.escape
    parts: list[str] = []

    # 핵심 요약
    summary = response.get("summary", "")
    if summary:
        parts.append(f"<b>{_esc(str(summary))}</b>")

    # 근거 지표
    evidence = response.get("evidence", [])
    if evidence and isinstance(evidence, list):
        parts.append('<div class="section-label">근거 지표</div>')
        badges = ""
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            metric = _esc(str(ev.get("metric", "")))
            value = _esc(str(ev.get("value", "")))
            scope = _esc(str(ev.get("scope", "")))
            badge_text = f"{metric}: {value}" + (f" ({scope})" if scope else "")
            badges += f'<span class="ev-badge">{badge_text}</span>'
        parts.append(badges)

    # 운영 해석
    interp = response.get("interpretation", "")
    if interp:
        parts.append(f'<div class="section-label">운영 해석</div>{_esc(str(interp))}')

    # 추천 액션
    actions = response.get("actions", [])
    if actions and isinstance(actions, list):
        parts.append('<div class="section-label">추천 액션</div>')
        for a in actions:
            parts.append(f'<span class="action-item">▸ {_esc(str(a))}</span><br>')

    # 출처 (지식베이스/뉴스 검색 참조)
    sources = response.get("sources", [])
    if sources and isinstance(sources, list):
        labels = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            doc = _esc(str(s.get("doc", "")))
            section = _esc(str(s.get("section", "")))
            url = s.get("url", "")
            label = f"{doc} › {section}" if section else doc
            if url:
                label = f'<a href="{_esc(str(url))}" target="_blank" rel="noopener">{label}</a>'
            labels.append(label)
        if labels:
            parts.append(f'<span class="source-caption">📄 출처: {" · ".join(labels)}</span>')

    return '<div class="chat-bubble-ai">' + "<br>".join(parts) + "</div>"


def render_chat_section(
    page_key: str,
    ad_summary: pd.DataFrame,
    kpis: dict,
    page_name: str,
    filters_desc: str = "",
):
    """AI Agent 채팅 섹션을 렌더링한다."""
    history_key = f"{page_key}_chat_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # CSS 주입
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # 헤더
    st.markdown(
        '<div class="chat-container">'
        '<div class="chat-header">AI Agent — 운영 어시스턴트</div>'
        '<div class="chat-subtitle">자연어로 광고 현황을 질문하거나 운영 액션을 요청하세요</div>',
        unsafe_allow_html=True,
    )

    # 채팅 이력 렌더링
    history = st.session_state[history_key]
    if history:
        msgs_html = '<div class="chat-messages">'
        for msg in history:
            if msg["role"] == "user":
                msgs_html += f'<div class="chat-bubble-user">{_html.escape(str(msg["content"]))}</div>'
            else:
                msgs_html += _render_ai_bubble(msg["content"])
        msgs_html += "</div>"
        st.markdown(msgs_html, unsafe_allow_html=True)

    # 컨테이너 닫기
    st.markdown("</div>", unsafe_allow_html=True)

    # 입력 영역 (chat-container와 좌우 폭을 맞춤)
    _input_row_key = f"{page_key}_chat_input_row"
    st.markdown(
        f'<style>.st-key-{_input_row_key} {{ padding: 0 20px; margin-top: -8px; }}</style>',
        unsafe_allow_html=True,
    )
    with st.container(key=_input_row_key):
        in_col, btn_col = st.columns([5, 1])
        with in_col:
            user_input = st.text_input(
                "질문 입력",
                placeholder="광고 현황 질문 또는 운영 액션 요청...",
                key=f"{page_key}_chat_input",
                label_visibility="collapsed",
            )
        with btn_col:
            send_clicked = st.button("전송 ↗", key=f"{page_key}_chat_send", type="primary",
                                      use_container_width=True)

    if send_clicked and user_input and user_input.strip():
        query = user_input.strip()

        # 데이터 컨텍스트 구성 (lazy import to avoid slow google.genai at startup)
        from src.agent import build_data_context, generate_response
        from src.rag import retrieve
        ctx = build_data_context(
            ad_summary=ad_summary,
            kpis=kpis,
            page_name=page_name,
            filters_desc=filters_desc,
        )
        retrieved = retrieve(query)

        # Gemini 응답 생성 (현재 메시지는 generate_response 내부에서 추가되므로 history에서 제외)
        with st.spinner("AI가 분석 중입니다..."):
            result = generate_response(
                user_message=query,
                data_context=ctx,
                chat_history=st.session_state[history_key],
                retrieved_chunks=retrieved,
            )

        # 사용자 메시지 + AI 응답 추가
        st.session_state[history_key].append({"role": "user", "content": query})
        st.session_state[history_key].append({"role": "assistant", "content": result})
        st.rerun()
