"""Plotly 차트 생성 함수"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from src.config import (
    COLORS, TYPE_COLORS, MEDIA_COLORS, CATEGORY_COLORS,
    fmt_pct, fmt_number, fmt_currency,
    MEDIA_NAME_MAP,
)


def _get_color(group_col: str, name: str) -> str:
    """그룹 기준에 따른 색상 반환"""
    if group_col == "analysis_ads_type_label":
        return TYPE_COLORS.get(name, "#888")
    elif group_col == "final_media":
        return MEDIA_COLORS.get(name, "#888")
    elif group_col == "category_name":
        return CATEGORY_COLORS.get(name, "#888")
    return "#888"


# ═══════════════════════════════════════════════════
# 히트맵
# ═══════════════════════════════════════════════════
def make_heatmap(df: pd.DataFrame) -> go.Figure:
    """시간대 × 요일 CVR 히트맵"""
    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]
    hours = list(range(24))

    # 요일 × 시간 별 CVR (벡터화)
    agg = df.groupby(["weekday_kr", "rpt_time_time"]).agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
    ).reset_index()
    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, np.nan)

    pivot_cvr = agg.pivot_table(index="weekday_kr", columns="rpt_time_time", values="cvr")
    pivot_cvr = pivot_cvr.reindex(weekday_order).reindex(columns=range(24))

    pivot_clk = agg.pivot_table(index="weekday_kr", columns="rpt_time_time", values="clk", fill_value=0)
    pivot_clk = pivot_clk.reindex(weekday_order, fill_value=0).reindex(columns=range(24), fill_value=0)

    # 클릭수 30 미만인 셀은 NaN 처리하여 회색으로 표시
    cvr_masked = pivot_cvr.where(pivot_clk >= 30)
    matrix = cvr_masked.values.tolist()

    # 호버 텍스트: 클릭수 부족 셀은 별도 표기
    clk_arr = pivot_clk.values
    cvr_arr = pivot_cvr.values
    hover_text = []
    for i, day in enumerate(weekday_order):
        row = []
        for j in range(24):
            cvr_str = f"{cvr_arr[i][j]:.1f}%" if not np.isnan(cvr_arr[i][j]) else "-"
            if clk_arr[i][j] < 30:
                row.append(f"{day} {j}시<br>클릭수: {int(clk_arr[i][j])} (30 미만)<br>CVR: {cvr_str}")
            else:
                row.append(f"{day} {j}시<br>CVR: {cvr_str}")
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"{h}시" for h in hours],
        y=weekday_order,
        colorscale=[
            [0, "#E8EFF5"],
            [0.5, "#8DA9C3"],
            [1, "#284C76"],
        ],
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        xgap=2,
        ygap=2,
        colorbar=dict(title=dict(text="CVR(%)", side="right")),
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis=dict(dtick=3),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════
# 시간대별 CVR 추이 (당일 vs 평소 평균)
# ═══════════════════════════════════════════════════
def make_hourly_cvr_trend(df: pd.DataFrame, today: pd.Timestamp) -> tuple:
    """시간대별 CVR 추이 — 당일 vs 평소 평균. Returns (fig, merged_df)."""
    today_df = df[df["rpt_time_date"] == today]
    past_df = df[df["rpt_time_date"] < today]

    # 벡터화 집계
    today_agg = today_df.groupby("rpt_time_time").agg(
        clk=("rpt_time_clk", "sum"), turn=("rpt_time_turn", "sum"),
    ).reset_index()
    today_agg["cvr_today"] = np.where(today_agg["clk"] > 0, today_agg["turn"] / today_agg["clk"] * 100, 0)
    today_hourly = today_agg[["rpt_time_time", "cvr_today"]]

    past_agg = past_df.groupby("rpt_time_time").agg(
        clk=("rpt_time_clk", "sum"), turn=("rpt_time_turn", "sum"),
    ).reset_index()
    past_agg["cvr_avg"] = np.where(past_agg["clk"] > 0, past_agg["turn"] / past_agg["clk"] * 100, 0)
    past_hourly = past_agg[["rpt_time_time", "cvr_avg"]]

    merged = pd.DataFrame({"rpt_time_time": range(24)})
    merged = merged.merge(today_hourly, on="rpt_time_time", how="left")
    merged = merged.merge(past_hourly, on="rpt_time_time", how="left")
    merged = merged.fillna(0)

    # 격차 최대 시점 — 음수 격차(오늘<평소) 중 가장 큰 시점, 모두 양수면 격차가 가장 작은 시점
    merged["gap"] = merged["cvr_today"] - merged["cvr_avg"]
    negative_gaps = merged[merged["gap"] < 0]
    if len(negative_gaps) > 0:
        max_gap_idx = negative_gaps["gap"].idxmin()
    else:
        max_gap_idx = merged["gap"].idxmin()
    max_gap_hour = merged.loc[max_gap_idx, "rpt_time_time"]
    max_gap_val = merged.loc[max_gap_idx, "gap"]

    merged["hour_label"] = merged["rpt_time_time"].astype(int).astype(str) + "시"

    fig = go.Figure()

    # 평시 평균
    fig.add_trace(go.Scatter(
        x=merged["hour_label"], y=merged["cvr_avg"],
        mode="lines", name="평시 평균 CVR",
        line=dict(color=COLORS["neutral"], dash="dot", width=2),
        hovertemplate="평시 평균 CVR: %{y:.1f}%<extra></extra>",
    ))

    # 당일
    fig.add_trace(go.Scatter(
        x=merged["hour_label"], y=merged["cvr_today"],
        mode="lines+markers", name="당일 CVR",
        line=dict(color=COLORS["weekend"], width=2.5),
        marker=dict(size=4),
        hovertemplate="당일 CVR: %{y:.1f}%<extra></extra>",
    ))

    # 격차 최고점 마커 (초록) — 당일이 평시 대비 가장 높은 시점
    best_gap_idx = merged["gap"].idxmax()
    best_gap_hour = merged.loc[best_gap_idx, "rpt_time_time"]
    best_gap_val = merged.loc[best_gap_idx, "gap"]
    fig.add_trace(go.Scatter(
        x=[merged.loc[best_gap_idx, "hour_label"]],
        y=[merged.loc[best_gap_idx, "cvr_today"]],
        mode="markers+text",
        name="",
        marker=dict(size=12, color=COLORS["positive"], symbol="circle"),
        text=[f"★ {int(best_gap_hour)}시 ({best_gap_val:+.1f}%p)"],
        textposition="top center",
        textfont=dict(size=11, color=COLORS["positive"]),
        customdata=[f"{best_gap_val:+.1f}"],
        hovertemplate="당일 vs 평시 격차: %{customdata}%p<extra></extra>",
        showlegend=False,
    ))

    # 격차 최저점 마커 (빨강) — 당일이 평시 대비 가장 낮은 시점
    fig.add_trace(go.Scatter(
        x=[merged.loc[max_gap_idx, "hour_label"]],
        y=[merged.loc[max_gap_idx, "cvr_today"]],
        mode="markers+text",
        name="",
        marker=dict(size=12, color=COLORS["negative"], symbol="circle"),
        text=[f"⚠ {int(max_gap_hour)}시 ({max_gap_val:+.1f}%p)"],
        textposition="bottom center",
        textfont=dict(size=11, color=COLORS["negative"]),
        customdata=[f"{max_gap_val:+.1f}"],
        hovertemplate="당일 vs 평시 격차: %{customdata}%p<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="시간대", type="category",
                   categoryorder="array", categoryarray=merged["hour_label"].tolist(),
                   tickvals=[f"{h}시" for h in range(0, 24, 3)]),
        yaxis=dict(title="CVR (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    # 클릭(선택)한 포인트는 원래 모습 유지, 클릭하지 않은 포인트만 투명 처리
    fig.update_traces(
        selector=dict(name="당일 CVR"),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.15)),
    )
    fig.update_traces(
        selector=dict(marker_color=COLORS["positive"]),
        selected=dict(marker=dict(opacity=1), textfont=dict(color=COLORS["positive"])),
        unselected=dict(marker=dict(opacity=0.2), textfont=dict(color="#cccccc")),
    )
    fig.update_traces(
        selector=dict(marker_color=COLORS["negative"]),
        selected=dict(marker=dict(opacity=1), textfont=dict(color=COLORS["negative"])),
        unselected=dict(marker=dict(opacity=0.2), textfont=dict(color="#cccccc")),
    )
    fig.update_layout(clickmode="event+select")
    return fig, merged


# ═══════════════════════════════════════════════════
# 그룹별 CVR · 마진율 막대 차트
# ═══════════════════════════════════════════════════
def _assign_alarm(row, avg_cvr, avg_margin, min_clk=30, min_turn=1):
    """CVR·마진율 기준 알람 이모지 + 설명 부여"""
    clk = row.get("clk", 0)
    turn = row.get("turn", 0)
    cvr = row.get("cvr", None)
    mr = row.get("margin_rate", None)

    if clk < min_clk or turn < min_turn or cvr is None or mr is None or pd.isna(cvr) or pd.isna(mr):
        return "⊘", "판단 보류", "규모 부족 (클릭수/완료수 미달)"

    cvr_up = cvr >= avg_cvr
    mr_up = mr >= avg_margin

    if cvr_up and mr_up:
        return "★", "최우수 (양지표)", f"CVR↑ 마진율↑\nCVR {cvr:.1f}% / 마진율 {mr:.1f}%"
    elif (not cvr_up) and mr_up:
        return "↺", "소재 개선 후보", f"CVR↓ 마진율↑\nCVR {cvr:.1f}% / 마진율 {mr:.1f}%"
    elif cvr_up and (not mr_up):
        return "⚠", "수익율 점검 필요", f"CVR↑ 마진율↓\nCVR {cvr:.1f}% / 마진율 {mr:.1f}%"
    else:
        return "", "", ""


def make_cvr_margin_bar(group_kpis: pd.DataFrame, group_col: str,
                        scale_col: str = "clk") -> go.Figure:
    """그룹별 CVR과 마진율 비교 막대 차트 (scale_col로 정렬 기준 완료)"""
    df = group_kpis.sort_values("cvr", ascending=False).copy()
    avg_cvr = df["cvr"].mean()
    avg_margin = df["margin_rate"].mean()

    scale_label = "클릭수" if scale_col == "clk" else "완료수"

    # 알람 라벨 계산
    alarms = []
    for _, row in df.iterrows():
        emoji, label, desc = _assign_alarm(row, avg_cvr, avg_margin)
        alarms.append({"emoji": emoji, "label": label, "desc": desc})

    # x축 라벨: 알람 이모지 + 유형명 + 선택된 규모지표 값
    x_labels = []
    for i, (_, row) in enumerate(df.iterrows()):
        name = row[group_col]
        val = fmt_number(row.get(scale_col, 0))
        alarm = alarms[i]
        prefix = f"{alarm['emoji']} " if alarm['emoji'] else ""
        x_labels.append(
            f"{prefix}{name}<br><span style='font-size:10px;color:#888'>"
            f"{scale_label} {val}</span>"
        )

    fig = go.Figure()

    # customdata: [scale_value, alarm_emoji, alarm_label, alarm_desc]
    alarm_labels = [a["label"] for a in alarms]
    alarm_descs = [a["desc"].replace("\n", "<br>") for a in alarms]
    alarm_emojis = [a["emoji"] for a in alarms]
    custom = list(zip(
        df[scale_col].values,
        alarm_emojis,
        alarm_labels,
        alarm_descs,
    ))

    # CVR 막대
    fig.add_trace(go.Bar(
        name="CVR",
        x=x_labels, y=df["cvr"],
        marker_color="#284C76",
        text=df["cvr"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        customdata=custom,
        hovertemplate=(
            "<b>%{x}</b><br>CVR: %{y:.1f}%<br>" + scale_label + ": %{customdata[0]:,}"
            "<br><br>%{customdata[1]} %{customdata[2]}<br>%{customdata[3]}"
            "<extra></extra>"
        ),
    ))

    # 마진율 막대
    fig.add_trace(go.Bar(
        name="마진율",
        x=x_labels, y=df["margin_rate"],
        marker_color="#8DA9C3",
        text=df["margin_rate"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-"),
        textposition="outside",
        customdata=custom,
        hovertemplate=(
            "<b>%{x}</b><br>마진율: %{y:.1f}%<br>" + scale_label + ": %{customdata[0]:,}"
            "<br><br>%{customdata[1]} %{customdata[2]}<br>%{customdata[3]}"
            "<extra></extra>"
        ),
    ))

    # ── 막대별 끊어진 평균 점선 (paper 좌표 기반) ──
    n = len(df)
    bar_w = 0.35 / n  # 막대 반폭 (paper 좌표)
    CVR_AVG_COLOR = "#FF8C00"      # 선명한 오렌지
    MARGIN_AVG_COLOR = "#1D3557"   # 진한 남색

    for i in range(n):
        center = (i + 0.5) / n
        # CVR 평균 세그먼트 (왼쪽 막대 위)
        fig.add_shape(
            type="line", xref="paper", yref="y",
            x0=center - bar_w, x1=center - 0.005,
            y0=avg_cvr, y1=avg_cvr,
            line=dict(dash="dot", color=CVR_AVG_COLOR, width=2),
        )
        # 마진율 평균 세그먼트 (오른쪽 막대 위)
        fig.add_shape(
            type="line", xref="paper", yref="y",
            x0=center + 0.005, x1=center + bar_w,
            y0=avg_margin, y1=avg_margin,
            line=dict(dash="dot", color=MARGIN_AVG_COLOR, width=2),
        )

    # 평균값 라벨 (차트 우측 상단에 범례처럼 표기)
    avg_text = (
        f"<span style='color:{CVR_AVG_COLOR}'>— 평균 CVR: {avg_cvr:.1f}%</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='color:{MARGIN_AVG_COLOR}'>— 평균 마진율: {avg_margin:.1f}%</span>"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=1, y=1.12,
        text=avg_text,
        showarrow=False, font=dict(size=11),
        xanchor="right", yanchor="bottom",
    )

    fig.update_layout(
        barmode="group", height=400,
        margin=dict(l=40, r=20, t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════
# 사분면 버블 차트
# ═══════════════════════════════════════════════════
def _detect_bubble_overlaps(df, group_col, x_metric, y_metric, size_metric):
    """버블 겹침 감지. 겹치는 버블 그룹을 {name: [group_names]} 형태로 반환."""
    if len(df) <= 1:
        return {}

    names = df[group_col].values
    x_vals = df[x_metric].values.astype(float)
    y_vals = df[y_metric].values.astype(float)

    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    if x_range == 0:
        x_range = abs(x_vals[0]) if x_vals[0] != 0 else 1
    if y_range == 0:
        y_range = abs(y_vals[0]) if y_vals[0] != 0 else 1

    x_norm = (x_vals - x_vals.min()) / x_range
    y_norm = (y_vals - y_vals.min()) / y_range

    max_size_val = df[size_metric].max() if df[size_metric].max() > 0 else 1
    sizes = (df[size_metric].values / max_size_val * 40) + 8

    n = len(df)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            dist = ((x_norm[i] - x_norm[j]) ** 2 + (y_norm[i] - y_norm[j]) ** 2) ** 0.5
            threshold = (sizes[i] + sizes[j]) / 2 / 350
            if dist < threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    result = {}
    for indices in groups.values():
        if len(indices) > 1:
            group_names = [names[i] for i in indices]
            for name in group_names:
                result[name] = group_names

    return result


def _build_bubble_hover(df, group_col, names_list, show_acost: bool = True):
    """겹치는 버블 그룹의 결합 hover 텍스트 생성."""
    parts = []
    for gname in names_list:
        grow = df[df[group_col] == gname].iloc[0]
        acost_str = fmt_currency(grow.get('acost')) if pd.notna(grow.get('acost')) else '-'
        turn_str = fmt_number(grow.get('turn', 0))
        cvr_str = f"{grow.get('cvr', 0):.1f}" if pd.notna(grow.get('cvr')) else '-'
        cpa_str = fmt_currency(grow.get('cpa')) if pd.notna(grow.get('cpa')) else '-'
        margin_str = f"{grow.get('margin_rate', 0):.1f}" if pd.notna(grow.get('margin_rate')) else '-'
        acost_line = f"광고비 : {acost_str}<br>" if show_acost else ""
        parts.append(
            f"<b>{gname}</b><br>"
            f"{acost_line}"
            f"완료수 : {turn_str}건<br>"
            f"CVR : {cvr_str}%<br>"
            f"CPA : {cpa_str}<br>"
            f"마진율 : {margin_str}%"
        )
    return "<br>──────────<br>".join(parts) + "<extra></extra>"


def make_bubble_chart(
    group_kpis: pd.DataFrame,
    group_col: str,
    x_metric: str = "cvr",
    y_metric: str = "margin_rate",
    size_metric: str = "clk",
    show_breakeven: bool = False,
    show_acost: bool = True,
):
    """종합 평가 버블 차트 (사분면 + 기준선). Returns (fig, overlap_groups)."""
    df = group_kpis.dropna(subset=[x_metric, y_metric]).copy()
    if df.empty:
        return go.Figure(), {}

    avg_x = df[x_metric].mean()
    avg_y = df[y_metric].mean()

    # X/Y축 라벨
    x_labels = {"cvr": "CVR (%)", "cpa": "CPA (원) — 낮을수록 좋음"}
    y_labels = {"margin_rate": "마진율 (%)"}

    # 버블 크기 정규화
    max_size = df[size_metric].max() if df[size_metric].max() > 0 else 1
    df["bubble_size"] = (df[size_metric] / max_size * 40) + 8

    # 겹침 감지
    overlap_groups = _detect_bubble_overlaps(df, group_col, x_metric, y_metric, size_metric)

    fig = go.Figure()

    for idx, row in df.iterrows():
        name = row[group_col]
        group_names = overlap_groups.get(name)

        if group_names and len(group_names) > 1:
            hover_text = _build_bubble_hover(df, group_col, group_names, show_acost=show_acost)
        else:
            acost_line = "광고비 : %{customdata[1]}<br>" if show_acost else ""
            hover_text = (
                "<b>%{customdata[0]}</b><br>"
                f"{acost_line}"
                "완료수 : %{customdata[2]}건<br>"
                "CVR : %{customdata[3]}%<br>"
                "CPA : %{customdata[4]}<br>"
                "마진율 : %{customdata[5]}%"
                "<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=[row[x_metric]],
            y=[row[y_metric]],
            mode="markers+text",
            marker=dict(
                size=row["bubble_size"],
                color=_get_color(group_col, name),
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            text=[name],
            textposition="top center",
            textfont=dict(size=10),
            name=name,
            customdata=[[
                name,
                fmt_currency(row.get('acost')) if pd.notna(row.get('acost')) else '-',
                fmt_number(row.get('turn', 0)),
                f"{row.get('cvr', 0):.1f}" if pd.notna(row.get('cvr')) else '-',
                fmt_currency(row.get('cpa')) if pd.notna(row.get('cpa')) else '-',
                f"{row.get('margin_rate', 0):.1f}" if pd.notna(row.get('margin_rate')) else '-',
            ]],
            hovertemplate=hover_text,
        ))

    # 세로 기준선 (X축 평균)
    x_label = {"cvr": "평균 CVR", "cpa": "평균 CPA"}.get(x_metric, "")
    fig.add_vline(x=avg_x, line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text=f"{x_label} {avg_x:.1f}", annotation_position="top")

    # 가로 기준선 (마진율 평균)
    fig.add_hline(y=avg_y, line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text=f"평균 마진율 {avg_y:.1f}%")

    # 사분면 배경색 + 라벨
    _add_quadrant_backgrounds(fig, x_metric, avg_x, avg_y, df, show_breakeven)
    _add_quadrant_labels(fig, x_metric, avg_x, avg_y, df, show_breakeven)

    # 데이터 기반 축 범위 (버블이 화면에 가깝게)
    x_min, x_max = df[x_metric].min(), df[x_metric].max()
    y_min, y_max = df[y_metric].min(), df[y_metric].max()
    x_pad = (x_max - x_min) * 0.2 if x_max != x_min else abs(x_min) * 0.3 or 1
    y_pad = (y_max - y_min) * 0.2 if y_max != y_min else abs(y_min) * 0.3 or 1

    fig.update_layout(
        height=500,
        margin=dict(l=50, r=20, t=30, b=60),
        xaxis=dict(title=x_labels.get(x_metric, x_metric),
                   range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(title=y_labels.get(y_metric, y_metric),
                   range=[y_min - y_pad, y_max + y_pad]),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig, overlap_groups


def _add_quadrant_backgrounds(fig, x_metric, avg_x, avg_y, df, show_breakeven=False):
    """사분면 배경색 추가"""
    x_ref = avg_x
    is_lower_better = (x_metric == "cpa")

    x_min = df[x_metric].min()
    x_max = df[x_metric].max()
    y_min = df["margin_rate"].min()
    y_max = df["margin_rate"].max()
    x_pad = (x_max - x_min) * 2 if x_max != x_min else 10
    y_pad = (y_max - y_min) * 2 if y_max != y_min else 10

    x_lo = x_min - x_pad
    x_hi = x_max + x_pad
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    BEST_BG = "rgba(200, 230, 201, 0.18)"
    IMPROVE_BG = "rgba(187, 222, 251, 0.18)"
    EXPENSIVE_BG = "rgba(255, 243, 224, 0.18)"
    LOSS_BG = "rgba(255, 205, 210, 0.18)"

    if is_lower_better:
        quads = [
            (x_lo, x_ref, avg_y, y_hi, BEST_BG),
            (x_ref, x_hi, avg_y, y_hi, IMPROVE_BG),
            (x_lo, x_ref, y_lo, avg_y, EXPENSIVE_BG),
            (x_ref, x_hi, y_lo, avg_y, LOSS_BG),
        ]
    else:
        quads = [
            (x_ref, x_hi, avg_y, y_hi, BEST_BG),
            (x_lo, x_ref, avg_y, y_hi, IMPROVE_BG),
            (x_ref, x_hi, y_lo, avg_y, EXPENSIVE_BG),
            (x_lo, x_ref, y_lo, avg_y, LOSS_BG),
        ]

    for x0, x1, y0, y1, color in quads:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, line=dict(width=0),
            layer="below",
        )


def _add_quadrant_labels(fig, x_metric, avg_x, avg_y, df, show_breakeven=False):
    """사분면 라벨 추가"""
    x_range = df[x_metric].max() - df[x_metric].min() if len(df) > 1 else 10
    y_range = df["margin_rate"].max() - df["margin_rate"].min() if len(df) > 1 else 10

    x_ref = avg_x
    is_lower_better = (x_metric == "cpa")

    if is_lower_better:
        labels = [
            (x_ref - x_range * 0.3, avg_y + y_range * 0.3, "★ 최우수", COLORS["quadrant_best"]),
            (x_ref + x_range * 0.3, avg_y + y_range * 0.3, "↺ 소재 개선", COLORS["quadrant_low_eff"]),
            (x_ref - x_range * 0.3, avg_y - y_range * 0.3, "⚠ 단가 점검", COLORS["quadrant_expensive"]),
            (x_ref + x_range * 0.3, avg_y - y_range * 0.3, "⊘ 비효율", COLORS["quadrant_loss"]),
        ]
    else:
        labels = [
            (x_ref + x_range * 0.3, avg_y + y_range * 0.3, "★ 최우수", COLORS["quadrant_best"]),
            (x_ref - x_range * 0.3, avg_y + y_range * 0.3, "↺ 소재 개선", COLORS["quadrant_low_eff"]),
            (x_ref + x_range * 0.3, avg_y - y_range * 0.3, "⚠ 단가 점검", COLORS["quadrant_expensive"]),
            (x_ref - x_range * 0.3, avg_y - y_range * 0.3, "⊘ 비효율", COLORS["quadrant_loss"]),
        ]

    for x, y, text, color in labels:
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False,
            font=dict(size=11, color=color), opacity=0.6,
        )


# ═══════════════════════════════════════════════════
# CVR 추이 라인 차트
# ═══════════════════════════════════════════════════
def make_cvr_trend(df: pd.DataFrame, group_col: str, period: str = "전체",
                   today=None, c_start=None, c_end=None,
                   highlight: list[str] | None = None) -> go.Figure:
    """그룹별 CVR 추이 라인 차트"""
    # X축 단위 결정: 단일 날짜이면 시간대별
    is_single_day = (
        period == "최근 1일"
        or (period == "사용자 지정" and c_start is not None and c_end is not None
            and str(c_start) == str(c_end))
    )

    if is_single_day:
        x_col = "rpt_time_time"
        x_label = "시간대"
        if period == "최근 1일" and today is not None:
            base_date = pd.Timestamp(today)
        elif c_start is not None:
            base_date = pd.Timestamp(c_start)
        else:
            base_date = None
    else:
        x_col = "rpt_time_date"
        x_label = "날짜"
        base_date = None

    agg = df.groupby([x_col, group_col]).agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
    ).reset_index()
    agg["cvr"] = np.where(agg["clk"] > 0, agg["turn"] / agg["clk"] * 100, np.nan)

    # 단일 날짜: 정수 시간을 datetime으로 변환 → 틱/호버 포맷 분리 가능
    if is_single_day and base_date is not None:
        agg["_dt"] = base_date + pd.to_timedelta(agg[x_col].astype(int), unit="h")
        plot_x_col = "_dt"
    else:
        plot_x_col = x_col

    # 호버 템플릿: 한 줄로 압축하여 unified 박스 높이 최소화
    hover_tpl = (
        "<b>%{fullData.name}</b> · CVR %{y:.1f}% · "
        "클릭 %{customdata[0]:,} · 완료 %{customdata[1]:,}"
        "<extra></extra>"
    )

    fig = go.Figure()
    for name in agg[group_col].unique():
        sub = agg[agg[group_col] == name].sort_values(x_col)
        if highlight and name not in highlight:
            color = "lightgrey"
            width = 1.5
            opacity = 0.3
        else:
            color = _get_color(group_col, name)
            width = 3 if highlight else 2
            opacity = 1.0
        fig.add_trace(go.Scatter(
            x=sub[plot_x_col], y=sub["cvr"],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=width),
            marker=dict(size=4 if not highlight or name in highlight else 3),
            opacity=opacity,
            customdata=np.column_stack([sub["clk"].values, sub["turn"].values]),
            hovertemplate=hover_tpl,
        ))

    # x축 설정
    xaxis_cfg = dict(title=x_label)
    if is_single_day:
        xaxis_cfg["tickformat"] = "%H시"
        xaxis_cfg["hoverformat"] = "%Y-%m-%d %H시"
        xaxis_cfg["dtick"] = 3600000  # 1시간 간격 (ms)
    else:
        xaxis_cfg["tickformat"] = "%Y-%m-%d"
        xaxis_cfg["hoverformat"] = "%Y-%m-%d"

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=xaxis_cfg,
        yaxis=dict(title="CVR (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(font_size=12, namelength=-1),
    )
    return fig


# ═══════════════════════════════════════════════════
# Pareto 차트
# ═══════════════════════════════════════════════════
def make_pareto_chart(group_kpis: pd.DataFrame, group_col: str) -> go.Figure:
    """Top 카테고리 Pareto 차트"""
    df = group_kpis.sort_values("turn", ascending=False).copy()
    total = df["turn"].sum()
    df["pct"] = df["turn"] / total * 100 if total > 0 else 0
    df["cum_pct"] = df["pct"].cumsum()

    # 80% 도달 인덱스 → 상위 N개 강조 색상 결정
    threshold_pos = (df["cum_pct"] >= 80).idxmax() if (df["cum_pct"] >= 80).any() else None
    if threshold_pos is not None:
        n = df.index.get_loc(threshold_pos) + 1
    else:
        n = len(df)
    bar_colors = ["#284C76" if i < n else "#9CA3AF" for i in range(len(df))]

    fig = go.Figure()

    # 막대 (완료수) + 비중 텍스트
    fig.add_trace(go.Bar(
        x=df[group_col], y=df["turn"],
        name="완료수",
        marker_color=bar_colors,
        yaxis="y",
        text=[f"{v:.1f}%" for v in df["pct"]],
        textposition="outside",
        textfont=dict(size=11),
        customdata=np.column_stack([df["pct"].values, df["cum_pct"].values]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "완료수: %{y:,}<br>"
            "비중: %{customdata[0]:.1f}%<br>"
            "누적: %{customdata[1]:.1f}%"
            "<extra></extra>"
        ),
    ))

    # 누적 비율 꺾은선 (버건디)
    fig.add_trace(go.Scatter(
        x=df[group_col], y=df["cum_pct"],
        name="누적 비율",
        mode="lines+markers",
        line=dict(color="#800020", width=2.5),
        marker=dict(size=5),
        yaxis="y2",
    ))

    # 80% 기준선 (버건디)
    fig.add_hline(y=80, line_dash="dot", line_color="#800020",
                  line_width=1, yref="y2",
                  annotation_text="80%", annotation_position="right")

    fig.update_layout(
        height=420,
        margin=dict(l=50, r=50, t=30, b=80),
        yaxis=dict(title="완료수", side="left"),
        yaxis2=dict(title="누적 비율 (%)", side="right", overlaying="y", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=45),
    )

    # 80% 도달 인사이트 텍스트 (차트 밖에서 표시)
    insight = None
    if threshold_pos is not None:
        names = ", ".join(df[group_col].iloc[:n].tolist())
        insight = f"상위 {n}개 ({names})가 전체 완료의 80% 이상을 차지합니다."

    return fig, insight


# ═══════════════════════════════════════════════════
# 단일 광고 일별 성과 추이 (클릭/완료 막대 + CVR 라인)
# ═══════════════════════════════════════════════════
def make_ad_daily_trend(daily_df: pd.DataFrame) -> go.Figure:
    """단일 광고 일별 성과 추이 — 클릭/완료 막대 + CVR 라인 (듀얼 Y축)."""
    df = daily_df.copy()
    df["date_str"] = df["rpt_time_date"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 클릭수 막대
    fig.add_trace(go.Bar(
        x=df["date_str"], y=df["clk"],
        name="클릭수",
        marker_color=COLORS["bar_dim"],
        opacity=0.7,
        hovertemplate="%{x}<br>클릭수: %{y:,}<extra></extra>",
    ), secondary_y=False)

    # 완료수 막대
    fig.add_trace(go.Bar(
        x=df["date_str"], y=df["turn"],
        name="완료수",
        marker_color=COLORS["primary"],
        hovertemplate="%{x}<br>완료수: %{y:,}<extra></extra>",
    ), secondary_y=False)

    # CVR 라인
    fig.add_trace(go.Scatter(
        x=df["date_str"], y=df["cvr"],
        name="CVR",
        mode="lines+markers",
        line=dict(color=COLORS["negative"], width=2),
        marker=dict(size=5),
        hovertemplate="%{x}<br>CVR: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=30, b=40),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="건수", secondary_y=False)
    fig.update_yaxes(title_text="CVR (%)", secondary_y=True)

    return fig


# ═══════════════════════════════════════════════════
# 평일 vs 주말 CVR 미니 막대 (KPI 카드용)
# ═══════════════════════════════════════════════════
def make_weekday_weekend_mini_bar(df: pd.DataFrame, avg_wd: float, avg_we: float) -> go.Figure:
    """KPI 카드 하단용 미니 막대 — 평일/주말 색상 구분 + 각 그룹 평균선 끊김."""
    NUM_BINS = 30
    daily = df.groupby("rpt_time_date").agg(
        clk=("rpt_time_clk", "sum"),
        turn=("rpt_time_turn", "sum"),
        is_weekend=("is_weekend", "first"),
    ).reset_index().sort_values("rpt_time_date")
    daily["cvr"] = np.where(daily["clk"] > 0, daily["turn"] / daily["clk"] * 100, np.nan)

    n = len(daily)

    if n > NUM_BINS:
        bin_size = n / NUM_BINS
        daily = daily.reset_index(drop=True)
        daily["bin"] = np.minimum((daily.index / bin_size).astype(int), NUM_BINS - 1)
        # bin별 CVR (합계 기반) + 주말 다수결
        binned = daily.groupby("bin").agg(
            clk=("clk", "sum"), turn=("turn", "sum"),
            is_weekend=("is_weekend", lambda s: s.sum() > len(s) / 2),
        ).reset_index()
        binned["cvr"] = np.where(binned["clk"] > 0, binned["turn"] / binned["clk"] * 100, 0)
        vals = binned["cvr"].values[:NUM_BINS]
        is_we = binned["is_weekend"].values[:NUM_BINS]
        pad = 0
    else:
        raw_vals = daily["cvr"].fillna(0).values
        raw_we = daily["is_weekend"].values
        pad = NUM_BINS - len(raw_vals)
        vals = np.concatenate([np.full(pad, np.nan), raw_vals])
        is_we = np.concatenate([np.full(pad, False, dtype=bool), raw_we])

    # 막대 색상: 평일 primary, 주말 negative
    colors = []
    for i in range(NUM_BINS):
        if isinstance(vals[i], float) and np.isnan(vals[i]):
            colors.append("rgba(0,0,0,0)")
        elif is_we[i]:
            colors.append(COLORS["weekend"])
        else:
            colors.append(COLORS["weekday"])

    x = list(range(NUM_BINS))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=vals,
        marker_color=colors,
        marker_line_width=0,
        showlegend=False,
        hoverinfo="skip",
    ))

    # 평일 평균선 — 주말 bin은 None으로 끊김
    wd_y = [avg_wd if not is_we[i] and not (isinstance(vals[i], float) and np.isnan(vals[i])) else None
            for i in range(NUM_BINS)]
    fig.add_trace(go.Scatter(
        x=x, y=wd_y,
        mode="lines", showlegend=False,
        line=dict(color=COLORS["avg_line"], dash="dot", width=3),
        connectgaps=False, hoverinfo="skip",
    ))

    # 주말 평균선 — 평일 bin은 None으로 끊김
    we_y = [avg_we if is_we[i] and not (isinstance(vals[i], float) and np.isnan(vals[i])) else None
            for i in range(NUM_BINS)]
    fig.add_trace(go.Scatter(
        x=x, y=we_y,
        mode="lines", showlegend=False,
        line=dict(color=COLORS["avg_line"], dash="dot", width=3),
        connectgaps=False, hoverinfo="skip",
    ))

    fig.update_layout(
        height=80, margin=dict(l=0, r=0, t=4, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.15,
    )
    return fig


def make_mini_sparkline(daily_df: pd.DataFrame, kpi_col: str, color: str, unit: str = "%") -> go.Figure:
    """매체 인사이트 카드용 미니 라인 스파크라인 — 축/범례 없이 단색 라인 + 옅은 영역 채우기.

    클릭이 없는 날(NaN)은 건너뛰고 실제 값이 있는 날만 이어 그린다 — 빈 날을 그대로
    플로팅하면 라인이 중간에 끊겨 보인다.
    """
    df = daily_df.sort_values("rpt_time_date").dropna(subset=[kpi_col])
    x = list(range(len(df)))
    y = df[kpi_col].to_numpy()

    r, g, b = (int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    fill_color = f"rgba({r},{g},{b},0.12)"

    y_max = np.nanmax(y) if np.any(~np.isnan(y)) else 1
    y_range = [0, y_max * 1.25 if y_max > 0 else 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines", showlegend=False,
        line=dict(color=color, width=2.5, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor=fill_color,
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=170, margin=dict(l=34, r=8, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=True, range=y_range, showgrid=True, gridcolor="#EFEDE7",
                   tickfont=dict(size=9), ticksuffix=unit),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_acost_rank_bar(group_kpis: pd.DataFrame, group_col: str,
                        highlight_name: str, highlight_color: str) -> go.Figure:
    """전체 매체 광고비 순위 막대(로그 스케일) — 대상 매체만 강조색, 나머지는 회색.

    일별 추이를 그릴 데이터가 부족한 매체("비효율" 카드 등)의 대체 시각화 —
    이 매체의 광고비가 전체 매체 중 어느 위치인지 보여준다.
    """
    df = group_kpis.dropna(subset=["acost"]).sort_values("acost", ascending=False)
    names = df[group_col].tolist()
    values = df["acost"].tolist()
    colors = [highlight_color if n == highlight_name else "#D8D4CC" for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=values,
        marker_color=colors,
        hovertemplate="%{x}<br>광고비: %{y:,.0f}원<extra></extra>",
    ))
    fig.update_layout(
        height=170, margin=dict(l=28, r=8, t=4, b=46),
        yaxis=dict(type="log", showgrid=True, gridcolor="#EFEDE7", tickfont=dict(size=9), title=None),
        xaxis=dict(tickfont=dict(size=9), tickangle=-35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.3,
    )
    return fig


# ═══════════════════════════════════════════════════
# 상세 페이지용 소형 차트
# ═══════════════════════════════════════════════════

def _compact_layout(fig, height=220):
    """상세 페이지 내 소형 차트 공통 레이아웃."""
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Pretendard, sans-serif", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10),
        ),
    )
    return fig


def make_media_cvr_bars(media_cvr_df: pd.DataFrame, current_media: str) -> go.Figure:
    """매체별 CVR 수평 바 차트 (매체확장용)."""
    df = media_cvr_df.copy()
    # 한글 매체명 적용
    df["media_kr"] = df["media"].map(
        lambda m: MEDIA_NAME_MAP.get(m, m)
    )
    colors = [
        "#4A6BAA" if row["is_current"] else "#DCE6F2"
        for _, row in df.iterrows()
    ]
    text_colors = [
        "white" if row["is_current"] else "#4A6BAA"
        for _, row in df.iterrows()
    ]

    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{row['media_kr']} (현재)" if row["is_current"] else row["media_kr"]
        fig.add_trace(go.Bar(
            y=[label], x=[row["cvr"]],
            orientation="h",
            marker_color=colors[i],
            text=f"{row['cvr']:.1f}%",
            textposition="auto",
            textfont=dict(color=text_colors[i], size=12, family="monospace"),
            showlegend=False,
        ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(title="CVR (%)", showgrid=True, gridcolor="#EAE5D3"),
        yaxis=dict(autorange="reversed"),
    )
    return _compact_layout(fig, height=max(160, len(df) * 36 + 60))


def make_daily_kpi_trend(daily_df: pd.DataFrame) -> go.Figure:
    """일별 CVR/클릭 추이 듀얼 축 차트 (승급추진용)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=daily_df["date"], y=daily_df["clk"],
        name="클릭수",
        marker_color="#DCE6F2",
        opacity=0.7,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["cvr"],
        name="CVR",
        mode="lines+markers",
        line=dict(color="#E0A437", width=2.5),
        marker=dict(size=5, color="#E0A437"),
    ), secondary_y=True)
    fig.update_yaxes(title_text="클릭수", secondary_y=False)
    fig.update_yaxes(title_text="CVR (%)", secondary_y=True)
    fig.update_xaxes(title="")
    return _compact_layout(fig, height=250)


def make_flow_bubble_chart(
    items_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    x_med: float,
    y_med: float,
):
    """유형별 운영 흐름 대시보드 버블 차트 (사분면 액션 색상 + 중앙값 기준선)."""
    from src.config import FLOW_ACTION_COLORS, FLOW_ACTION_LABELS

    fig = go.Figure()
    if items_df.empty:
        return fig

    spend = items_df["acost"]
    min_s, max_s = spend.min(), spend.max()
    if max_s == min_s:
        radius = pd.Series([24] * len(items_df), index=items_df.index)
    else:
        radius = 18 + (spend - min_s) / (max_s - min_s) * 46

    for action in ["urgent", "stable", "expand", "hold"]:
        sub = items_df[items_df["action"] == action]
        if sub.empty:
            continue
        color = FLOW_ACTION_COLORS[action]
        fig.add_trace(go.Scatter(
            x=sub[x_metric],
            y=sub[y_metric],
            mode="markers",
            marker=dict(
                size=radius.loc[sub.index],
                color=color,
                opacity=0.7,
                line=dict(width=1.5, color=color),
            ),
            name=FLOW_ACTION_LABELS[action],
            customdata=list(zip(sub["ads_name"], [FLOW_ACTION_LABELS[action]] * len(sub), sub["acost"])),
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                f"{x_label} : %{{x}}<br>{y_label} : %{{y}}<br>"
                "지출액 : %{customdata[2]:,.0f}원<extra></extra>"
            ),
        ))

    fig.add_vline(x=x_med, line_dash="dash", line_color=COLORS["neutral"])
    fig.add_hline(y=y_med, line_dash="dash", line_color=COLORS["neutral"])

    fig.update_layout(
        height=320,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig


def make_click_decline_bars(
    decline_data: pd.DataFrame,
) -> go.Figure:
    """D+N별 코호트 평균 대비 이 광고 클릭 비율(%) 바 차트 (즉시조치용)."""
    fig = go.Figure()
    x_labels = [f"D+{int(d)}" for d in decline_data["campaign_n_day"]]
    # 코호트 대비 비율 바
    fig.add_trace(go.Bar(
        x=x_labels,
        y=decline_data["vs_cohort_pct"],
        name="코호트 대비",
        marker_color="#C25E55",
        text=[
            (f"{v:.2f}%" if abs(v) < 1 else f"{v:.1f}%") if pd.notna(v) else "-"
            for v in decline_data["vs_cohort_pct"]
        ],
        textposition="outside",
        textfont=dict(color="#9A4842", size=12, family="monospace"),
        customdata=list(zip(decline_data["click_cnt"].values, decline_data["cohort_avg"].values)),
        hovertemplate="%{x}<br>코호트 대비: %{y:.1f}%<br>이 광고: %{customdata[0]:,}건<br>유형 평균: %{customdata[1]:,.0f}건<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=""),
        yaxis=dict(title="유형 평균 대비 (%)"),
        barmode="group",
    )
    return _compact_layout(fig)


