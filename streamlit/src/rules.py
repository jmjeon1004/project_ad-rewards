"""룰 기반 운영 판단 — 긴급/관심/모니터링"""
import pandas as pd
import numpy as np

from src.metrics import calc_daily_kpis
from src.config import fmt_won_man


def classify_operation_status(
    ad_summary: pd.DataFrame,
    model_scores: pd.DataFrame | None = None,
) -> dict:
    """운영 상태 분류 — 품질 점수(M1 등급) 기반: S/A/B 정상, C 주의, D 긴급"""
    df = ad_summary.copy()

    # ── 모델 점수 병합 (이미 ad_summary에 있으면 재병합 생략) ──
    if model_scores is not None and not model_scores.empty:
        merge_cols = ["ads_idx"]
        if "m1_grade" not in df.columns and "m1_grade" in model_scores.columns:
            merge_cols.append("m1_grade")
        if len(merge_cols) > 1:
            df = df.merge(model_scores[merge_cols], on="ads_idx", how="left")
    if "m1_grade" not in df.columns:
        df["m1_grade"] = np.nan

    has_model = df["m1_grade"].notna()

    # ── 품질 점수 등급 기반 분류 ──
    # 긴급: D등급
    df["is_urgent"] = has_model & (df["m1_grade"] == "D")

    # 주의: C등급
    df["is_caution"] = has_model & (df["m1_grade"] == "C")

    # 정상: S/A/B등급
    df["is_normal"] = has_model & df["m1_grade"].isin(["S", "A", "B"])

    # ── 모델 미적용 광고는 집계에서 제외 ──
    total_all = len(df)
    model_count = int(has_model.sum())
    normal_count = int(df["is_normal"].sum())

    return {
        "total": model_count,
        "total_all": total_all,
        "normal_count": normal_count,
        "normal_pct": normal_count / model_count * 100 if model_count > 0 else 0,
        "urgent_count": int(df["is_urgent"].sum()),
        "caution_count": int(df["is_caution"].sum()),
        "unscored_count": total_all - model_count,
        "model_coverage_pct": model_count / total_all * 100 if total_all > 0 else 0,
        "urgent_ads": df[df["is_urgent"]].nlargest(10, "clk"),
        "caution_ads": df[df["is_caution"]].head(10),
    }


def calc_dependency_ratio(ad_summary: pd.DataFrame) -> float:
    """TOP 10 의존도 = 완료수 상위 10개 합 / 전체 완료 합"""
    total_turn = ad_summary["turn"].sum()
    if total_turn == 0:
        return 0
    top10_turn = ad_summary.nlargest(10, "turn")["turn"].sum()
    return top10_turn / total_turn * 100


def classify_quadrant(
    group_kpis: pd.DataFrame,
    x_metric: str = "cpa",
    y_metric: str = "margin_rate",
) -> dict:
    """매체별 사분면 분류 → 4종 카드 데이터"""
    df = group_kpis.dropna(subset=[x_metric, y_metric]).copy()
    if df.empty:
        return {"loss": [], "expensive": [], "best": [], "low_eff": []}

    avg_x = df[x_metric].mean()
    is_lower_better = (x_metric == "cpa")

    result = {"loss": [], "expensive": [], "best": [], "low_eff": []}

    for _, row in df.iterrows():
        name = row.iloc[0]  # group_col name
        x_val = row[x_metric]
        y_val = row[y_metric]

        if is_lower_better:
            x_good = x_val <= avg_x
        else:
            x_good = x_val >= avg_x

        y_good = y_val >= df[y_metric].mean()

        if x_good and y_good:
            result["best"].append(row)
        elif not x_good and y_good:
            result["expensive"].append(row)
        elif x_good and not y_good:
            result["low_eff"].append(row)
        else:
            result["loss"].append(row)

    return result


def generate_alerts(group_kpis: pd.DataFrame, group_col: str) -> list[dict]:
    """룰 기반 운영 알림 생성"""
    alerts = []
    if group_kpis.empty:
        return alerts

    avg_cvr = group_kpis["cvr"].mean()

    for _, row in group_kpis.iterrows():
        name = row[group_col]
        cvr = row.get("cvr", 0)
        margin_rate = row.get("margin_rate", 0)

        if pd.notna(margin_rate) and margin_rate < 0:
            alerts.append({
                "level": "red",
                "icon": "🔴",
                "text": f"주의: {name} — 마진율 {margin_rate:.1f}% (비용 초과)",
            })
        elif pd.notna(cvr) and cvr < avg_cvr * 0.7:
            alerts.append({
                "level": "yellow",
                "icon": "🟡",
                "text": f"관찰: {name} — CVR {cvr:.1f}% (평균 대비 낮음)",
            })
        elif pd.notna(cvr) and pd.notna(margin_rate) and cvr >= avg_cvr and margin_rate > 10:
            alerts.append({
                "level": "green",
                "icon": "🟢",
                "text": f"양호: {name} — CVR {cvr:.1f}% / 마진율 {margin_rate:.1f}% 안정",
            })

    # 정렬: red > yellow > green
    order = {"red": 0, "yellow": 1, "green": 2}
    alerts.sort(key=lambda x: order.get(x["level"], 3))
    return alerts


def _stability_streak(daily: pd.DataFrame, col: str, threshold: float = 0.05) -> int:
    """최근일부터 역순으로 일별 변화율이 threshold 이내인 연속 일수."""
    vals = daily.sort_values("rpt_time_date")[col].dropna().to_numpy()
    if len(vals) < 2:
        return len(vals)
    streak = 1
    for i in range(len(vals) - 1, 0, -1):
        prev = vals[i - 1]
        if prev == 0:
            break
        if abs(vals[i] - prev) / abs(prev) <= threshold:
            streak += 1
        else:
            break
    return streak


def _decline_day(daily: pd.DataFrame, col: str, window: int = 14) -> int | None:
    """최근 window일 중 col이 가장 크게 악화(상승)된 날짜로부터 며칠 전인지."""
    d = daily.sort_values("rpt_time_date").tail(window).dropna(subset=[col])
    if len(d) < 2:
        return None
    vals = d[col].to_numpy()
    dates = d["rpt_time_date"].to_numpy()
    diffs = np.diff(vals)
    idx = int(np.argmax(diffs))
    delta = pd.Timestamp(dates[-1]) - pd.Timestamp(dates[idx + 1])
    return max(delta.days, 0)


def _period_label(period: str, filtered: pd.DataFrame) -> str:
    """카드 서브타이틀용 기간 라벨 (예: '전체 30일'). 이미 일수가 포함된 라벨은 그대로 둔다."""
    n_days = filtered["rpt_time_date"].nunique()
    return period if "일" in period else f"{period} {n_days}일"


def _spend_rank_insight(group_kpis: pd.DataFrame, group_col: str, name: str, acost: float) -> tuple[str, float]:
    """전체 매체 중 광고비 순위/비중 인사이트 텍스트 생성. (텍스트, 비중%) 반환.

    rank=1이 광고비 최대. 매체가 상위권이면 '상위 N위', 하위권이면 '하위 N위'로
    실제 위치와 일치하게 표기한다(예: 9개 중 2위 매체를 '하위 8위'로 뒤집어 표기하지 않음).
    """
    n_media = len(group_kpis)
    total_acost = group_kpis["acost"].sum()
    rank = int((group_kpis["acost"] > acost).sum()) + 1  # 1위 = 광고비 최대
    share_pct = acost / total_acost * 100 if total_acost else 0
    if rank == n_media:
        return f"광고비 {fmt_won_man(acost)} — 전체 매체 중 가장 작은 비중", share_pct
    if rank <= (n_media + 1) // 2:
        return f"광고비 {fmt_won_man(acost)} — 전체 매체 중 상위 {rank}위 (비중 {share_pct:.1f}%)", share_pct
    return f"광고비 {fmt_won_man(acost)} — 전체 매체 중 하위 {n_media - rank + 1}위 (비중 {share_pct:.1f}%)", share_pct


def build_media_insight_cards(
    quadrants: dict,
    group_kpis: pd.DataFrame,
    filtered: pd.DataFrame,
    group_col: str,
    period: str = "전체",
    min_trend_days: int = 2,
    trend_df: pd.DataFrame | None = None,
) -> dict:
    """매체별 사분면 분류 결과로 인사이트 카드 4종(최우수/비효율/단가협업/소재개선) 데이터 생성.

    어떤 매체가 최우수/비효율인지는 선택한 관측 기간(filtered)의 group_kpis로 판단하고,
    카드 내 마진율 추이 차트도 우선 filtered의 일별 데이터로 그려 선택한 기간을 그대로
    반영한다(예: 최근 7일/전체/사용자 지정마다 다른 추이가 나와야 함).
    다만 "최근 1일"처럼 선택 기간 자체가 하루뿐이라 filtered만으로 추이를 그릴 수 없으면
    trend_df(미지정 시 filtered, 기간 필터 이전의 더 긴 일별 시리즈)로 대체한다.
    그래도 min_trend_days일 미만이면(예: 틱톡처럼 전환이 특정 하루에만 몰려 있어 추이를
    그릴 수 없는 경우) 추이 차트 대신 전체 매체 광고비 순위 차트 + 인사이트 문구로
    대체한다(mode="spend").
    """
    result = {"best": None, "loss": None, "loss_empty": None, "price_coop": None, "material": None}

    trend_source = trend_df if trend_df is not None else filtered
    overall_margin_avg = group_kpis["margin_rate"].mean()
    overall_cvr_avg = group_kpis["cvr"].mean()
    period_label = _period_label(period, filtered)

    # Card① 최우수 — best 사분면 중 마진율 최고
    best_rows = quadrants.get("best", [])
    if best_rows:
        row = max(best_rows, key=lambda r: r["margin_rate"])
        name = row[group_col]
        daily = calc_daily_kpis(filtered[filtered[group_col] == name])
        n_valid = int(daily["margin_rate"].notna().sum())
        if n_valid < min_trend_days:
            # 선택한 기간만으로는 추이를 그릴 수 없을 때만(예: 최근 1일) 더 긴 시리즈로 대체
            daily = calc_daily_kpis(trend_source[trend_source[group_col] == name])
            n_valid = int(daily["margin_rate"].notna().sum())
        if n_valid >= min_trend_days:
            streak = _stability_streak(daily, "margin_rate")
            badge = f"{streak}일 연속 변동 없이 안정" if streak >= 2 else "안정적인 추이 유지 중"
            result["best"] = dict(
                icon="☆", title=f"최우수·{name}", media_name=name, mode="trend",
                quote="안정적인가, 더 키울 수 있나",
                daily_df=daily, kpi_col="margin_rate", color="green",
                badge_text=badge, button_label="예산 확대 검토",
            )
        else:
            spend_insight, share_pct = _spend_rank_insight(group_kpis, group_col, name, row["acost"])
            spend_action = (
                "비중이 이미 커서 확대보다는 효율 유지·구조 점검이 우선"
                if share_pct >= 30
                else "여기서 더 늘릴 여지가 있는지보다, 비중이 작아 효과가 제한적임을 먼저 확인"
            )
            result["best"] = dict(
                icon="☆", title=f"최우수·{name}", media_name=name, mode="spend",
                period_label=period_label, color="green",
                group_kpis=group_kpis, group_col=group_col,
                spend_insight=spend_insight,
                spend_action=spend_action,
            )

    # Card② 비효율 — loss 사분면 중 마진율 최저
    loss_rows = quadrants.get("loss", [])
    if loss_rows:
        row = min(loss_rows, key=lambda r: r["margin_rate"])
        name = row[group_col]
        daily = calc_daily_kpis(filtered[filtered[group_col] == name])
        n_valid = int(daily["margin_rate"].notna().sum())
        if n_valid < min_trend_days:
            # 선택한 기간만으로는 추이를 그릴 수 없을 때만(예: 최근 1일) 더 긴 시리즈로 대체
            daily = calc_daily_kpis(trend_source[trend_source[group_col] == name])
            n_valid = int(daily["margin_rate"].notna().sum())
        if n_valid >= min_trend_days:
            days_ago = _decline_day(daily, "cpa")
            badge = (f"{days_ago}일 전 CPA 급등과 동시 하락" if days_ago is not None
                      else "CPA 상승과 마진 하락 동시 진행")
            result["loss"] = dict(
                icon="⚠", title=f"비효율·{name}", media_name=name, mode="trend",
                quote="언제 나빠졌나, 뭘 해야 하나",
                daily_df=daily, kpi_col="margin_rate", color="red",
                badge_text=badge, button_label="단가 재협의",
            )
        else:
            spend_insight, share_pct = _spend_rank_insight(group_kpis, group_col, name, row["acost"])
            spend_action = (
                "비중이 커서 정밀 진단이 필요 — 단가 재협의를 우선 검토"
                if share_pct >= 30
                else f"{name} 예산을 재배분해도 효과가 미미합니다. 다른 매체 성과 개선을 우선 검토하세요."
            )
            result["loss"] = dict(
                icon="⚠", title=f"비효율·{name}", media_name=name, mode="spend",
                period_label=period_label, color="red",
                group_kpis=group_kpis, group_col=group_col,
                spend_insight=spend_insight,
                spend_action=spend_action,
            )
    else:
        result["loss_empty"] = "비효율 매체 없음 — 모든 매체가 CPA·마진율 기준 평균 이상"

    # Card③ 단가 협업 — low_eff 사분면 전체 (마진율 평균비 테이블)
    low_eff_rows = quadrants.get("low_eff", [])
    if low_eff_rows:
        df = pd.DataFrame(low_eff_rows)
        grp_avg = df["margin_rate"].mean()
        rows_df = pd.DataFrame({
            "media": df[group_col],
            "margin_rate": df["margin_rate"],
            "cpa": df["cpa"],
            "avg_ratio": (df["margin_rate"] - grp_avg) / grp_avg * 100 if grp_avg else np.nan,
        }).sort_values("avg_ratio", ascending=False).reset_index(drop=True)
        result["price_coop"] = dict(
            title=f"단가 협업·{len(rows_df)}개 매체",
            subtitle=" · ".join(rows_df["media"].tolist()),
            rows=rows_df,
        )

    # Card④ 소재 개선 — expensive 사분면 중 CPA 최고
    exp_rows = quadrants.get("expensive", [])
    if exp_rows:
        row = max(exp_rows, key=lambda r: r["cpa"])
        name = row[group_col]
        margin_val = row["margin_rate"]
        cvr_val = row["cvr"]
        margin_caption = ("평균 대비 매우 양호" if margin_val >= overall_margin_avg * 1.2
                           else "평균 대비 양호")
        if pd.notna(cvr_val) and cvr_val > overall_cvr_avg:
            cvr_caption = f"평균({overall_cvr_avg:.1f}%)보다는 높으나 정체"
        else:
            cvr_caption = f"평균({overall_cvr_avg:.1f}%) 대비 낮음"
        result["material"] = dict(
            media_name=name, quote="마진은 괜찮은데 전환이 약한 이유는?",
            margin_val=margin_val, margin_caption=margin_caption,
            cvr_val=cvr_val, cvr_caption=cvr_caption,
            insight_text="전환은 낮지만 단가효율은 유지 — 소재 또는 타겟 피로 가능성",
            insight_action="소재 자체보다 타겟 범위를 먼저 점검하세요",
            button_label="소재/타겟 개선안",
        )

    return result


def classify_flow_items(items_df: pd.DataFrame, x_metric: str, y_metric: str):
    """유형별 운영 흐름 대시보드의 사분면 분류.

    x/y 중앙값과 지출액(acost) 중앙값을 기준으로 즉시조치/안정운영/확장후보/보류 4분류.
    """
    from src.config import FLOW_LOWER_IS_BETTER

    df = items_df.copy()
    if df.empty:
        df["action"] = pd.Series(dtype=object)
        return df, None, None

    x_med = df[x_metric].median()
    y_med = df[y_metric].median()
    spend_med = df["acost"].median()
    x_lower = FLOW_LOWER_IS_BETTER.get(x_metric, False)
    y_lower = FLOW_LOWER_IS_BETTER.get(y_metric, False)

    def _classify(row):
        x_good = row[x_metric] <= x_med if x_lower else row[x_metric] >= x_med
        y_good = row[y_metric] <= y_med if y_lower else row[y_metric] >= y_med
        good = x_good and y_good
        big = row["acost"] >= spend_med
        if big and not good:
            return "urgent"
        if big and good:
            return "stable"
        if not big and good:
            return "expand"
        return "hold"

    df["action"] = df.apply(_classify, axis=1)
    return df, x_med, y_med
