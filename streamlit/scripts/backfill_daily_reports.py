"""분석 기간 전체를 순회하며 일일 리포트를 백필해 RAG 지식베이스에 누적 저장하는 스크립트.

채팅에서 "오늘자 리포트 써줘"를 물었을 때와 동일한 경로(build_data_context +
build_daily_period_context -> generate_response -> save_daily_report)를 각 날짜에 대해 반복한다.
save_daily_report는 문서명(날짜) 기준으로 upsert하므로 재실행해도 안전하다.

Gemini 무료 티어는 gemini-2.5-flash 기준 분당 5회 요청으로 제한되어 있어, 호출 사이에
간격을 두고 429(RESOURCE_EXHAUSTED) 발생 시 재시도한다. API 오류 응답은 검증에서 걸러
저장하지 않는다(is_valid_report_result) — 오류 메시지가 리포트로 저장되는 사고 방지.

사용법:
    python scripts/backfill_daily_reports.py                # 전체 기간
    python scripts/backfill_daily_reports.py --start 2025-08-01 --end 2025-08-05
    python scripts/backfill_daily_reports.py --dry-run       # API 호출 없이 대상 일자만 확인
    python scripts/backfill_daily_reports.py --force         # 이미 저장된 날짜도 재생성
"""
import argparse
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.agent import (
    build_data_context, build_daily_period_context, format_report_markdown,
    generate_response, is_daily_report_request, is_valid_report_result,
)
from src.config import RAG_REPORTS_DIR
from src.data_loader import (
    load_hourly_report, load_ad_attr_map, load_ad_classification,
    load_ad_outcome_full, load_ad_master_clean, load_sched_clean,
)
from src.preprocessing import build_base_table, build_ad_summary, filter_by_period
from src.metrics import calc_all_kpis
from src.model import score_all_ads
from src.rag import retrieve, save_daily_report

MIN_INTERVAL_SEC = 13  # 무료 티어 RPM 5 기준 여유 있게 간격 확보
MAX_RETRIES = 4


def _generate_with_retry(query: str, ctx: str, retrieved: list) -> dict:
    """429 등 일시 오류 시 대기 후 재시도해 유효한 리포트를 받는다."""
    delay = 20
    for attempt in range(1, MAX_RETRIES + 1):
        result = generate_response(
            user_message=query, data_context=ctx, chat_history=[], retrieved_chunks=retrieved,
            use_search=False,  # 과거 날짜 백필은 최신 뉴스 grounding이 불필요 + 별도 쿼터 소모 방지
        )
        if is_valid_report_result(result):
            return result
        print(f"    응답 실패(시도 {attempt}/{MAX_RETRIES}): {str(result.get('summary'))[:80]} -> {delay}s 대기 후 재시도")
        time.sleep(delay)
        delay = min(delay * 1.5, 60)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (미지정 시 데이터 최소일)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (미지정 시 데이터 최대일)")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 대상 일자만 출력")
    parser.add_argument("--force", action="store_true", help="이미 유효하게 저장된 날짜도 재생성")
    args = parser.parse_args()

    print("데이터 로딩 중...")
    hourly = load_hourly_report()
    attr = load_ad_attr_map()
    classification = load_ad_classification()
    outcome_full = load_ad_outcome_full()
    ad_master = load_ad_master_clean()
    sched = load_sched_clean()
    outcome = outcome_full[["ads_idx", "avg_ctit"]]

    base = build_base_table(hourly, attr, classification, outcome)
    model_scores = score_all_ads(attr, ad_master, classification, outcome_full, sched)
    ms_cols = [c for c in ["ads_idx", "m1_score", "m1_grade", "m2_proba", "m2_decision"]
               if c in model_scores.columns]

    all_dates = sorted(base["rpt_time_date"].dropna().unique())
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp(all_dates[0])
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(all_dates[-1])
    target_dates = [d for d in all_dates if start <= pd.Timestamp(d) <= end]

    print(f"대상 일자: {len(target_dates)}건 ({target_dates[0]} ~ {target_dates[-1]})")

    _error_markers = ("API 호출 오류", "RESOURCE_EXHAUSTED", "응답을 생성하지 못했습니다", "응답 형식 오류")

    for i, day in enumerate(target_dates, 1):
        day_ts = pd.Timestamp(day)
        date_str = day_ts.strftime("%Y-%m-%d")
        print(f"[{i}/{len(target_dates)}] {date_str} 처리 중...", flush=True)

        report_path = os.path.join(RAG_REPORTS_DIR, f"daily_report_{date_str}.md")
        if not args.force and os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                existing = f.read()
            if not any(m in existing for m in _error_markers):
                print("  이미 유효한 리포트 존재 - 스킵 (--force로 재생성 가능)")
                continue

        day_df = filter_by_period(base, "최근 1일", day_ts)
        if len(day_df) == 0:
            print("  데이터 없음 - 스킵")
            continue

        if args.dry_run:
            continue

        day_summary = build_ad_summary(day_df)
        if model_scores is not None:
            day_summary = day_summary.merge(model_scores[ms_cols], on="ads_idx", how="left")
        kpis = calc_all_kpis(day_df)

        ctx = build_data_context(
            ad_summary=day_summary, kpis=kpis,
            page_name="일일 리포트 백필", filters_desc=f"{date_str} 단일 일자",
        )
        ctx += build_daily_period_context(base, day_ts)

        query = f"{date_str}자 일일 리포트 써줘"
        assert is_daily_report_request(query)

        retrieved = retrieve(query)
        result = _generate_with_retry(query, ctx, retrieved)

        if not is_valid_report_result(result):
            print(f"  최종 실패 - 저장하지 않음: {str(result.get('summary'))[:100]}")
            time.sleep(MIN_INTERVAL_SEC)
            continue

        report_md = format_report_markdown(result)
        ok = save_daily_report(date_str, report_md)
        print(f"  저장 {'성공' if ok else '실패(임베딩 오류 - 파일만 저장됨)'}")
        time.sleep(MIN_INTERVAL_SEC)

    print("백필 완료." if not args.dry_run else "dry-run 완료 (API 호출 없음).")


if __name__ == "__main__":
    main()
