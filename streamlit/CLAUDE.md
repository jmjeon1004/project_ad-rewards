# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

아이브코리아 광고운영 최적화 대시보드 — Streamlit 기반 5페이지 멀티페이지 앱으로, 광고 성과 분석 + XGBoost M1 품질등급(S/A/B/C/D) + LightGBM M2 조기부진 예측 + AI-agent 자연어 질의를 결합한 운영 의사결정 지원 도구.

PRD는 `PRD_v2_4.md`(최신, 현재 코드 기준)에 있으며, 이전 버전은 `PRD_v2_3.md`/`PRD_current.md`로 보존되어 있다. 와이어프레임 이미지는 `와이어프레임/` 디렉토리에 있다. 구현 전 반드시 최신 PRD를 참조할 것.

## Architecture

Streamlit `pages/` 기반 멀티페이지 구조:

```
app.py                  # 엔트리포인트 (st.navigation)
pages/
  01_전체_광고_현황_오버뷰.py   # Page 1: 오버뷰 (필터 없음, 전체 데이터)
  02_유형별_운영_탐색.py       # Page 2: analysis_ads_type_label 기준
  03_매체별_운영_탐색.py       # Page 3: final_media 기준
  04_카테고리별_운영_탐색.py    # Page 4: category_name 기준
  05_광고_상세.py             # Page 5: ML 인사이트 → 개별 광고 상세 + 배지별 액션
src/
  config.py             # 상수, 색상, 포맷 함수, 배지 설정, CTIT_SOURCE, 기간 기본값
  init.py               # 세션 초기화 (데이터 로딩 + ML 스코어링, ensure_data_loaded)
  data_loader.py        # parquet 로딩 + @st.cache_resource
  preprocessing.py      # JOIN, 날짜/시간 변환, base table 생성
  metrics.py            # CVR, CPA, 마진율 (합계 기반 비율)
  model.py              # XGBoost M1 + LightGBM M2 파이프라인, 피처 빌드 + 예측
  ml_insight_data.py    # ML 인사이트 데이터 준비 + 배지 로직 (기회/위험)
  detail_data.py        # P5 상세 페이지 데이터 준비 (Hero, "왜?", KPI 컨텍스트)
  filters.py            # 공통 필터 렌더링 (관측 기준, 기간, 그룹, 시간대)
  charts.py             # Plotly 차트 (사분면 버블, 히트맵, 파레토, 콤보 등)
  components.py         # KPI 카드, 배너, ML 인사이트, 운영 알림 카드, 포맷 함수
  rules.py              # 룰 기반 운영 판단 (긴급/주의/정상)
  agent.py              # Gemini 2.5 Flash API 기반 AI-agent (JSON 구조화 출력)
models/                 # ML 모델 아티팩트
  model1_artifacts/     # M1 (XGBoost) — model.joblib, feature_list.json, grade_info.json 등
  model2_artifacts/     # M2 (LightGBM) — model.joblib, threshold_info.json, rule_info.json 등
  preprocessing_pipeline.joblib  # 공통 전처리 파이프라인 (OHE + TF-IDF)
outputs/                # 내보내기용 디렉토리
data/                   # 원본 parquet 12개 파일
```

## Data

`data/` 디렉토리에 parquet 파일. 핵심 JOIN 키는 `ads_idx`.

- **hourly_report.parquet** — 시간대별 광고 성과 fact table (rpt_time_clk, rpt_time_turn, rpt_time_scost, rpt_time_acost, rpt_time_cost, rpt_time_earn)
- **ad_attr_map.parquet** — 광고 속성 (유형, 카테고리, 리워드, 저장방식, 일cap)
- **ive_ad_classification.parquet** — 매체/액션 분류 (`final_media`, `final_action`)
- **ad_outcome.parquet** — 광고별 전환 결과 집계 (avg_ctit, CTIT 1순위 소스)
- **main_funnel.parquet** — 원시 퍼널 이벤트 ("오늘" 날짜 결정용)
- **ad_master_clean.parquet** — 광고 등록 마스터 (ML 피처용)
- **sched_clean.parquet** — 캠페인 스케줄 (운영 일수, 클릭)
- **finance_clean1.parquet**, **finance_clean2.parquet** — 정산 데이터
- **user_daily_activity_clean.parquet** — 유저 일별 활동

## Key Domain Rules

- 모든 지표는 **합계 기반 비율**로 계산 (행별 비율의 산술평균 금지).
  - CVR = `sum(turn) / sum(clk) * 100`
  - CPA = `sum(acost) / sum(turn)`
  - 마진율 = `(sum(acost) - sum(earn)) / sum(acost) * 100`
- 분모 0일 때 `null` 처리, UI에서 `-` 표시.
- **M1 등급 (XGBoost, 50피처)**: 등록 시점 속성 기반 품질 등급 — S(>80.7) / A(59.9~80.7) / B(40.0~59.9) / C(20.02~40.0) / D(<20.02). quantile-based percentile 스코어링.
- **M2 조기부진 (LightGBM, 101피처)**: D+3 초기 실적 기반 부진 위험 예측 — `early_click < 10` → rule_based_review, `m2_proba >= threshold` → decline_risk, 나머지 normal.
- **운영 상태**: M1 등급 기반 — S/A/B → 정상, C → 주의, D → 긴급.
- **배지 시스템**: 기회(등급경계/매체확장) + 위험(즉시조치/우선검토).
- 칩 정렬: "전체" 맨 앞 + "기타" 맨 끝 + 나머지 가나다순.

## Commands

```bash
# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py

# 특정 포트로 실행
streamlit run app.py --server.port 8502
```

## Tech Stack

- **Frontend**: Streamlit (멀티페이지, `st.navigation`)
- **Charts**: Plotly
- **Data**: Pandas + PyArrow (parquet)
- **ML**: XGBoost, LightGBM, scikit-learn, joblib
- **AI-agent**: Google Generative AI (Gemini 2.5 Flash, JSON 구조화 출력)


모든 답변은 한국어로 한다.
