# PRD 발췌 — 비전, 사용자, 데이터, 페이지

## 비전과 해결하는 문제

아이브코리아는 수백 건의 광고를 동시 운영하며 다음을 반복 판단해야 한다:

- 어떤 광고가 잘 되고 있는가 (성과 지표가 데이터 소스에 산재)
- 신규 광고의 품질을 사전에 평가할 수 있는가 (등록 시점 속성만으로 판별하는 기준 부재)
- 부진 광고를 얼마나 빨리 포착할 수 있는가 (조기 경보 체계 부재)
- 직관이 아닌 근거 기반 판단을 내릴 수 있는가

이 대시보드는 ① 성과 분석, ② ML 품질 등급(M1), ③ 조기부진 예측(M2), ④ AI Agent 자연어 질의를 하나의 인터페이스에 결합해 답한다.

## 대상 사용자 (페르소나)

| 역할 | 주요 관심사 | 사용 패턴 |
| --- | --- | --- |
| 광고 운영 매니저 | 일일 성과 변동, 긴급/주의 광고 파악 | 매일 오전 P1 오버뷰 확인 → 이상 감지 시 P2~P4 드릴다운 |
| 미디어 바이어 | 매체별 효율, 예산 배분 최적화 | P3 매체별 탐색 → 버블차트 효율 비교 → P5 매체확장 액션 |
| 운영 의사결정자 | 포트폴리오 전체 건전성, ML 인사이트 | P1 배너 확인 → ML 인사이트 탭 → AI Agent 질의 |

## 의사결정 흐름

Overview(P1, "무엇이 문제인가?") → Drill-down(P2~P4, "어디서 문제인가?") → Detail(P5, "어떻게 조치할 것인가?")

## 데이터 원천 (data/ 디렉토리, 모두 Parquet, JOIN 키는 ads_idx)

| 파일 | 역할 |
| --- | --- |
| hourly_report.parquet | 시간대별 광고 성과 fact table (클릭/전환/광고비/정산비/수익) |
| ad_attr_map.parquet | 광고 속성 (유형, 카테고리, 보상가, 저장방식, 일캡) |
| ive_ad_classification.parquet | 매체/액션 분류 (final_media, final_action) |
| ad_outcome.parquet | 전환 결과 집계, avg_ctit(CTIT 1순위 소스) |
| main_funnel.parquet | 원시 퍼널 이벤트 — "오늘" 날짜 결정용 |
| ad_master_clean.parquet | 광고 등록 마스터 (ML 피처용) |
| sched_clean.parquet | 캠페인 스케줄 (운영 일수, early_click 계산용) |
| finance_clean1/2.parquet | 정산 데이터 |
| user_daily_activity_clean.parquet | 유저 일별 활동 |

분석 기간: 2025-07-26 ~ 2025-08-25 (약 1개월). "오늘"은 실제 시스템 날짜가 아니라 `main_funnel.parquet`의 click_date 최댓값으로 결정한다.

## 전처리 파이프라인 순서

1. `build_base_table()` (`src/preprocessing.py`): hourly_report + ad_attr_map + ive_ad_classification + ad_outcome를 ads_idx 기준 LEFT JOIN, 요일/주말 파생, 매체명·유형 라벨 매핑
2. `build_ad_summary()`: 광고별 1행 집계 (clk/turn/acost/earn 합계 + cvr/cpa/margin_rate 파생)
3. `score_all_ads()` (`src/model.py`): M1(XGBoost)+M2(LightGBM) 예측 → m1_score, m1_grade, m2_proba, m2_decision
4. ad_summary + model_scores 병합 → `st.session_state`에 저장 (앱 최초 접근 시 1회, `ensure_data_loaded()`)

## 페이지 구성

| 페이지 | 파일 | 필터 기준 | 목적 |
| --- | --- | --- | --- |
| P1 오버뷰 | pages/01_전체_광고_현황_오버뷰.py | 없음(전체) | KPI, 운영 건전성 배너, 시공간 히트맵, 차원별 Top3, ML 인사이트, AI Agent |
| P2 유형별 탐색 | pages/02_유형별_운영_탐색.py | analysis_ads_type_label | 유형별 사분면/추이 드릴다운 |
| P3 매체별 탐색 | pages/03_매체별_운영_탐색.py | final_media | 매체별 효율 비교, 단가 협업 후보 발굴 |
| P4 카테고리별 탐색 | pages/04_카테고리별_운영_탐색.py | category_name | 카테고리별 운영 흐름 분류 |
| P5 광고 상세 | pages/05_광고_상세.py | 개별 ads_idx | 배지 기반 액션 카드, Hero/"왜?" 분석 |

## 캐싱 전략 요약

- 원본 데이터/베이스 테이블/ML 스코어링: `@st.cache_resource`로 앱 프로세스 수명 동안 1회만 계산, 모든 세션이 공유
- 필터링 결과 및 ML 인사이트: 필터 조합을 직렬화한 fingerprint 키로 캐싱, 동일 필터면 재계산하지 않음
