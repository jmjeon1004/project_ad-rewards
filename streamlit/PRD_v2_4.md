# PRD v2.4 — 아이브코리아 광고운영 최적화 대시보드

> **버전**: 2.4
> **작성일**: 2026-06-30
> **상태**: 현재 구현 기준 (Production) — `app.py` 및 `src/`, `pages/` 코드를 직접 조사하여 작성
> **이전 버전**: PRD_v2_3.md (2026-05-11) — 주요 변경사항은 [부록 0](#0-v23--v24-변경-요약) 참조

---

## 목차

**Part I — 프로젝트 정의**

- [1. 비전과 목적](#1-비전과-목적)
- [2. 대상 사용자와 시나리오](#2-대상-사용자와-시나리오)
- [3. 핵심 가치 제안](#3-핵심-가치-제안)

**Part II — 시스템 아키텍처**

- [4. 기술 스택 및 실행 환경](#4-기술-스택-및-실행-환경)
- [5. 데이터 아키텍처](#5-데이터-아키텍처)
- [6. 지표 체계](#6-지표-체계)

**Part III — ML 모델**

- [7. M1: 광고 품질 등급](#7-m1-광고-품질-등급)
- [8. M2: 조기부진 예측](#8-m2-조기부진-예측)
- [9. 배지 시스템](#9-배지-시스템)

**Part IV — 사용자 경험**

- [10. 전체 네비게이션 흐름](#10-전체-네비게이션-흐름)
- [11. P1: 전체 광고 현황 오버뷰](#11-p1-전체-광고-현황-오버뷰)
- [12. P2: 유형별 운영 탐색](#12-p2-유형별-운영-탐색)
- [13. P3: 매체별 운영 탐색](#13-p3-매체별-운영-탐색)
- [14. P4: 카테고리별 운영 탐색](#14-p4-카테고리별-운영-탐색)
- [15. P5: 광고 상세](#15-p5-광고-상세)

**Part V — 공통 시스템**

- [16. 디자인 시스템](#16-디자인-시스템)
- [17. AI Agent + RAG](#17-ai-agent--rag)
- [18. 운영 규칙 엔진](#18-운영-규칙-엔진)

**Part VI — 부록**

- [0. v2.3 → v2.4 변경 요약](#0-v23--v24-변경-요약)
- [19. 설정 상수 레퍼런스](#19-설정-상수-레퍼런스)
- [20. 소스 파일 맵](#20-소스-파일-맵)

---

# Part I — 프로젝트 정의

## 1. 비전과 목적

### 1.1 해결하는 문제

아이브코리아는 수백 건의 광고를 동시 운영하며, 매일 다음과 같은 의사결정을 반복한다:

- **어떤 광고가 잘 되고 있는가?** — 성과 지표(CVR, CPA, 마진율)가 흩어진 데이터 소스에 산재
- **신규 광고의 품질을 사전에 평가할 수 있는가?** — 등록 시점 속성만으로 품질 등급을 판별하는 기준 부재
- **부진 광고를 얼마나 빨리 포착할 수 있는가?** — D+3 초기 실적만으로 부진 위험을 감지하는 조기 경보 체계 부재
- **오늘 어떤 광고부터 손대야 하는가?** — 등급/위험 신호는 있지만, 유형별로 한눈에 "조치 순서"를 보여주는 화면 부재
- **근거 기반 판단을 내릴 수 있는가?** — 직관과 경험에 의존하는 운영, 데이터 근거 기반 표준화된 판단 프레임워크 필요

### 1.2 솔루션

Streamlit 기반 5페이지 멀티페이지 대시보드로, 5가지 핵심 역량을 하나의 인터페이스에 결합한다:

```
┌────────────────────────────────────────────────────────────────────┐
│                       광고운영 최적화 대시보드                        │
│                                                                      │
│  ① 성과 분석     ② ML 품질 등급    ③ 부진 예측     ④ 운영 우선순위   │
│  (KPI, 히트맵,   (XGBoost M1:     (LightGBM M2:    (유형별 조치     │
│   버블, 추이)     S/A/B/C/D)       decline_risk)    필요 광고 정렬)  │
│                                                                      │
│       ⑤ AI Agent 자연어 질의 (Gemini 2.5 Flash + RAG + 외부 검색)    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 기대 효과

| 영역           | Before                                      | After                                                  |
| -------------- | ------------------------------------------- | ------------------------------------------------------ |
| 성과 모니터링  | 수동 엑셀 집계                              | 실시간 대시보드 + 당일 vs 평시 비교 자동 탐지          |
| 신규 광고 평가 | 감에 의존                                   | M1 품질 등급(S~D) + 점수(0~100) 객관적 판별           |
| 부진 감지      | 1~2주 후 사후 분석                          | D+3 시점 조기 경보 (M2 부진확률)                       |
| 운영 우선순위  | 담당자별 기준 상이, 어떤 것부터 볼지 불명확 | 유형별 "이동 조치 필요 광고" 4모드로 우선순위 정렬     |
| 애드혹 분석    | SQL 작성 필요                               | AI Agent 자연어 질의 + 정책 문서(RAG) + 외부 뉴스 검색 |

---

## 2. 대상 사용자와 시나리오

### 2.1 페르소나

| 역할                       | 주요 관심사                         | 사용 패턴                                                                        |
| -------------------------- | ----------------------------------- | -------------------------------------------------------------------------------- |
| **광고 운영 매니저** | 일일 성과 변동, 긴급/주의 광고 파악 | P1 오버뷰 "당일 vs 평시" 모드로 이상 감지 → P2~P4 드릴다운                      |
| **미디어 바이어**    | 매체별 효율, 예산 배분 최적화       | P3 매체별 탐색 → 버블차트 + 매체 인사이트 카드 → P5 매체확장 액션              |
| **운영 의사결정자**  | 포트폴리오 전체 건전성, ML 인사이트 | P1 "모델 결과 기반 운영 추천" 배너 → P2 운영 우선순위 대시보드 → AI Agent 질의 |

### 2.2 일상 워크플로우

```
[전체 점검]
P1 오버뷰 (전체 기간 모드) → KPI 카드 + 히트맵으로 패턴 확인
                           → 모델 결과 기반 운영 추천 배너 확인

[당일 이상 감지]
P1 오버뷰 (당일 vs 평시 모드) → 시간대별 CVR 추이에서 급락 시점 클릭
                              → 해당 시간대 이상 광고 리스트 확인
                              → 차원별 Top3로 유형/매체/카테고리 단위 이상 파악

[우선순위 기반 조치]
P2 유형별 탐색 → "이동 조치 필요 광고" 4모드(즉시조치/우선검토/매체확장/승급추진) 확인
              → 유형별로 그룹화된 리스트에서 "상세" 클릭 → P5 진입

[액션 실행]
P5 광고 상세 → 배지별 액션 카드 (매체확장/승급추진/즉시조치/우선검토)
            → 구체적 액션 실행 (결재 요청, 예산 사전 확보, 일시 중단 등)

[애드혹 분석]
각 페이지 하단 AI Agent → "가입형 광고 중 CVR이 가장 낮은 매체는?",
                         "오늘 CVR이 왜 떨어졌어?" (외부 뉴스 검색 자동 활용) 등 자연어 질의
```

---

## 3. 핵심 가치 제안

### 3.1 다섯 가지 축

| 축                       | 설명                                                      | 주요 기능                                                |
| ------------------------ | --------------------------------------------------------- | -------------------------------------------------------- |
| **성과 분석**      | 시간대/요일/차원별 광고 성과를 다각도로 시각화            | KPI 카드, 히트맵, 버블차트, Pareto, 그룹 요약 테이블     |
| **품질 등급 (M1)** | 등록 시점 속성으로 광고 품질을 5등급 예측                 | XGBoost 48피처 → S/A/B/C/D, 0~100점                     |
| **부진 예측 (M2)** | D+3 초기 실적으로 조기부진 위험 예측                      | LightGBM 101피처 → 부진확률 0~1                         |
| **운영 우선순위**  | 유형별로 "지금 봐야 할 광고"를 모드별로 정렬              | P2 "이동 조치 필요 광고" 4모드 대시보드                  |
| **AI 질의**        | 필터 범위 데이터 + 정책 문서 + 외부 뉴스 기반 자연어 분석 | Gemini 2.5 Flash, RAG 지식베이스, Google Search 그라운딩 |

### 3.2 의사결정 흐름: Overview → Drill-down/Priority → Detail → Action

```
P1 전체 오버뷰 ──────────────────────────── "무엇이 문제인가?"
     │                                       (KPI, 히트맵, 운영 추천 배너)
     ├── P2 유형별 (+ 운영 우선순위 대시보드) ─┐
     ├── P3 매체별 (+ 매체 인사이트 카드 4종) ─┼─ "어디서, 무엇부터 문제인가?"
     └── P4 카테고리별 (+ Pareto)            ─┘
                          │
                          └── P5 광고 상세 ──── "어떻게 조치할 것인가?"
                               (배지, 액션 카드, 모달)
```

---

# Part II — 시스템 아키텍처

## 4. 기술 스택 및 실행 환경

### 4.1 의존성

| 영역   | 패키지                | 역할                                                                                            |
| ------ | --------------------- | ----------------------------------------------------------------------------------------------- |
| UI     | `streamlit`         | 멀티페이지 네비게이션,`st.Page`, `st.navigation`                                            |
| 차트   | `plotly`            | 인터랙티브 시각화 (히트맵, 버블, 파레토)                                                        |
| 데이터 | `pandas`, `numpy` | DataFrame/수치 처리                                                                             |
| I/O    | `pyarrow`           | Parquet 읽기 (pushdown filter)                                                                  |
| ML     | `xgboost`           | M1 모델 (광고 품질 등급)                                                                        |
| ML     | `lightgbm`          | M2 모델 (조기부진 예측)                                                                         |
| ML     | `scikit-learn`      | OHE, TF-IDF, LabelEncoder                                                                       |
| 직렬화 | `joblib`            | 모델/파이프라인 저장·로드                                                                      |
| AI     | `google-genai`      | Gemini 2.5 Flash (응답 생성),`gemini-embedding-001` (RAG 임베딩), Google Search 그라운딩 도구 |
| 환경   | `python-dotenv`     | 로컬 개발용`.env` 로드 (RAG 인덱스 빌드 스크립트에서 사용)                                    |

### 4.2 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py                     # 기본 포트 (8501)
streamlit run app.py --server.port 8502  # 특정 포트
```

### 4.3 인증 / 환경변수

| 키                 | 위치                                         | 용도                                                                              |
| ------------------ | -------------------------------------------- | --------------------------------------------------------------------------------- |
| `GEMINI_API_KEY` | **Streamlit Secrets** (`st.secrets`) | 런타임 AI Agent / RAG 검색 —`src/agent.py`, `src/rag.py`                     |
| `GEMINI_API_KEY` | `.env` (로컬)                              | RAG 인덱스 빌드 스크립트(`scripts/build_rag_index.py`) 전용 — 배포 환경과 분리 |

> v2.3까지는 `.env` 단일 소스였으나, 배포 환경(Streamlit Community Cloud 등)에서는 `st.secrets`를 사용하도록 전환됨. RAG 인덱스는 사전 계산된 산출물이므로 빌드 스크립트만 `.env`를 그대로 사용한다.

### 4.4 디렉토리 구조

```
streamlit/
├── app.py                              # 엔트리포인트 (네비게이션 설정)
├── PRD_v2_4.md                         # 이 문서 (PRD_v2_3.md는 이전 버전으로 보존)
├── CLAUDE.md                           # 개발 가이드
│
├── pages/                              # Streamlit 멀티페이지
│   ├── 01_전체_광고_현황_오버뷰.py       # P1: 오버뷰 (전체 기간 / 당일 vs 평시)
│   ├── 02_유형별_운영_탐색.py           # P2: 유형 분석 + 운영 우선순위 대시보드
│   ├── 03_매체별_운영_탐색.py           # P3: 매체 분석 + 매체 인사이트 카드 4종
│   ├── 04_카테고리별_운영_탐색.py        # P4: 카테고리 분석 + Pareto
│   └── 05_광고_상세.py                 # P5: 광고 상세 + 배지별 액션
│
├── src/                                # 비즈니스 로직 모듈
│   ├── config.py                       # 상수, 색상, 포맷 함수, RAG 경로
│   ├── init.py                         # 세션 초기화 (데이터 + ML 스코어링)
│   ├── data_loader.py                  # Parquet 로더 (@st.cache_resource)
│   ├── preprocessing.py                # JOIN, 필터링, 베이스 테이블
│   ├── metrics.py                      # KPI 계산 + 프로세스 전역 캐시
│   ├── model.py                        # ML 파이프라인 (피처 빌드 + 예측)
│   ├── ml_insight_data.py              # ML 인사이트 데이터 준비 + 배지 로직
│   ├── detail_data.py                  # P5 상세 페이지 데이터 준비
│   ├── filters.py                      # 필터 UI (기간, 그룹, 시간대, 보조)
│   ├── charts.py                       # Plotly 차트 함수 모음
│   ├── components.py                   # 재사용 UI 컴포넌트
│   ├── rules.py                        # 룰 기반 운영 판단 + 매체 인사이트 카드 빌더
│   ├── agent.py                        # Gemini AI Agent (RAG + Google Search 연동)
│   └── rag.py                          # RAG 검색 모듈 (지식베이스 임베딩 검색)
│
├── scripts/
│   └── build_rag_index.py              # data/rag_docs/*.md → RAG 인덱스 빌드 (수동 실행)
│
├── data/                               # 원천 Parquet 파일 + rag_docs/*.md
├── models/                             # ML 모델 아티팩트
│   ├── model1_artifacts/               # M1 (XGBoost)
│   ├── model2_artifacts/               # M2 (LightGBM)
│   ├── rag_artifacts/                  # RAG 인덱스 (index.parquet)
│   └── preprocessing_pipeline.joblib   # 공통 전처리 파이프라인
│
├── assets/                             # 정적 리소스 (logo.png)
├── outputs/                            # 내보내기용 디렉토리
└── 와이어프레임/                         # UI 와이어프레임 이미지
```

---

## 5. 데이터 아키텍처

### 5.1 원천 데이터 파일

모든 파일은 `data/` 디렉토리에 Parquet 형식으로 저장된다. 핵심 JOIN 키는 `ads_idx`이다.

| #  | 파일                                                   | 역할                                             | 주요 컬럼                                                                                                                                                                    |
| -- | ------------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `hourly_report.parquet`                              | **시간대별 광고 성과 fact table**          | `rpt_time_date`, `rpt_time_time`, `rpt_time_clk`, `rpt_time_turn`, `rpt_time_acost`, `rpt_time_scost`, `rpt_time_earn`, `ads_idx`, `mda_idx`               |
| 2  | `ad_attr_map.parquet`                                | 광고 속성 (유형, 카테고리, 보상 등)              | `ads_idx`, `ads_name`, `category_name`, `ads_reward_price`, `reward_band`, `ads_rejoin_type`, `ads_sdate`, `ads_edate`                                       |
| 3  | `ive_ad_classification.parquet`                      | 매체/액션 분류                                   | `ads_idx`, `final_media`, `final_action`                                                                                                                               |
| 4  | `ad_outcome.parquet`                                 | 전환 결과 (CTIT 1순위 소스)                      | `ads_idx`, `avg_ctit`, `is_valid_click10`, `click_cnt`                                                                                                               |
| 5  | `main_funnel.parquet`                                | 원시 퍼널 이벤트 ("오늘" 날짜 결정용)            | `click_date`                                                                                                                                                               |
| 6  | `ad_master_clean.parquet`                            | 광고 등록 마스터 (ML 피처)                       | `ads_idx`, `regdate`, `ads_os_type`, `ads_require_adid`, `action_target_cnt`, `mentioned_media_cnt`, `target_media_cnt`, `ads_summary`, `ads_reward_price` |
| 7  | `sched_clean.parquet`                                | 캠페인 스케줄 (운영 일수, 클릭)                  | `ads_idx`, `click_date`, `click_cnt`, `campaign_n_day`, `ads_sdate`                                                                                                |
| 8  | `finance_clean1.parquet`, `finance_clean2.parquet` | 정산 데이터                                      | —                                                                                                                                                                           |
| 9  | `user_daily_activity_clean.parquet`                  | 유저 일별 활동                                   | —                                                                                                                                                                           |
| 10 | `rag_docs/*.md`                                      | AI Agent RAG 지식베이스 원문 (PRD/룰/ML 정의 등) | `## ` 헤딩 단위로 청크 분할                                                                                                                                                |

> **v2.3 대비 제거**: `ads_join_info_labeled.parquet` 기반 `price_agg`(매체단가 vs 광고주단가 합계) 파이프라인은 완전히 제거됨. 관련 스크립트(`scripts/precompute_price_agg.py`)도 삭제 대상이다. 이에 따라 `settle_efficiency`(광고비 대비 정산수익률) 지표도 코드에서 제거되었고, 모든 사분면·알림 로직은 `margin_rate` 기준으로 통일되었다.

### 5.2 분석 기간

```python
ANALYSIS_DATE_START = "2025-07-26"
ANALYSIS_DATE_END   = "2025-08-25"   # 약 1개월
```

`hourly_report.parquet` 로드 시 PyArrow pushdown filter로 해당 범위만 읽어 메모리 사용량을 최소화한다.

### 5.3 "오늘" 날짜 기준

"오늘"은 `main_funnel.parquet`의 `click_date` 최댓값(`load_today_date()`)으로 결정한다. 실제 시스템 날짜가 아닌 데이터 기준 날짜를 사용하여 분석의 일관성을 보장한다.

### 5.4 전처리 파이프라인

```
                    ┌── load_hourly_report()
                    ├── load_ad_attr_map()
  데이터 로딩 ────── ├── load_ad_classification()
  (@st.cache_      └── load_ad_outcome()
   resource)
                         │
                         ▼
              build_base_table()           ← src/preprocessing.py
              ─────────────────
              1. hourly_report + ad_attr_map       (LEFT JOIN on ads_idx)
              2. + ive_ad_classification            (LEFT JOIN on ads_idx)
              3. + ad_outcome                       (LEFT JOIN on ads_idx)
              4. 요일 파생: weekday, weekday_kr (월~일), is_weekend
              5. final_action → analysis_ads_type_label 매핑 (ACTION_TYPE_MAP)
              6. final_media → 한글 매체명 매핑 (MEDIA_NAME_MAP)
              7. category_name: "선택안함" → "기타"
              8. 결측 → "기타"
                         │
                         ▼
              build_ad_summary()           ← 광고별 1행 집계
              ──────────────────
              groupby(ads_idx) → clk, turn, acost, earn 합계
              파생: cvr, cpa, margin, margin_rate
                         │
                         ▼
              score_all_ads()              ← src/model.py
              ────────────────
              M1(XGBoost) + M2(LightGBM) 예측 → m1_score, m1_grade, m2_proba, m2_decision
                         │
                         ▼
              ad_summary + model_scores 병합 → session_state 저장
```

### 5.5 세션 초기화 (`ensure_data_loaded()`)

앱 최초 접근 시 `src/init.py`의 `ensure_data_loaded()`가 1회 실행되며, 결과를 `st.session_state`에 저장한다.

| session_state 키                                                            | 내용                                   |
| --------------------------------------------------------------------------- | -------------------------------------- |
| `base`                                                                    | 시간대별 성과 base table (전처리 완료) |
| `today`                                                                   | 데이터 기준 "오늘" 날짜                |
| `ad_summary`                                                              | 광고별 1행 집계 + ML 스코어 병합       |
| `attr`, `classification`, `ad_master`, `sched`, `ad_outcome_full` | 원본 데이터                            |
| `model_scores`                                                            | ML 스코어링 결과 (M1+M2)               |
| `_core_data_loaded`                                                       | 로딩 완료 플래그                       |

### 5.6 캐싱 전략 — v2.4 변경점

| 계층                            | 메커니즘                                                                   | 수명                                        | 적용                                                             |
| ------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| **데이터 로드**           | `@st.cache_resource`                                                     | 앱 프로세스 수명                            | 모든 parquet 로더, 모델 아티팩트, RAG 클라이언트/인덱스          |
| **기본 테이블**           | `@st.cache_resource`                                                     | 앱 프로세스 수명                            | `build_base_table()`, `build_ad_summary()`                   |
| **ML 스코어링**           | `@st.cache_resource`                                                     | 앱 프로세스 수명                            | `score_all_ads()`                                              |
| **필터링/지표 계산 결과** | **프로세스 전역 dict 캐시** (`_GLOBAL_CACHE` + `threading.Lock`) | 앱 프로세스 수명, fingerprint 무효화 시까지 | `_cached_compute()`, `_cached_filter()` (`src/metrics.py`) |
| **ML 인사이트**           | 프로세스 전역 메모이제이션 (session_state가 아닌 모듈 전역 dict)           | fingerprint 변경 시 갱신                    | `get_cached_ml_insight_data()`                                 |

> **변경 핵심**: v2.3까지 `_cached_compute`/`_cached_filter`는 `st.session_state`에 결과를 저장해 **세션(사용자)별로 재계산**했다. v2.4는 이를 프로세스 전역 `dict`(`_GLOBAL_CACHE`)로 전환하여, 원본 데이터가 앱 수명 동안 불변이라는 전제하에 **동일 fingerprint면 다른 세션의 사용자라도 캐시를 공유**한다. 다중 사용자 환경에서 반복 계산 비용을 제거하기 위한 성능 최적화다. `@st.fragment`는 더 이상 사용하지 않으며(P2~P4 모두 일반 페이지 흐름), P5는 광고별 사전 계산 결과를 `session_state[f"_detail_precomputed_{ads_idx}"]`에 캐싱해 재진입 시 재사용한다.

---

## 6. 지표 체계

### 6.1 도메인 규칙

> **모든 비율 지표는 합계 기반으로 계산한다.** 행별 비율의 산술평균은 사용하지 않는다.
> **분모가 0일 때는 `None`(null)을 반환하고, UI에서는 `-`로 표시한다.**

### 6.2 핵심 KPI 정의

| 지표                            | 수식                                             | 함수                        | UI 포맷     |
| ------------------------------- | ------------------------------------------------ | --------------------------- | ----------- |
| **CVR** (전환율)          | `sum(turn) / sum(clk) × 100`                  | `calc_cvr()`              | `{n}%`    |
| **CPA** (전환단가)        | `sum(acost) / sum(turn)`                       | `calc_cpa()`              | `{n:,}원` |
| **마진율**                | `(sum(acost) - sum(earn)) / sum(acost) × 100` | `calc_margin_rate()`      | `{n}%`    |
| **총 클릭수 / 총 완료수** | `sum(clk)` / `sum(turn)`                     | inline                      | K/M 축약    |
| **평일 vs 주말 CVR 격차** | `cvr_weekday - cvr_weekend`                    | inline                      | `{n}%p`   |
| **TOP 10 의존도**         | `sum(전환수 상위10) / sum(전체 전환) × 100`   | `calc_dependency_ratio()` | `{n}%`    |

> **v2.3 대비 제거**: `settle_efficiency`(= `sum(media_price) / sum(adv_price)`, 광고비 대비 정산수익률) 지표는 `calc_all_kpis()`, `calc_group_kpis()`, `calc_daily_kpis()`에서 모두 제거되었다. `fmt_ratio()` 포맷 함수도 함께 제거됨.

### 6.3 집계 레벨

| 함수                               | 집계 레벨 | 반환 형식                                                       |
| ---------------------------------- | --------- | --------------------------------------------------------------- |
| `calc_all_kpis(df)`              | 전체 단일 | dict:`{clk, turn, acost, scost, earn, cvr, cpa, margin_rate}` |
| `calc_group_kpis(df, group_col)` | 그룹별    | DataFrame (그룹 컬럼 + KPI 컬럼)                                |
| `calc_daily_kpis(df)`            | 일자별    | DataFrame (`rpt_time_date` + KPI)                             |

### 6.4 전기 비교 (Period-over-Period)

`calc_change_rate(current, previous)` → 단순 차이 (`current - previous`).

| 현재 기간         | 비교 기간       |
| ----------------- | --------------- |
| 최근 1일          | 전일 1일        |
| 최근 7일          | 직전 7일        |
| 사용자 지정 (N일) | 시작일 직전 N일 |
| 전체              | 비교 없음       |

---

# Part III — ML 모델

## 7. M1: 광고 품질 등급

### 7.1 목적

광고 등록 시점의 속성(보상가, 유형, 매체, 일캡 등) **만으로** 품질 등급을 예측한다.

### 7.2 모델 사양 (아티팩트 직접 확인 기준)

| 항목        | 내용                                                           |
| ----------- | -------------------------------------------------------------- |
| 알고리즘    | `XGBClassifier`                                              |
| 피처 수     | **48개** (`models/model1_artifacts/feature_list.json`) |
| 적용 대상   | `is_valid_click10 == 1` 광고                                 |
| Test AUC    | **0.8346**                                               |
| Test PR-AUC | 0.6278                                                         |
| Val AUC     | 0.8180                                                         |
| 학습 일시   | 2026-05-10                                                     |

> v2.3 PRD는 "50피처 · AUC ~0.85"로 기재했으나, `models/model1_artifacts/metadata.json` 실측값은 **48피처, Test AUC 0.8346**이다. 본 문서는 아티팩트 실측값을 기준으로 한다.

### 7.3 스코어링 파이프라인

```
raw data
  → OHE 적용 (pipeline['ohe'])
  → Label Encoding (art['le_dict'])
  → 수치형 + OHE 결합
  → clean_col_names_unique() → align_features()
  → model.predict_proba(X)[:, 1]    ← raw probability
  → searchsorted(ref_proba_sorted, proba) / len(ref) × 100    ← 0~100 점수
  → pd.cut(score, bins, labels)     ← S/A/B/C/D 등급
```

### 7.4 등급 체계 (`grade_info.json` 실측, 5-grade quantile cut)

| 등급        | 라벨     | 점수 범위    | 색상        | 운영 의미                         |
| ----------- | -------- | ------------ | ----------- | --------------------------------- |
| **S** | 최우수   | > 80.7       | `#1B5E20` | 적극 집행, 매체 확장 검토         |
| **A** | 우수     | 59.9 ~ 80.7  | `#2E7D32` | 안정 운영, S 진입 가능성 모니터링 |
| **B** | 보통     | 40.0 ~ 59.9  | `#F9A825` | 관찰, 소재/타겟 최적화 여지       |
| **C** | 주의     | 20.02 ~ 40.0 | `#E65100` | 경고, 개선 미시행 시 축소 검토    |
| **D** | 개선필요 | < 20.02      | `#C62828` | 즉시 축소/중단 검토               |

### 7.5 운영 상태 연동

| M1 등급 | 운영 상태 |
| ------- | --------- |
| S, A, B | 정상      |
| C       | 주의      |
| D       | 긴급      |

---

## 8. M2: 조기부진 예측

### 8.1 목적

D+0 ~ D+3 초기 실적을 기반으로, 향후 부진할 위험을 예측한다.

### 8.2 모델 사양 (아티팩트 직접 확인 기준)

| 항목                         | 내용                                   |
| ---------------------------- | -------------------------------------- |
| 알고리즘                     | `LightGBM` (LGBM)                    |
| 피처 수                      | **101개**                        |
| 적용 대상                    | `early_click >= min_early_click(10)` |
| Test AUC                     | 0.8548                                 |
| Test PR-AUC                  | **0.6009**                       |
| Test Precision / Recall / F1 | 0.4351 / 0.6706 / 0.5278               |
| 학습 일시                    | 2026-05-08                             |

> v2.3 PRD는 "PR-AUC ~0.72"로 기재했으나, `models/model2_artifacts/metadata.json` 실측값은 **Test PR-AUC 0.6009**다.

### 8.3 피처 구성

M1 공통 피처 + TF-IDF(`ads_name` + `ads_summary` 텍스트) + `early_click`(D+3 이내 클릭 합계). `feature_set: "C_tfidf+click"`.

### 8.4 의사결정 로직

```
early_click < 10 (min_early_click)
  → "rule_based_review"  (데이터 불충분, ML 미적용 — 별도 관리)

early_click >= 10:
  m2_proba >= threshold → "decline_risk"   (부진 위험)
  m2_proba <  threshold → "normal"         (정상)
```

### 8.5 Threshold (`threshold_info.json` 실측)

| 키                 | 설명                                           | 값               |
| ------------------ | ---------------------------------------------- | ---------------- |
| `best_f1`        | F1 점수 최적화 threshold (**기본 사용**) | **0.5905** |
| `default_05`     | 기본 0.5                                       | 0.50             |
| `Recall >= 0.70` | 재현율 70% 타겟                                | 0.5064           |
| `Recall >= 0.80` | 재현율 80% 타겟                                | 0.3856           |

> v2.3 PRD는 best_f1을 "~0.54"로 기재했으나 실측값은 **0.5905**다. note: "val 기준 탐색값. test에 적용 시 recall이 달라질 수 있음(보장 X)".

**Threshold 재적용**: `reapply_threshold(scores_df, threshold_value)` — 캐시된 `m2_proba`에 새 threshold를 적용하여 `m2_decision`을 재계산한다. 모델 재추론 불필요.

---

## 9. 배지 시스템

### 9.1 개요

ML 모델 결과를 운영자가 즉시 이해하고 행동할 수 있도록, **배지(Badge)** 기반 액션 제시 체계를 운영한다.

### 9.2 기회 배지 (M1 S/A 등급 광고)

`_assign_opportunity_badge()` — 우선순위 순서:

| 우선순위 | 배지               | 조건                        | 아이콘 | 색상        | 운영 액션                                   |
| -------- | ------------------ | --------------------------- | ------ | ----------- | ------------------------------------------- |
| 1        | **승급추진** | A등급 AND`m1_score >= 75` | ◐     | `#2E7D32` | S등급 진입 직전 — 예산 사전 확보, 모니터링 |
| 2        | **매체확장** | S등급 또는 나머지 A등급     | ↗     | `#1565C0` | 다른 매체로 확장 가능성 검토                |

> **v2.3 → v2.4 변경**: 배지명 "등급경계" → **"승급추진"** 으로 변경. 조건(A등급 + 점수 ≥75)과 우선순위 로직 자체는 동일하며, 색상만 보라(`#6A1B9A`)에서 초록(`#2E7D32`)으로 변경되어 "기회" 계열 색상 톤으로 통일됨.

### 9.3 위험 배지 (M2 decline_risk 광고)

`_assign_risk_badge()`:

| 배지               | 조건                 | 아이콘 | 색상        | 운영 액션                            |
| ------------------ | -------------------- | ------ | ----------- | ------------------------------------ |
| **즉시조치** | `m2_proba >= 0.70` | ⚠     | `#C62828` | 소재 교체, 랜딩 점검, 일시 중단 검토 |
| **우선검토** | `m2_proba < 0.70`  | 👁     | `#E65100` | 3일 재평가, 핵심 지표 모니터링       |

### 9.4 배지 → P5 진입 흐름

```python
st.session_state["detail_ads_idx"]  = ads_idx       # 광고 ID
st.session_state["detail_badge"]    = badge          # 매체확장/승급추진/즉시조치/우선검토
st.session_state["detail_model"]    = "m1" or "m2"   # 어느 모델에서 진입
st.session_state["detail_source"]   = source_page    # 돌아가기 대상 페이지
→ st.switch_page("pages/05_광고_상세.py")
```

P2의 "이동 조치 필요 광고" 대시보드 역시 동일한 session_state 키를 사용해 P5로 진입한다 (`detail_model`은 모드가 "urgent"/"early"면 `"m2"`, "expand"/"promote"면 `"m1"`).

---

# Part IV — 사용자 경험

## 10. 전체 네비게이션 흐름

### 10.1 5페이지 구조

```
app.py → st.navigation([
    P1: "전체 광고 현황 오버뷰"  (default=True)
    P2: "유형별 운영 탐색"
    P3: "매체별 운영 탐색"
    P4: "카테고리별 운영 탐색"
    P5: "광고 상세"
])
```

| 페이지 | 파일                            | 필터                                | 진입 경로                                        |
| ------ | ------------------------------- | ----------------------------------- | ------------------------------------------------ |
| P1     | `01_전체_광고_현황_오버뷰.py` | 없음 (전체 데이터) + 보기 모드 토글 | 기본 랜딩                                        |
| P2     | `02_유형별_운영_탐색.py`      | 기간 + 유형 + 시간대                | 필터의 관측 기준 선택, ML/우선순위 리스트 "상세" |
| P3     | `03_매체별_운영_탐색.py`      | 기간 + 매체 + 시간대 + 유형(보조)   | 동일                                             |
| P4     | `04_카테고리별_운영_탐색.py`  | 기간 + 카테고리 + 시간대            | 동일                                             |
| P5     | `05_광고_상세.py`             | 없음 (단일 광고)                    | ML 인사이트/운영 우선순위 "상세" 버튼            |

### 10.2 페이지 간 전환

- P1 ↔ P2/P3/P4: 필터 관측 기준 라디오, 차원별 Top3 카드 버튼 (`st.switch_page()`)
- P1~P4 → P5: 배지/우선순위 리스트의 "상세" 버튼 → session_state 설정 + `st.switch_page()`
- P5 → 원래 페이지: 뒤로가기 버튼(`st.switch_page(detail_source)`), 복귀 시 ML 탭을 m1→grade, m2→risk로 자동 설정

### 10.3 세션 상태 관리 (주요 키)

| 키 패턴                                               | 설명                                                                |
| ----------------------------------------------------- | ------------------------------------------------------------------- |
| `{page_key}_*`                                      | 페이지별 독립 필터 상태 (p1, p2, p3, p4)                            |
| `detail_*`                                          | P5 진입 컨텍스트 (ads_idx, badge, model, source)                    |
| `_detail_precomputed_{ads_idx}`                     | P5 광고별 사전 계산 캐시 (재진입 시 재사용, 다른 광고 진입 시 정리) |
| `p2_opd_mode`, `p2_opd_show_n`                    | P2 운영 우선순위 대시보드 모드/표시 건수                            |
| `p3_hide_adlist`, `p3_last_click_name` / `p4_*` | P3/P4 버블 클릭 패널 표시 상태                                      |
| `p1_view_mode`                                      | P1 보기 모드 ("전체 기간" / "당일 vs 평시")                         |
| `{page_key}_chat_history`                           | 페이지별 AI Agent 대화 이력                                         |

### 10.4 `@st.fragment` 사용 여부 (v2.4 변경)

v2.3까지 P2~P4의 버블차트/CVR·마진율 섹션은 `@st.fragment`로 부분 리렌더링을 최적화했다. **v2.4에서는 `@st.fragment`를 더 이상 사용하지 않는다** — 대신 [§5.6](#56-캐싱-전략--v24-변경점)의 프로세스 전역 캐시와 P5의 광고별 session_state 사전 계산 패턴으로 리렌더링 비용을 관리한다.

---

## 11. P1: 전체 광고 현황 오버뷰

**목적**: 전체 기간의 핵심 패턴(전체 기간 모드)과 당일의 이상 감지(당일 vs 평시 모드)를 하나의 보기 모드 토글로 전환하며 파악한다.

**파일**: `pages/01_전체_광고_현황_오버뷰.py`

> **v2.3 대비 가장 크게 바뀐 페이지.** 기존의 상태 배너/의존도 배너/시간대 추이/Top3/2-tab ML 인사이트가 모두 들어있던 단일 레이아웃에서, **"전체 기간" / "당일 vs 평시" 두 보기 모드**로 명확히 분리되었다.

### 11.1 헤더

로고 + 제목 "전체 광고 현황 오버뷰" + 분석 범위 배너(`{date_min} ~ {date_max}` + 총 레코드 수).

### 11.2 보기 모드 토글

`st.segmented_control(["전체 기간", "당일 vs 평시"])`, 기본값 "전체 기간". 세션 키 `p1_view_mode`.

### 11.3 KPI 카드

| 보기 모드    | 카드 수 | 구성                                                                                   |
| ------------ | ------- | -------------------------------------------------------------------------------------- |
| 전체 기간    | 4       | 총 클릭수, 평균 CVR, 평균 마진율, 평일 vs 주말 CVR 격차(미니 바 포함)                  |
| 당일 vs 평시 | 3       | 총 클릭수, 평균 CVR, 평균 마진율 (스파크라인의 첫/끝 bin 색상을 평시/당일 강조로 변경) |

### 11.4 전체 기간 모드 전용 섹션

1. **모델 결과 기반 운영 추천 배너** — `render_ml_recommendation_banner()`. 품질등급(기회: 매체확장/승급추진) + 조기부진(위험: 즉시조치/우선검토) 4개 배지의 카운트를 한 줄에 표시하고, 배지별 대표 광고 1건씩(최대 4건)을 통합 리스트로 렌더링. *(v2.3의 2-tab "ML 인사이트" 구조를 단일 배너로 통합)*
2. **시간대 × 요일 CVR 히트맵 + Top 10 테이블** — 좌측 히트맵(요일/시간대 선택 가능), 우측에 선택 조건 또는 전체 기간 CVR Top 10 테이블(클릭 ≥30, `render_ad_list_table_with_scores()`).

### 11.5 당일 vs 평시 모드 전용 섹션

1. **시간대별 CVR — 당일 vs 평시** — `make_hourly_cvr_trend(base, today)`. 평소 평균(점선) vs 오늘(실선+마커). 격차 최대 시점 자동 강조. Plotly `on_select="rerun"`으로 임의 시간대 클릭 시 우측에 해당 시간대 이상 광고 리스트(CVR 오름차순, 클릭 ≥30) 표시.
2. **차원별 Top3 — 당일 vs 평시** — 유형별/매체별/카테고리별 각각 오늘 CVR Top3을 평시(전체 기간) 대비로 산출. 배지 판정: TOP3 평균 CVR이 전체 평균보다 **10% 초과 높으면 정상**, 그 이하지만 평균 이상이면 **주의**, 평균 이하면 **위험**. 각 카드 하단 버튼으로 P2/P3/P4 진입.

> **v2.3 대비 제거된 기능**: 운영 상태 배너(`render_status_banner`, 정상/주의/긴급 비율), TOP10 의존도 배너(`render_dependency_banner`)는 P1 페이지에서 더 이상 렌더링되지 않는다. 해당 함수(`render_status_banner`, `render_dependency_banner`, `classify_operation_status`, `calc_dependency_ratio`)는 `src/components.py`/`src/rules.py`에 코드는 남아 있으나 페이지에서 호출되지 않는다.

### 11.6 AI Agent 채팅

`render_chat_section(page_key="p1", filters_desc="전체 기간, 필터 없음")` — [§17 참조](#17-ai-agent--rag).

---

## 12. P2: 유형별 운영 탐색

**목적**: `analysis_ads_type_label` 기준으로 광고 유형별 성과를 비교하고, **"이동 조치 필요 광고" 운영 우선순위 대시보드**로 어떤 유형의 어떤 광고부터 봐야 하는지 정렬해서 보여준다.

**파일**: `pages/02_유형별_운영_탐색.py`

### 12.1 필터

`render_filters(page_key="p2", group_col="analysis_ads_type_label", group_label="유형", show_sub_filter=False)` — 관측 기준(유형/매체/카테고리 전환) + 관측 시점(최근1일/7일/전체/사용자지정) + 유형 pills + 시간대 슬라이더.

### 12.2 KPI 카드 (3종)

평균 CVR, 평균 CPA(`invert_color=True`), 평균 마진율 — 전기비교 + 30bin 일별 스파크라인.

### 12.3 이동 조치 필요 광고 (운영 우선순위 대시보드) — **v2.4 신규**

P2의 핵심 신규 섹션. 광고를 유형별로 그룹화하고, 4가지 운영 모드 중 선택한 모드에 해당하는 광고만 골라 "지금 봐야 할 순서"로 제시한다.

**모드 버튼 4종** (각 버튼에 건수 표시, 클릭 시 `p2_opd_mode` 갱신):

| 모드 키     | 라벨     | 분류 조건                                       | 모델 | 색상        |
| ----------- | -------- | ----------------------------------------------- | ---- | ----------- |
| `urgent`  | 즉시조치 | `m1_grade == "D"`                             | M1   | `#C62828` |
| `early`   | 우선검토 | `m2_decision == "decline_risk"`               | M2   | `#E65100` |
| `expand`  | 매체확장 | `m1_grade ∈ {S, A}` AND 승급추진 조건 미충족 | M1   | `#1565C0` |
| `promote` | 승급추진 | `m1_grade == "A"` AND `m1_score >= 75`      | M1   | `#2E7D32` |

**우측 패널 — 유형별 비중 차트**: 선택된 모드에 해당하는 광고가 유형별로 전체 대비 몇 %를 차지하는지 가로 바 차트로 표시(비중 내림차순).

**좌측 패널 — 유형별 광고 리스트** (기본 5건, "+ 다음 5건 더보기"로 확장):

- 유형 순서: `["클릭형", "감상형", "수행형", "참여형", "구매형", "설치형", "노출형", "기타"]` 우선, 나머지는 등장 순.
- 유형별 **기준 지표**가 다름: 매체확장/승급추진 모드는 항상 **CVR** 기준. 그 외 모드는 유형별 1차 지표 매핑(`_OPD_PRIMARY_METRIC`) — 예: 설치형/클릭형/수행형/감상형/기타 → CPA, 참여형 → 완료수, 구매형/노출형 → 마진율.
- 각 행에 동료(같은 유형 내) 비교 기반 **상태 아이콘**(↓ 빨강 하위33% / − 주황 중간 / ↑ 초록 상위33%, 지표 방향성 반영) + 배지 pill + "상세" 버튼(P5 진입).

### 12.4 ML 인사이트 (기회발굴 + 손실방어)

`get_cached_ml_insight_data(opp_top_n=1, risk_top_n=1)` → `render_ml_insight()` — P1과 달리 P2~P4는 여전히 **2-tab(품질등급/조기부진) 구조**를 유지한다. 배지별 대표 1건씩만 노출.

### 12.5 AI Agent 채팅

필터링된 데이터 범위로 스코핑.

> **v2.3 대비 제거된 기능**: 유형별 CVR·마진율 비교 그룹 바 차트(`make_cvr_margin_bar`, 알람 이모지 ★/↺/⚠), 유형별 종합 평가 버블차트(사분면 4종 알림 카드), 유형별 CVR 추이선(`make_cvr_trend`)은 P2 페이지에서 제거되었다. 해당 차트 함수는 `src/charts.py`에 남아 있으나 P2에서는 호출되지 않으며, 그 자리를 "이동 조치 필요 광고" 대시보드가 대체한다.

---

## 13. P3: 매체별 운영 탐색

**목적**: `final_media` 기준으로 매체별 성과를 비교하고, 사분면 분류 기반 매체 인사이트 카드로 효율 차이를 진단한다.

**파일**: `pages/03_매체별_운영_탐색.py`

### 13.1 필터 (보조 필터 포함)

`render_filters(page_key="p3", group_col="final_media", show_sub_filter=True, sub_filter_items=유형 목록)` — 매체 × 유형 교차 분석 가능 (P3만 보조 필터 보유).

### 13.2 KPI 카드 (3종)

평균 CVR, 평균 CPA(`invert_color=True`), 총 완료수.

### 13.3 매체별 종합 평가 버블 차트

`make_bubble_chart(group_kpis, x_metric="cpa", y_metric="margin_rate", size_metric="turn", show_breakeven=False, show_acost=False)` — X: CPA(낮을수록 좋음), Y: 마진율, 크기: 완료수. 버블 클릭(`on_select="rerun"`) → 우측에 해당 매체 Top 10 광고(CPA 오름차순, 클릭 ≥30). 겹치는 매체는 radio로 선택, "✕"로 패널 닫기.

> **v2.3 대비**: 사분면 기준축이 `y_metric="settle_efficiency"` → **`y_metric="margin_rate"`** 로 통일됨(`classify_quadrant()` 기본값 변경).

### 13.4 매체 인사이트 카드 4종

`classify_quadrant(group_kpis, x_metric="cpa", y_metric="margin_rate")` + `build_media_insight_cards()` (`src/rules.py`):

| 카드                | 선정 기준                         | 표시 모드                                                                                                                      |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **최우수**    | best 사분면 중 마진율 최고 매체   | `mode="trend"`(일별 마진율 추이 + 안정성 배지) 또는 `mode="spend"`(추이 데이터 부족 시 광고비 순위 + 인사이트 문구로 대체) |
| **비효율**    | loss 사분면 중 마진율 최저 매체   | 동일 (trend: CPA 급등 시점 추적 / spend: 광고비 비중 기반 인사이트)                                                            |
| **단가 협업** | low_eff 사분면 매체 목록          | 마진율 평균비 테이블                                                                                                           |
| **소재 개선** | expensive 사분면 중 CPA 최고 매체 | 마진율/CVR 비교 + 인사이트 텍스트                                                                                              |

**trend → spend 폴백 로직**: 선택 기간(filtered)의 일별 데이터가 `min_trend_days`(기본 2일) 미만이면(예: "최근 1일" 선택, 또는 전환이 특정 하루에 몰린 매체) 더 긴 시리즈(`trend_df`)로 대체 시도하고, 그래도 부족하면 추이 차트 대신 **전체 매체 광고비 순위 + 비중 기반 인사이트 문구**(`mode="spend"`)로 전환한다.

### 13.5 매체별 볼륨·효율·수익성 요약 테이블

`render_media_summary_table(group_kpis, group_col="final_media")` — expander 내 정렬 가능(완료수순/마진율순/CPA순) 테이블.

### 13.6 ML 인사이트 + AI Agent 채팅

P2와 동일 구조(2-tab, opp/risk top_n=1).

> **v2.3 대비 제거**: P3의 유형별과 동일했던 CVR·마진율 그룹 바 차트, CVR 추이선은 제거되고 "매체별 볼륨·효율·수익성 요약 테이블"로 대체됨.

---

## 14. P4: 카테고리별 운영 탐색

**목적**: `category_name` 기준으로 카테고리별 성과를 비교하고, Pareto 분석으로 핵심 카테고리를 식별한다.

**파일**: `pages/04_카테고리별_운영_탐색.py`

### 14.1 필터

`render_filters(page_key="p4", group_col="category_name", show_sub_filter=False)`.

### 14.2 KPI 카드 (3종)

평균 CVR, 총 완료수, 평균 마진율.

### 14.3 카테고리별 종합 평가 버블 차트

X축은 사용자가 라디오로 선택(`CVR` 또는 `CPA`), Y축은 마진율(고정), 크기는 클릭수. 버블 클릭 시 우측에 선택 카테고리의 Top 10 광고(X축이 CPA면 오름차순, CVR이면 내림차순, 클릭 ≥30).

### 14.4 운영 알림 4종 카드

`classify_quadrant(group_kpis, x_metric=x_metric, y_metric="margin_rate")` 기반 — 최우수(★, 초록)/소재 개선(↺, 파랑)/단가 점검(⚠, 노랑)/비효율(⊘, 빨강), `render_alert_card()`로 렌더링.

### 14.5 카테고리별 완료수 비중(누적) — Pareto 차트

`make_pareto_chart(group_kpis, group_col="category_name")` — 완료수 막대 + 누적 비율 꺾은선 + 80% 기준선. 인사이트 메시지: "상위 {N}개({카테고리명})가 전체 완료의 80% 이상을 차지합니다."

### 14.6 카테고리별 볼륨·효율·수익성 요약 테이블

`render_media_summary_table(group_kpis, group_col="category_name")` — P3과 동일 함수 재사용(범용화됨).

### 14.7 ML 인사이트 + AI Agent 채팅

P2/P3와 동일 구조.

> **v2.3 대비 제거**: 카테고리별 CVR 추이선(`make_cvr_trend`, CATEGORY_COLORS)은 제거됨. Pareto 차트의 표시 캡션이 "카테고리별 완료수 비중(누적)"으로 명확화됨.

---

## 15. P5: 광고 상세

**목적**: ML 인사이트/운영 우선순위 대시보드에서 선택한 개별 광고를 심층 분석하고, 배지에 따른 구체적 운영 액션을 제시한다.

**파일**: `pages/05_광고_상세.py`

### 15.1 진입 방식 및 네비게이션

P1~P4의 ML 인사이트 또는 P2 운영 우선순위 대시보드 "상세" 버튼 → session_state 컨텍스트(`detail_ads_idx`, `detail_source`, `detail_badge`, `detail_model`) 설정 후 `st.switch_page()`. 뒤로가기 시 원래 페이지의 ML 탭을 m1→grade, m2→risk로 자동 전환.

### 15.2 광고 헤더 + 히어로 카드

광고 ID + 운영 일수 + 배지 pill + 광고명 + 태그(유형·매체·카테고리·보상가·재진입 타입) + 배지별 헤드라인/설명/CTA (`build_hero_content()`).

### 15.3 모델 카드

- **M1 카드** (매체확장/승급추진 진입 시): 등급 배지 + 점수 + "승급추진"인 경우 S등급 cutpoint까지의 점수 차이 + A등급 전체 건수 대비 위치.
- **M2 카드** (즉시조치/우선검토 진입 시): 부진확률 서클(0.00~1.00) + 판정(decline_risk/normal/rule_based_review) + best_f1 threshold 대비 위치.

### 15.4 KPI Grid

M1 진입: 클릭수/완료수/CVR/CPA/마진율 5열. M2 진입: early_click/완료수/CVR/CPA/광고비 5열. 각 KPI에 같은 유형 그룹 평균 대비 컨텍스트 표시(`get_kpi_contexts()` — v2.4에서 정산수익률 컨텍스트는 제거되어 **5개 KPI** 컨텍스트만 반환).

### 15.5 배지별 액션 섹션

| 배지                                 | Primary 액션                                                                                           | Secondary 액션                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------- |
| **매체확장**                   | 매체별 CVR 비교 바 + Top3 확장 추천 매체 카드 + "결재 요청"                                            | "3일 후 재평가 알림 설정"         |
| **승급추진** *(구 등급경계)* | M1 점수 게이지(SVG, 5등급 구간 + 현재 위치 마커) + "S 진입 시 알림 설정"                               | "S 진입 시 알림 설정"             |
| **즉시조치**                   | 클릭-완료 분석 + "랜딩 점검"/"소재 교체" (빨간 배경)                                                   | "광고 일시 중단 검토" (빨간 배경) |
| **우선검토**                   | 부진확률 게이지(HTML, safe/warning/danger 3구간) + 추적 지표 테이블(부진확률·CVR·click_day1→3 비율) | "소재 사전 점검", "모니터링 등록" |

액션 모달(`@st.dialog`)은 UI 시연용이며 실제 API 연동은 없다.

### 15.6 "왜?" 섹션 (Expandable)

`build_why_steps()` — 배지 할당 근거를 단계별로 설명. 승급추진의 경우 "CVR·**마진율** 모두 같은 유형 평균 이상"으로 텍스트가 갱신됨(구 "정산수익률" 표현 제거).

---

# Part V — 공통 시스템

## 16. 디자인 시스템

### 16.1 색상 토큰 (변경분만 표기, 나머지는 [부록 19](#19-설정-상수-레퍼런스) 참조)

| 역할                              | v2.3               | v2.4               |
| --------------------------------- | ------------------ | ------------------ |
| 기회 배지 "승급추진"(구 등급경계) | `#6A1B9A` (보라) | `#2E7D32` (초록) |
| Hero tone_soft (승급추진)         | `#EDE0F2`        | `#E8F5E9`        |
| Hero tone_deep (승급추진)         | `#4A1170`        | `#1B5E20`        |

### 16.2 KPI 카드 / 알림 카드 / 광고 리스트 테이블 / 필터 시스템

세부 레이아웃·동작은 v2.3과 동일하다 (`render_kpi_card()`, `render_alert_card()`, `render_ad_list_table_with_scores()`, `render_filters()`). 자세한 명세는 PRD_v2_3.md §16.3~16.6을 참조.

### 16.3 신규 컴포넌트 (v2.4)

| 함수                                                        | 용도                                                                          |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `render_ml_recommendation_banner()`                       | P1 "모델 결과 기반 운영 추천" 통합 배너 (배지 4종 + 대표 광고)                |
| `render_media_highlight_card()`                           | 매체 인사이트 최우수/비효율 카드 (trend/spend 모드)                           |
| `render_media_price_coop_card()`                          | 단가 협업 카드 (마진율 평균비 테이블)                                         |
| `render_media_material_card()`                            | 소재 개선 카드 (마진율/CVR 비교)                                              |
| `render_media_summary_table()`                            | 그룹별 볼륨·효율·수익성 요약 테이블 (P3/P4 공용)                            |
| `render_grade_distribution()` / `render_risk_summary()` | 등급/위험 분포 시각화 (현재 P1에서는 통합 배너로 대체되어 직접 호출되지 않음) |

### 16.4 빌드되었으나 미연동된 기능 (참고)

`src/config.py`의 `TYPE_CATEGORY_MAP`/`TYPE_CATEGORY_GROUPS`/`FLOW_LOWER_IS_BETTER`/`FLOW_ACTION_COLORS`/`FLOW_ACTION_LABELS`, `src/rules.py`의 `classify_flow_items()`, `src/charts.py`의 `make_flow_bubble_chart()`는 "유형별 운영 흐름"(설치형·가입형 / 클릭형·실행형 / 참여형·기타 / 구매형·노출형 4대분류 사분면 분류, urgent/stable/expand/hold 액션 라벨) 기능을 위해 구현되어 있으나, **현재 어떤 `pages/*.py`에서도 호출되지 않는다.** 향후 P2 "이동 조치 필요 광고" 대시보드의 대안/확장판으로 연동될 가능성이 있는 미사용 인프라로 간주한다.

---

## 17. AI Agent + RAG

### 17.1 모델 사양

| 항목            | 값                                                               |
| --------------- | ---------------------------------------------------------------- |
| 응답 생성 모델  | Gemini 2.5 Flash (`gemini-2.5-flash`)                          |
| RAG 임베딩 모델 | `gemini-embedding-001`                                         |
| API             | Google Generative AI (`google-genai`)                          |
| Temperature     | 0.3                                                              |
| 인증            | **`st.secrets["GEMINI_API_KEY"]`** (v2.3까지는 `.env`) |
| 클라이언트 캐싱 | `@st.cache_resource`                                           |

### 17.2 시스템 프롬프트 — v2.4 확장분

기존 도메인 규칙(합계 기반 비율, 분모 0 처리, 근거 없는 추정 금지, 현재 필터 기준 집계만 사용)에 다음이 추가됨:

- **근거 인용 범위 제한**: 데이터 컨텍스트에는 ML 등급/위험 분포가 항상 포함되지만, 질문에 "등급", "S/A/B/C/D", "부진", "decline_risk", "위험 광고" 등의 단어가 명시적으로 없으면 evidence/interpretation/actions 어디에도 등급·위험 분포 숫자를 언급하지 않는다. 서로 무관한 데이터(예: 외부 뉴스 ↔ ML 분포)를 인과관계로 추측해 엮지 않는다.
- **지식베이스 활용**: "## 참고 지식베이스" 섹션이 주어지면 PRD/운영 룰/ML 정의 등 정책성 질문에 근거로 사용하고, 사용한 문서·섹션을 `sources`에 명시한다.
- **외부 뉴스 검색(Google Search 도구)**: "오늘/이번주 왜 떨어졌어/올랐어" 같은 원인 해석 질문, 매체·브랜드·업계 최신 이슈/정책 변화 질문, "오늘자 보고서" 같은 일일 해석 요청일 때만 사용. 단순 집계·순위 질문에는 사용하지 않는다.

### 17.3 데이터 컨텍스트 (`build_data_context()`)

```
## 페이지: {page_name}
## 필터: {filters_desc}

## 전체 KPI
- 총 클릭수 / 총 완료수 / CVR(%) / CPA(원) / 마진율(%)

## ML 품질등급 분포
- S/A/B/C/D 건수

## 조기부진 예측 분포
- decline_risk / normal / rule_based_review 건수

## 광고 요약 (총 {total}건 중 상위 50건, CVR 내림차순)
{CSV: ads_idx, ads_name, type, media, category, clk, turn, cvr, cpa, margin_rate, m1_grade, m2_decision}
```

> v2.3 대비 `settle_efficiency` 관련 항목은 컨텍스트에서 제거됨(코드 전체에서 지표 삭제와 일치).

### 17.4 JSON 응답 스키마 — `sources` 필드 추가

```json
{
  "summary": "핵심 요약 한 문장",
  "evidence": [
    {"metric": "지표명", "value": "값", "scope": "범위/대상"}
  ],
  "interpretation": "왜 이런 결과가 나왔는지 1~2문장 해석",
  "actions": ["추천 액션 1", "추천 액션 2"],
  "sources": [
    {"doc": "문서명", "section": "섹션명"}
  ]
}
```

- 마크다운 코드펜스 없는 순수 JSON. Google Search 도구를 함께 사용할 경우 강제 JSON 모드를 쓸 수 없어, 응답 텍스트에서 코드펜스를 정규식으로 제거한 뒤 파싱한다(`_parse_json_response()`).
- 검색을 사용했을 경우, 모델이 자체 보고한 출처 대신 **Google Search 그라운딩 메타데이터(`grounding_chunks`)에서 직접 추출한 신뢰 가능한 뉴스 출처**를 `sources`에 병합한다(`_extract_news_sources()`) — 환각 방지를 위해 모델 자가 보고 제목은 신뢰하지 않는다.

### 17.5 RAG 지식베이스 (`src/rag.py`, `scripts/build_rag_index.py`) — **v2.4 신규**

| 구성      | 내용                                                                                                             |
| --------- | ---------------------------------------------------------------------------------------------------------------- |
| 원문      | `data/rag_docs/*.md` (PRD/운영 룰/ML 정의 등)                                                                  |
| 인덱스    | `models/rag_artifacts/index.parquet` (사전 계산, 수동 빌드)                                                    |
| 청크 단위 | `## ` 헤딩 단위로 분할, 각 청크에 `# 문서 제목`을 prefix로 포함                                              |
| 임베딩    | `gemini-embedding-001`, 문서 청크는 `task_type="RETRIEVAL_DOCUMENT"`, 질의는 `task_type="RETRIEVAL_QUERY"` |
| 검색 방식 | 코사인 유사도(L2 정규화 후 내적),`top_k=4`, 유사도 임계값 **0.65** 미만은 제외                           |
| 빌드 방법 | `python scripts/build_rag_index.py` — 문서가 바뀔 때마다 수동 재실행 (배포 파이프라인 미포함)                 |
| 장애 허용 | 인덱스 파일이 없거나 임베딩 호출이 실패하면 빈 리스트 반환 — AI Agent의 기존 데이터 기반 응답 흐름에 영향 없음  |

런타임에는 사용자 질문당 임베딩 API를 **1회만** 호출하고, 인덱스는 `@st.cache_resource`로 캐싱된 결과를 재사용한다.

### 17.6 대화 이력 / UI 렌더링

v2.3과 동일: `st.session_state["{page_key}_chat_history"]`에 저장, 최근 10턴만 API 전달, user/model 역할 교대. UI는 진한 남색 배경 채팅 패널 + Summary/Evidence/Interpretation/Actions 블록 구조에 더해, 사용된 경우 출처(`sources`) 표시가 추가된다.

---

## 18. 운영 규칙 엔진

### 18.1 운영 상태 분류 (`classify_operation_status()`)

M1 등급 기반(긴급=D, 주의=C, 정상=S/A/B). 함수는 유지되어 있으나, **P1 페이지에서는 더 이상 직접 호출되지 않는다** ([§11.5](#115-당일-vs-평시-모드-전용-섹션) 참조).

### 18.2 사분면 분류 (`classify_quadrant()`)

```python
def classify_quadrant(group_kpis, x_metric="cpa", y_metric="margin_rate"):
    ...
    is_lower_better = (x_metric == "cpa")
```

- **v2.4 변경**: `y_metric` 기본값이 `"settle_efficiency"` → **`"margin_rate"`**로 변경. `is_lower_better` 판정도 `x_metric in ("cpa", "settle_efficiency")` → **`x_metric == "cpa"`** 로 단순화. Y축은 항상 "높을수록 좋음"으로 통일(과거 settle_efficiency는 낮을수록 좋음이라 별도 분기가 있었음).

### 18.3 룰 기반 운영 알림 (`generate_alerts()`)

| 조건                                    | 레벨   | 메시지                                    |
| --------------------------------------- | ------ | ----------------------------------------- |
| `margin_rate < 0`                     | red    | "주의: {name} — 마진율 {n}% (비용 초과)" |
| `cvr < avg_cvr × 0.7`                | yellow | "관찰: {name} — CVR 평균 대비 낮음"      |
| `cvr >= avg_cvr AND margin_rate > 10` | green  | "양호: {name} — 안정"                    |

> v2.3에서는 `settle_efficiency >= 1.0` 조건으로 비용 초과를 판정했으나, v2.4는 **`margin_rate < 0`** 으로 대체됨.

### 18.4 매체 인사이트 카드 빌더 (`build_media_insight_cards()`) — v2.4 신규

P3에서 사용. 사분면 분류 결과로 최우수/비효율/단가협업/소재개선 4종 카드 데이터를 생성한다. 보조 함수:

- `_stability_streak()`: 최근일부터 역순으로 일별 변화율이 5% 이내인 연속 일수 계산 → "N일 연속 안정" 배지 문구.
- `_decline_day()`: 최근 14일 중 지표가 가장 크게 악화된 날짜로부터 며칠 전인지 계산.
- `_spend_rank_insight()`: 추이 데이터가 부족할 때 전체 매체 중 광고비 순위/비중 기반 대체 인사이트 텍스트 생성.

### 18.5 유형별 운영 흐름 사분면 (`classify_flow_items()`) — 구현됨, 미연동

x/y 중앙값과 지출액(acost) 중앙값 기준 4분류(urgent/stable/expand/hold). [§16.4](#164-빌드되었으나-미연동된-기능-참고) 참조.

---

# Part VI — 부록

## 0. v2.3 → v2.4 변경 요약

| 구분                                              | v2.3 (2026-05-11)                                                         | v2.4 (2026-06-30)                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| AI Agent 인증                                     | `.env` (`GEMINI_API_KEY`)                                             | `st.secrets["GEMINI_API_KEY"]` (런타임), `.env`는 RAG 빌드 스크립트 전용      |
| AI Agent 외부 컨텍스트                            | 없음                                                                      | RAG 지식베이스 검색 + Google Search 그라운딩 +`sources` 필드                    |
| 배지명                                            | 등급경계 (보라`#6A1B9A`)                                                | **승급추진** (초록 `#2E7D32`) — 조건 동일                                |
| `settle_efficiency` 지표                        | 존재 (`price_agg` 파이프라인 기반)                                      | **완전 제거** — `margin_rate`로 통일                                     |
| `price_agg` / `ads_join_info_labeled.parquet` | 사용                                                                      | **제거**, `scripts/precompute_price_agg.py` 삭제 대상                     |
| 캐싱                                              | session_state 기반 (세션별 재계산)                                        | **프로세스 전역 dict 캐시** (세션 간 공유)                                  |
| `@st.fragment`                                  | P2~P4 버블/CVR 섹션에 사용                                                | **미사용** (전역 캐시 + session_state 사전계산으로 대체)                    |
| P1 레이아웃                                       | 단일 레이아웃(상태배너+의존도배너+히트맵+시간대추이+Top3+2tab ML인사이트) | **보기 모드 토글** (전체 기간 / 당일 vs 평시)로 분리, 상태/의존도 배너 제거 |
| P2 신규 기능                                      | 없음                                                                      | **"이동 조치 필요 광고" 운영 우선순위 대시보드** (4모드)                    |
| P2/P3/P4 제거 기능                                | CVR·마진율 그룹 바, 버블 사분면 알림 카드(P2), CVR 추이선                | 제거(P2),`render_media_summary_table()`로 대체(P3/P4 일부)                      |
| P3 신규 기능                                      | 없음                                                                      | 매체 인사이트 카드 4종 trend/spend 폴백 로직, 매체 요약 테이블                    |
| M1 모델 사양(문서 vs 실측)                        | 50피처, AUC ~0.85 (서술)                                                  | **48피처, Test AUC 0.8346**(아티팩트 실측)                                  |
| M2 모델 사양(문서 vs 실측)                        | 101피처, PR-AUC ~0.72 (서술)                                              | 101피처,**Test PR-AUC 0.6009**, best_f1 **0.5905** (아티팩트 실측)    |

---

## 19. 설정 상수 레퍼런스

### 19.1 경로 상수

| 상수                                                           | 값                                                                     |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `BASE_DIR` / `DATA_DIR` / `MODELS_DIR` / `OUTPUTS_DIR` | 프로젝트 루트 기준 표준 하위 경로                                      |
| `MODEL1_ARTIFACTS` / `MODEL2_ARTIFACTS`                    | `{MODELS_DIR}/model{1,2}_artifacts`                                  |
| `PIPELINE_PATH`                                              | `{MODELS_DIR}/preprocessing_pipeline.joblib`                         |
| `RAG_DOCS_DIR`                                               | `{DATA_DIR}/rag_docs` *(v2.4 신규)*                                |
| `RAG_ARTIFACTS` / `RAG_INDEX_PATH`                         | `{MODELS_DIR}/rag_artifacts` / `.../index.parquet` *(v2.4 신규)* |

### 19.2 분석 기간 / 필터 옵션

```python
ANALYSIS_DATE_START = "2025-07-26"
ANALYSIS_DATE_END   = "2025-08-25"
DEFAULT_PERIOD      = "전체"
PERIOD_OPTIONS      = ["최근 1일", "최근 7일", "전체", "사용자 지정"]
CTIT_SOURCE         = "ad_outcome"
MIN_CLICK_FILTER    = 30
```

### 19.3 ML 등급/위험 임계값 (아티팩트 실측)

```python
# M1 grade_info.json
GRADE_BINS   = [0.0, 20.02, 40.0, 59.9, 80.7, 100.0]
GRADE_LABELS = ["D", "C", "B", "A", "S"]

# M2 threshold_info.json
M2_BEST_F1_THRESHOLD = 0.5905
M2_DEFAULT_THRESHOLD  = 0.50

# M2 rule_info.json
MIN_EARLY_CLICK = 10   # early_click < 10 → rule_based_review
```

### 19.4 배지 색상 (v2.4)

```python
OPPORTUNITY_BADGE_COLORS = {
    "매체확장": "#1565C0",
    "승급추진": "#2E7D32",   # v2.3: "#6A1B9A"
}
RISK_ACTION_BADGE_COLORS = {
    "즉시조치": "#C62828",
    "우선검토": "#E65100",
}
BADGE_HERO_CONFIG = {
    "매체확장": {"icon": "↗", "tone": "#4A6BAA", "tone_soft": "#DCE6F2", "tone_deep": "#3A5A8A", "cta": "매체 확장 검토 →"},
    "승급추진": {"icon": "◐", "tone": "#2E7D32", "tone_soft": "#E8F5E9", "tone_deep": "#1B5E20", "cta": "예산 사전 확보 →"},
    "즉시조치": {"icon": "⚠", "tone": "#C25E55", "tone_soft": "#F4DCDA", "tone_deep": "#9A4842", "cta": "즉시 조치 →"},
    "우선검토": {"icon": "👁", "tone": "#E65100", "tone_soft": "#FFDAB3", "tone_deep": "#E65100", "cta": "모니터링 추가 →"},
}
BADGE_ENTRY_LABELS = {
    "매체확장": "M1 매체 확장 후보로 진입",
    "승급추진": "M1 승급 추진 대상으로 진입",
    "즉시조치": "M2 즉시 조치 필요로 진입",
    "우선검토": "M2 우선 검토로 진입",
}
```

### 19.5 운영 상태/룰 임계값

```python
NORMAL_RATIO_GREEN  = 80    # 정상 ≥ 80% → 초록 배너 (현재 P1에서 비활성)
NORMAL_RATIO_YELLOW = 60
DEPENDENCY_RED      = 70
DEPENDENCY_YELLOW   = 55
GRADE_THRESHOLDS    = {"A": 0.55, "C": 0.15}   # 레거시 룰 기반 등급(현재 ML 기반으로 대체)
QUALITY_WEIGHTS     = {"profitability": 0.35, "conversion_quality": 0.30, "scale": 0.20, "speed": 0.15}
```

### 19.6 유형별 운영 흐름 상수 (구현됨, 미연동 — [§16.4](#164-빌드되었으나-미연동된-기능-참고))

```python
TYPE_CATEGORY_MAP = {
    "설치형": "install", "가입형": "install",
    "클릭형": "click", "실행형": "click",
    "참여형": "engage", "기타": "engage",
    "구매형": "purchase", "노출형": "purchase",
}
FLOW_LOWER_IS_BETTER = {"cpa": True, "cvr": False, "turn": False, "margin_rate": False}
FLOW_ACTION_COLORS = {"urgent": "#e34948", "stable": "#2a78d6", "expand": "#1baf7a", "hold": "#898781"}
FLOW_ACTION_LABELS = {"urgent": "즉시조치", "stable": "안정운영", "expand": "확장후보", "hold": "보류"}
```

### 19.7 매핑 테이블 / 색상 팔레트 / 타이포그래피 / 칩 정렬 규칙

v2.3과 동일 — `ACTION_TYPE_MAP`, `MEDIA_NAME_MAP`, `ML_GRADE_COLORS`, `RISK_COLORS`, `TYPE_COLORS`, `MEDIA_COLORS`, `CATEGORY_COLORS`, `COLORS`(브랜드/KPI 막대/사분면 색상), 타이포그래피 토큰, `sort_chips()` 규칙은 변경 없음. 단, **사분면 색상의 "최우수/비싼우등/저효율/손실" 분류는 이제 항상 `margin_rate`를 Y축으로 사용**한다(§18.2).

### 19.8 제거된 상수/함수 (v2.4)

```python
# config.py에서 제거
HBAR_MIN_SHARE_PCT, HBAR_ROLLUP_LABEL   # 소액 매체 롤업 (settle_efficiency 차트 전용)
fmt_ratio()                              # 정산수익률 포맷 함수

# metrics.py에서 제거
calc_settle_efficiency()
```

### 19.9 숫자 포맷 함수

| 함수                    | 입력 예시 | 출력 예시  |
| ----------------------- | --------- | ---------- |
| `fmt_number(1234567)` | 1234567   | "1.2M"     |
| `fmt_pct(12.345)`     | 12.345    | "12.3%"    |
| `fmt_currency(50000)` | 50000     | "50,000원" |
| `fmt_won_man(125000)` | 125000    | "12.5만원" |

---

## 20. 소스 파일 맵

| 파일                                  | 역할               | 주요 함수 (v2.4 신규/변경 굵게)                                                                                                                                                                                                    |
| ------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                            | 엔트리포인트       | `st.navigation()`, `st.set_page_config()`                                                                                                                                                                                      |
| `pages/01_전체_광고_현황_오버뷰.py` | P1                 | **보기 모드 토글**, `render_ml_recommendation_banner()`                                                                                                                                                                    |
| `pages/02_유형별_운영_탐색.py`      | P2                 | **이동 조치 필요 광고(운영 우선순위) 섹션**                                                                                                                                                                                  |
| `pages/03_매체별_운영_탐색.py`      | P3                 | 버블차트,`build_media_insight_cards()`, `render_media_summary_table()`                                                                                                                                                         |
| `pages/04_카테고리별_운영_탐색.py`  | P4                 | 버블차트, Pareto,`render_media_summary_table()`                                                                                                                                                                                  |
| `pages/05_광고_상세.py`             | P5                 | 배지별 액션(`승급추진` 반영), Hero, SVG 게이지, 모달                                                                                                                                                                             |
| `src/config.py`                     | 상수, 포맷 함수    | **RAG 경로**, `OPPORTUNITY_BADGE_COLORS`(승급추진), `TYPE_CATEGORY_*`/`FLOW_*`(미연동)                                                                                                                                 |
| `src/init.py`                       | 세션 초기화        | `ensure_data_loaded()` (price_agg 인자 제거)                                                                                                                                                                                     |
| `src/data_loader.py`                | Parquet 로딩       | `load_price_agg()` **제거**                                                                                                                                                                                                |
| `src/preprocessing.py`              | JOIN, 필터링       | `build_base_table()` (price_agg 인자 제거), `build_ad_summary()`, `filter_by_period()`, `filter_by_hour()`, `get_previous_period()`                                                                                      |
| `src/metrics.py`                    | KPI 계산           | **`_GLOBAL_CACHE` 프로세스 전역 캐시**, `calc_settle_efficiency()` 제거                                                                                                                                                  |
| `src/model.py`                      | ML 파이프라인      | `score_all_ads()`, `predict_model1()`, `predict_model2()`                                                                                                                                                                    |
| `src/ml_insight_data.py`            | ML 인사이트 데이터 | `_assign_opportunity_badge()`(승급추진), `_assign_risk_badge()`, `opp_top_n`/`risk_top_n` 파라미터                                                                                                                         |
| `src/detail_data.py`                | P5 지원            | `build_hero_content()`/`build_why_steps()`(승급추진 반영), `get_kpi_contexts()`(5개 KPI로 축소)                                                                                                                              |
| `src/filters.py`                    | 필터 UI            | `render_filters()`, `render_applied_chips()`, `render_page_header()`                                                                                                                                                         |
| `src/charts.py`                     | Plotly 차트        | `make_heatmap()`, `make_hourly_cvr_trend()`, `make_bubble_chart()`, `make_pareto_chart()`, `make_media_cvr_bars()`, **`make_flow_bubble_chart()`(미연동)**                                                       |
| `src/components.py`                 | 재사용 UI          | **`render_ml_recommendation_banner()`**, **`render_media_highlight_card()`**, **`render_media_price_coop_card()`**, **`render_media_material_card()`**, **`render_media_summary_table()`** |
| `src/rules.py`                      | 룰 기반 로직       | `classify_quadrant()`(margin_rate 기본), `generate_alerts()`(margin_rate<0), **`build_media_insight_cards()`**, **`classify_flow_items()`(미연동)**                                                            |
| `src/agent.py`                      | Gemini AI          | `build_data_context()`, `generate_response()`(**RAG·Google Search·sources 통합**), `_extract_news_sources()`                                                                                                         |
| `src/rag.py`                        | **RAG 검색** | `retrieve()`, `_load_index()`, `_get_client()`                                                                                                                                                                               |

### 모델 아티팩트 맵 (실측 기준)

**M1** (`models/model1_artifacts/`): `model.joblib`, `label_encoders.joblib`, `feature_list.json`(48개), `ref_proba_sorted.npy`, `grade_info.json`(bins/labels), `metadata.json`(XGBClassifier, test_auc 0.8346, 학습일 2026-05-10).

**M2** (`models/model2_artifacts/`): `model.joblib`, `label_encoders.joblib`, `feature_list.json`(101개), `threshold_info.json`(best_f1 0.5905), `rule_info.json`(min_early_click 10), `metadata.json`(LGBM, test_prauc 0.6009, 학습일 2026-05-08).

**RAG** (`models/rag_artifacts/`): `index.parquet`(doc/section/text/embedding 컬럼, `scripts/build_rag_index.py`로 생성).

**공통**: `preprocessing_pipeline.joblib` (OHE + TF-IDF + numeric feats 정의).
