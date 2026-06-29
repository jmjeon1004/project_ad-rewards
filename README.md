# 아이브코리아 광고운영 최적화 대시보드

> **13조 최종 프로젝트** | Streamlit 기반 광고 성과 분석 + ML 품질 등급 + 조기부진 예측 + AI Agent 통합 운영 의사결정 지원 도구

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [데이터 구성](#데이터-구성)
- [ML 모델](#ml-모델)
- [AI Agent](#ai-agent)

---

## 프로젝트 개요

**평가 기준 부재, 광고 분류 체계의 혼재, 사후 대응 중심 운영** 문제를 해결하기 위해 다음을 구축하였습니다.

| 문제 | 해결책 |
|------|--------|
| 광고 분류 체계 혼재 | BERTopic 기반 광고 재분류 체계 |
| 평가 기준 부재 | XGBoost 기반 광고 품질 평가 모델 (M1: S/A/B/C/D 등급) |
| 사후 대응 중심 운영 | LightGBM 기반 조기 부진 예측 모델 (M2: D+3 조기 경보) |

운영자가 광고 현황을 모니터링하고 의사결정을 지원받을 수 있도록 **Streamlit 대시보드**와 **AI Agent(Gemini 2.5 Flash)**를 개발하였으며, 광고 운영 전 과정을 데이터 기반으로 지원하는 의사결정 체계를 설계하는 것을 목표로 하였습니다.

---

## 주요 기능

| 페이지 | 핵심 기능 |
|--------|-----------|
| **P1 전체 오버뷰** | 전체 기간 모드: KPI 카드 + 시간대×요일 CVR 히트맵 + Top 10 광고 리스트<br>당일 vs 평시 모드: CVR 추이 라인차트(시간 클릭 → 이상 광고) + 차원별 Top 3 카드 |
| **P2 유형별 탐색** | 기간/유형/시간대 필터 + 이동 조치 필요 광고 패널(즉시조치/우선검토/매체확장/승급추진) + 유형별 광고 리스트 |
| **P3 매체별 탐색** | CPA×마진율×완료수 버블 차트(클릭 → 매체 광고 리스트) + 매체 인사이트 카드 4종(최우수/비효율/단가협업/소재개선) |
| **P4 카테고리별 탐색** | CVR or CPA 선택형 버블 차트 + 운영 알림 4종(★최우수/↺소재개선/⚠단가점검/⊘비효율) + 파레토 차트 |
| **P5 광고 상세** | M1 등급 or M2 부진확률 스코어 카드 + KPI 그리드 + 배지별 액션(매체확장/승급추진/즉시조치/우선검토) + 판정 근거 설명 |

> P1~P4 각 페이지 하단에 AI Agent 채팅(Gemini 2.5 Flash) 제공.

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Frontend | Streamlit (멀티페이지, `st.navigation`) |
| Charts | Plotly |
| Data | Pandas, PyArrow (Parquet) |
| ML | XGBoost, LightGBM, scikit-learn, joblib |
| AI Agent | Google Generative AI (Gemini 2.5 Flash) |
| 전처리 분류 | BERTopic |

---

## 프로젝트 구조

```
project_ad-rewards/
├── DATA/                          # 원본 데이터
│   ├── xlsx/                      # 원본 엑셀 파일
│   ├── parquet_table/             # 변환된 parquet 파일
│   └── eda_ready_new/             # 전처리 완료 데이터
│
├── Project/                       # 분석 노트북
│   ├── 01_01_ad_classification_bertopic.ipynb   # BERTopic 광고 분류
│   ├── 01_02_preprocessing.ipynb                # 데이터 전처리
│   ├── 02_EDA.ipynb                             # 탐색적 데이터 분석
│   ├── 03_00_ML_EDA.ipynb                       # ML용 EDA
│   ├── 03_01_ML_mart.ipynb                      # ML 마트 구성
│   ├── 03_02_ML1.ipynb                          # M1 (XGBoost) 학습
│   ├── 03_03_ML2.ipynb                          # M2 (LightGBM) 학습
│   └── 테이블 생성 스크립트.sql
│
└── streamlit/                     # 대시보드 앱
    ├── app.py                     # 엔트리포인트
    ├── requirements.txt
    ├── pages/
    │   ├── 01_전체_광고_현황_오버뷰.py
    │   ├── 02_유형별_운영_탐색.py
    │   ├── 03_매체별_운영_탐색.py
    │   ├── 04_카테고리별_운영_탐색.py
    │   └── 05_광고_상세.py
    ├── src/
    │   ├── config.py              # 상수, 색상, 포맷 함수
    │   ├── init.py                # 세션 초기화 (데이터 로딩 + ML 스코어링)
    │   ├── data_loader.py         # Parquet 로딩 + 캐싱
    │   ├── preprocessing.py       # JOIN, 날짜/시간 변환
    │   ├── metrics.py             # CVR, CPA, 마진율 계산
    │   ├── model.py               # M1 + M2 파이프라인
    │   ├── ml_insight_data.py     # ML 인사이트 + 배지 로직
    │   ├── detail_data.py         # P5 상세 데이터 준비
    │   ├── filters.py             # 공통 필터 렌더링
    │   ├── charts.py              # Plotly 차트 (버블, 히트맵, 파레토 등)
    │   ├── components.py          # KPI 카드, 배너, 운영 알림 카드
    │   ├── rules.py               # 룰 기반 운영 판단 (긴급/주의/정상)
    │   ├── agent.py               # Gemini 2.5 Flash AI Agent
    │   └── rag.py                 # RAG 검색 (지식베이스 활용)
    ├── models/
    │   ├── model1_artifacts/      # M1 (XGBoost) — model.joblib 외
    │   ├── model2_artifacts/      # M2 (LightGBM) — model.joblib 외
    │   └── preprocessing_pipeline.joblib
    └── data/                      # 대시보드용 Parquet 파일 12개
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
cd streamlit
pip install -r requirements.txt
```

### 2. API 키 설정

AI Agent(Gemini) 기능을 사용하려면 Streamlit Secrets에 API 키를 설정합니다.

`.streamlit/secrets.toml` 파일을 생성합니다:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

> AI Agent 없이도 대시보드 성과 분석 및 ML 기능은 정상 동작합니다.

### 3. 앱 실행

```bash
streamlit run app.py

# 포트 지정 시
streamlit run app.py --server.port 8502
```

---

## 데이터 구성

`streamlit/data/` 디렉토리의 Parquet 파일 (핵심 JOIN 키: `ads_idx`)

| 파일 | 설명 |
|------|------|
| `hourly_report.parquet` | 시간대별 광고 성과 fact table (클릭, 완료, 광고비, 정산비) |
| `ad_attr_map.parquet` | 광고 속성 (유형, 카테고리, 리워드, 저장방식, 일cap) |
| `ive_ad_classification.parquet` | 매체/액션 분류 (`final_media`, `final_action`) |
| `ad_outcome.parquet` | 광고별 전환 결과 집계 (avg_ctit, CTIT 소스) |
| `main_funnel.parquet` | 원시 퍼널 이벤트 |
| `ad_master_clean.parquet` | 광고 등록 마스터 (ML 피처용) |
| `sched_clean.parquet` | 캠페인 스케줄 (운영 일수, 클릭) |
| `finance_clean1.parquet` | 정산 데이터 1 |
| `finance_clean2.parquet` | 정산 데이터 2 |
| `user_daily_activity_clean.parquet` | 유저 일별 활동 |

### 핵심 지표 계산 규칙

> 모든 지표는 **합계 기반 비율**로 계산합니다 (행별 비율의 산술평균 금지).

```
CVR(%) = sum(완료수) / sum(클릭수) × 100
CPA(원) = sum(광고비) / sum(완료수)
마진율(%) = (sum(광고비) − sum(정산비)) / sum(광고비) × 100
```

분모가 0인 경우 `null` 처리, UI에서 `-` 표시.

---

## ML 모델

### M1 — 광고 품질 등급 (XGBoost, 50 features)

등록 시점 속성 기반으로 광고의 초기 품질을 예측합니다.

| 등급 | 설명 | 점수 범위 |
|------|------|-----------|
| S | 최우수 | > 80.7 |
| A | 우수 | 59.9 ~ 80.7 |
| B | 보통 | 40.0 ~ 59.9 |
| C | 주의 | 20.02 ~ 40.0 |
| D | 개선필요 | < 20.02 |

- 점수: quantile-based percentile 스코어링 (0~100)
- 운영 상태: S/A/B → 정상, C → 주의, D → 긴급

### M2 — 조기부진 예측 (LightGBM, 101 features)

D+3 초기 실적 기반으로 부진 위험을 조기 감지합니다.

| 판정 | 조건 | 설명 |
|------|------|------|
| `rule_based_review` | early_click < 10 | 데이터 부족으로 판단 보류 |
| `decline_risk` | m2_proba ≥ threshold | 부진 위험 |
| `normal` | 나머지 | 정상 |

### 배지 시스템

- **기회 배지**: 등급경계(상향 가능), 매체확장 추천
- **위험 배지**: 즉시조치 필요, 우선검토 대상

---

## AI Agent

Gemini 2.5 Flash 기반 자연어 질의 인터페이스입니다.

- **현재 필터 기준 집계 데이터**를 컨텍스트로 제공하여 답변
- **Google Search 그라운딩**: 외부 맥락이 필요한 질문(원인 해석, 업계 이슈 등)에 실시간 검색 활용
- **RAG 지식베이스**: PRD, 운영 룰, ML 등급 정의 등 정책성 질문에 활용
- 응답 형식: JSON 구조화 출력 (요약, 근거, 해석, 액션, 출처)

응답 예시:
```json
{
  "summary": "핵심 요약 한 문장",
  "evidence": [{"metric": "CVR", "value": "12.3%", "scope": "전체"}],
  "interpretation": "왜 이런 결과가 나왔는지 해석",
  "actions": ["추천 액션 1", "추천 액션 2"],
  "sources": [{"doc": "운영룰", "section": "CPA 기준"}]
}
```

---

## 팀 정보

**13조** — 데이터 분석 최종 프로젝트
