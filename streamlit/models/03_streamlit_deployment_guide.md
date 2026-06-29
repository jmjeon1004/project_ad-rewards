# Streamlit 연동 가이드 — Claude AI Agent 활용

작성일: 2026-05-08
대상 독자: Streamlit 구현 담당자
환경: VSCode + Claude AI Agent

---

## 1. 이 문서의 목적

ML1, ML2 두 모델은 노트북에서 학습 완료 후 배포용 산출물을 별도 폴더에 저장한 상태입니다. Streamlit 팀은 **노트북 코드를 직접 가져올 필요 없이**, 산출물만 로드해서 예측 함수를 만들면 됩니다.

이 문서는:
- 산출물 파일 구성과 각각의 역할
- Streamlit 앱이 따라야 할 추론 흐름
- 반드시 지켜야 할 주의사항
- VSCode + Claude AI Agent로 작업할 때 활용할 프롬프트 템플릿

을 담고 있습니다.

## 2. 받게 될 파일 구성

데이터분석 팀에서 넘겨주는 산출물은 다음 두 폴더입니다.

```
model1_artifacts/                    # 광고 고성과 예측 + 등급화
├── model.joblib                     # 튜닝된 XGB 모델
├── label_encoders.joblib            # LabelEncoder dict (train fit 결과)
├── feature_list.json                # 피처 리스트 (순서가 중요)
├── ref_proba_sorted.npy             # 점수 변환용 train+val proba 분포
├── grade_info.json                  # 등급 cutpoint
└── metadata.json                    # 모델/성능/전처리 노트

model2_artifacts/                    # 광고 조기부진 예측
├── model.joblib                     # 튜닝된 모델 (LGBM/XGB/RF/LR 중 하나)
├── label_encoders.joblib            # LabelEncoder dict
├── feature_list.json                # 피처 리스트
├── threshold_info.json              # threshold 후보 (default/best_f1/recall_targets)
├── rule_info.json                   # early_click >= 10 필터 룰
└── metadata.json                    # 모델/성능/전처리 노트
```

각 파일의 역할:

| 파일 | 왜 필요한가 |
|------|-------------|
| `model.joblib` | 학습된 모델 본체. `joblib.load`로 불러와 그대로 `predict_proba` 호출 |
| `label_encoders.joblib` | 학습 때 만들어진 인코더. **새로 fit하면 매핑이 달라져 예측이 어그러짐** |
| `feature_list.json` | 모델이 기대하는 피처와 그 **순서**. 입력 컬럼 순서가 다르면 잘못된 예측 |
| `ref_proba_sorted.npy` (M1) | 0~100점 점수 변환용 train+val proba 분포. 신규 데이터로 다시 만들면 안 됨 |
| `grade_info.json` (M1) | S/A/B/C/D 등급 cutpoint. 학습 시점에 확정된 값 사용 |
| `threshold_info.json` (M2) | 부진 판정 threshold 후보. 운영팀이 어떤 후보를 쓸지 선택 |
| `rule_info.json` (M2) | early_click 최소값 등 비즈니스 룰 |
| `metadata.json` | 디버깅/모니터링 참고. 직접 추론에 사용하진 않음 |

## 3. 디렉토리 구성 권장안

```
streamlit_app/
├── app.py                          # Streamlit 메인
├── requirements.txt                # 의존성
├── model1_artifacts/               # ML팀에서 받은 폴더 그대로
└── model2_artifacts/               # ML팀에서 받은 폴더 그대로
```

`requirements.txt`는 최소 다음을 포함:

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
joblib>=1.3
pyarrow>=14.0
```

## 4. 추론 흐름 (반드시 이 순서)

### 4.1 공통 전처리

학습 때 사용한 두 가지 함수를 **그대로** Streamlit 앱에 옮겨야 합니다. 정의를 바꾸면 컬럼명·인코딩이 학습 때와 어긋납니다.

```python
import re
import numpy as np
import pandas as pd

def clean_col_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols, seen = [], {}
    for c in df.columns:
        clean = re.sub(r'[^A-Za-z0-9_]', '_', c)
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols
    return df


def safe_transform(series: pd.Series, le) -> np.ndarray:
    known = set(le.classes_)
    return le.transform(
        [v if v in known else 'unknown' for v in series.astype(str)]
    )


def apply_label_encoders(df: pd.DataFrame, le_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = safe_transform(df[col], le)
    return df


def align_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    학습 때 피처 순서·구성과 정확히 맞춘다.
    - 누락된 컬럼은 0으로 채움 (운영 합의에 따라 변경 가능)
    - 학습에 없는 추가 컬럼은 버림
    - 컬럼 순서를 feature_list 순서로 강제
    """
    missing = [c for c in feature_list if c not in df.columns]
    for c in missing:
        df[c] = 0
    return df[feature_list]
```

### 4.2 Model 1 추론

```
입력 → clean_col_names → label_encode → align_features
     → model.predict_proba → percentile rank 매핑
     → 등급(S/A/B/C/D) 부여
```

### 4.3 Model 2 추론

```
입력 → early_click 확인
       ├── < 10 → 'rule_based_review' (ML 미적용)
       └── >= 10 → clean_col_names → label_encode → align_features
                  → model.predict_proba → threshold 비교
                  → 'decline_risk' / 'normal'
```

## 5. 반드시 지켜야 할 주의사항

다음 항목들은 어기면 예측이 조용히 잘못된 결과를 냅니다 (에러는 안 나지만 결과가 신뢰 불가).

1. **LabelEncoder 새로 fit 금지**
   학습 때 만든 `le_dict`를 그대로 로드해서 사용. unseen 값은 `safe_transform`으로 'unknown'에 매핑.

2. **피처 순서 보존**
   `feature_list.json` 순서대로 컬럼을 정렬. dict나 set으로 다루면 순서가 깨짐.

3. **`clean_col_names_unique` 동일 적용**
   학습 때 컬럼명을 정규화한 함수를 입력 데이터에도 똑같이 적용.

4. **점수 reference 분포 그대로 사용 (Model 1)**
   `ref_proba_sorted.npy`를 그대로 사용. 신규 데이터로 다시 percentile rank 만들면 안 됨.

5. **등급 cutpoint 그대로 사용 (Model 1)**
   `grade_info.json`의 bins/labels 그대로. 입력 데이터로 새로 qcut 하면 안 됨.

6. **early_click 룰 적용 (Model 2)**
   `rule_info.json`의 `min_early_click=10` 미만은 ML에 넣지 않음.

7. **`@st.cache_resource`로 1회 로드**
   모델은 무거우니 매 요청마다 로드하면 안 됨. 데코레이터로 1회 로드해 재사용.

## 6. app.py 골격 (제공됨)

같이 전달받은 `app.py`는 위 흐름을 구현한 골격입니다. 그대로 실행해도 동작하지만, 운영 환경에 맞춰 다음을 조정하세요.

- 입력 폼/업로드 UI (현재는 parquet/csv 업로드 가능)
- 결과 시각화 (등급 분포, threshold별 분기 등)
- 예외 처리 (파일 형식, 필수 컬럼 누락 등)
- 인증/권한 (사내 배포 시)

## 7. VSCode + Claude AI Agent 활용 가이드

VSCode에 Claude AI Agent를 연결해서 프롬프트로 Streamlit을 구현하는 워크플로우입니다.

### 7.1 작업 환경 준비

1. VSCode에서 `streamlit_app/` 폴더를 열기
2. `model1_artifacts/`, `model2_artifacts/`를 폴더 안에 복사
3. 받은 보고서 파일 3종(`01_model1_report.md`, `02_model2_report.md`, `03_streamlit_deployment_guide.md`)을 같은 폴더에 두기 (Agent가 컨텍스트로 참조)
4. `app.py` 파일이 있으면 그것을 시작점으로 사용

### 7.2 권장 프롬프트 패턴

#### 패턴 A — 산출물 검증 먼저

처음 받았을 때 산출물이 정상인지 검증하는 스크립트를 먼저 만들면 안전합니다.

```
프롬프트 예시:

streamlit_app/model1_artifacts와 model2_artifacts에 들어 있는
산출물을 모두 로드해서 다음을 점검하는 검증 스크립트를 작성해줘.

1. 모든 파일이 정상 로드되는지
2. feature_list.json의 피처 수와 model.joblib의 expected feature 수가 맞는지
3. label_encoders.joblib의 각 인코더에 'unknown' 클래스가 포함돼 있는지
4. ref_proba_sorted.npy가 정렬된 상태인지 (Model 1)
5. threshold_info.json의 모든 threshold가 0~1 범위인지 (Model 2)
6. metadata.json에 기록된 n_features가 feature_list 길이와 일치하는지

결과를 표로 출력하고 실패 항목은 빨간색으로 표시해줘.
```

#### 패턴 B — 추론 함수 단위로 만들기

```
프롬프트 예시:

03_streamlit_deployment_guide.md의 4.2절에 정의된
Model 1 추론 흐름을 그대로 구현하는 predict_model1 함수를
app.py에 추가해줘.

요구사항:
- 입력: pandas DataFrame (광고 등록 정보)
- 출력: DataFrame with columns [proba, score, grade]
- clean_col_names_unique, safe_transform, align_features 헬퍼 사용
- artifacts는 @st.cache_resource로 로드된 dict에서 받음
- 이미 정의된 함수는 재정의하지 말고 import만

테스트용으로 train parquet을 입력했을 때 정상 동작하는지
간단한 assertion도 같이 작성해줘.
```

#### 패턴 C — UI 추가

```
프롬프트 예시:

app.py의 Model 1 탭에 다음 입력 폼을 추가해줘.

- 파일 업로드 (parquet, csv) — 이미 있음, 유지
- 직접 입력 폼 — ads_idx 1개에 대해 주요 등록 정보를 입력해
  단건 예측 가능
  필요한 입력 필드는 metadata.json의 preprocessing_notes를
  참고해서 결정

UI는 sidebar가 아닌 메인 영역에 두고, 폼 제출 시 결과는
표가 아닌 카드 형태로 표시 (proba, score, grade를 큰 글씨로).
```

### 7.3 프롬프트 작성 팁

좋은 프롬프트의 공통점:
- **참조 문서 명시** ("03_streamlit_deployment_guide.md의 4.2절" 같이)
- **입출력 명확히** (DataFrame 컬럼명까지)
- **금지사항 명시** ("LabelEncoder 새로 fit하지 말 것" 등)
- **검증 같이 요청** (assertion, 시각 확인 등)

자주 빠지는 실수:
- "알아서 잘 만들어줘"식 프롬프트 → Agent가 학습 코드를 추측해서 잘못 구현할 위험
- 산출물 파일을 안 보여주고 추론 함수만 요청 → 피처 순서를 임의로 가정할 위험
- 한 프롬프트에 너무 많은 요구사항 → 일부가 누락되거나 충돌

### 7.4 Agent 작업 후 검증 체크리스트

Agent가 코드를 만든 뒤 다음을 직접 확인하세요.

- [ ] `joblib.load(...)`로 모델을 로드한 뒤 `model.feature_names_in_`(있다면)이 `feature_list.json`과 동일한 순서인가
- [ ] `apply_label_encoders` 호출 시 `le_dict`를 새로 fit하는 코드가 없는가
- [ ] `align_features`가 `feature_list.json` 순서대로 컬럼을 정렬하는가
- [ ] Model 1 점수 변환에 `ref_proba_sorted.npy`를 사용하고 있는가 (`np.searchsorted`)
- [ ] Model 2에서 `early_click < 10`인 row가 ML 추론을 우회하는가
- [ ] `@st.cache_resource`가 모델 로드 함수에 붙어 있는가
- [ ] 학습 때 사용한 parquet 파일을 입력해서 노트북 추론 결과와 동일한 proba가 나오는가 (가장 강력한 검증)

마지막 항목은 특히 중요합니다. 학습 시점 데이터에 동일한 모델을 적용했을 때 proba가 다르면, 어딘가에 전처리 차이가 있다는 신호입니다.

## 8. 실행

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열리면서 `http://localhost:8501`에 앱이 뜹니다.

## 9. 정기 재배포 (참고)

ML팀이 모델을 재학습하면 산출물 폴더 두 개를 새 버전으로 받게 됩니다.

권장 작업 순서:

1. 새 산출물 폴더를 받음
2. 패턴 A 검증 스크립트로 구조 점검
3. 학습 시점 test parquet으로 추론 결과 검증 (이전 버전과 성능 차이 확인)
4. 운영 환경에 폴더 교체 배포
5. metadata.json의 `created_at`, `test_auc`, `test_prauc`을 운영 모니터링에 기록


