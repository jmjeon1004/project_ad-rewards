# ML 모델 — M1 품질등급 / M2 조기부진 예측

## M1: 광고 품질 등급 개요

광고 등록 시점의 속성(보상가, 유형, 매체, 일캡 등)만으로 품질 등급을 예측하는 모델.

- 알고리즘: XGBClassifier
- 피처 수: 50개
- 적용 대상: `is_valid_click10 == 1` AND `click_cnt >= 30`
- 아티팩트: `models/model1_artifacts/`
- 성능: Test AUC 약 0.85

## M1 등급 체계 (5-grade quantile cut)

점수는 0~100점, 등급은 quantile 기반 분위 컷(`grade_info.json`)으로 결정한다.

| 등급 | 라벨 | 점수 범위 | 운영 의미 |
| --- | --- | --- | --- |
| S | 최우수 | 80.7 초과 | 적극 집행, 매체 확장 검토 |
| A | 우수 | 59.9 ~ 80.7 | 안정 운영, S 진입 가능성 모니터링 |
| B | 보통 | 40.0 ~ 59.9 | 관찰, 소재/타겟 최적화 여지 |
| C | 주의 | 20.02 ~ 40.0 | 경고, 개선 미시행 시 축소 검토 |
| D | 개선필요 | 20.02 미만 | 즉시 축소/중단 검토 |

## M1 등급 → 운영 상태 연동

| M1 등급 | 운영 상태 |
| --- | --- |
| S, A, B | 정상 |
| C | 주의 |
| D | 긴급 |

`src/rules.py`의 `classify_operation_status()`가 이 매핑을 구현한다. 모델 미적용(점수 없음) 광고는 운영 상태 집계에서 제외한다.

## M2: 조기부진 예측 개요

광고 집행 후 D+0~D+3 초기 실적을 기반으로 향후 부진 위험을 예측하는 모델. 사후 1~2주 뒤가 아니라 D+3 시점에 조기 경보를 발령한다.

- 알고리즘: LightGBM
- 피처 수: 101개 (M1 공통 피처 50개 + TF-IDF(ads_name, ads_summary) + early_click)
- 적용 대상: `is_valid_click10 == 1`
- 아티팩트: `models/model2_artifacts/`
- 성능: Test PR-AUC 약 0.72

## M2 의사결정 로직

```
early_click < 10 (min_early_click)
  → rule_based_review  (관측 부족, ML 판단 보류, 별도 룰 기반 관리)

early_click >= 10:
  m2_proba >= threshold → decline_risk  (부진 위험)
  m2_proba <  threshold → normal        (정상)
```

- `min_early_click = 10`: early_click(D+3 이내 클릭 합)이 10 미만이면 ML 예측을 적용하지 않고 rule_based_review로 분류 (출처: `models/model2_artifacts/rule_info.json`)
- early_click 계산: 캠페인 스케줄(`sched_clean.parquet`)에서 `campaign_n_day < 3`인 행의 클릭수 합

## M2 Threshold 옵션

`models/model2_artifacts/threshold_info.json` 기준:

| 키 | 설명 | 값 |
| --- | --- | --- |
| `best_f1` | F1 점수 최적화 threshold (기본 사용) | 약 0.59 |
| `default_05` | 기본값 | 0.50 |
| `recall_targets["Recall >= 0.70"]` | 재현율 0.70 보장 threshold | 약 0.51 |
| `recall_targets["Recall >= 0.80"]` | 재현율 0.80 보장 threshold | 약 0.39 |

Threshold를 바꿔도 모델 재추론은 필요 없다 — 캐시된 `m2_proba`에 새 threshold를 재적용(`reapply_threshold()`)해 `m2_decision`만 다시 계산한다. 단, val 기준 탐색값이므로 test 데이터에 적용 시 recall이 보장되지 않을 수 있다.
