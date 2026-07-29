# 지표 정의

## 도메인 규칙

- 모든 비율 지표는 합계 기반으로 계산한다. 행별 비율의 산술평균은 사용하지 않는다.
- 분모가 0일 때는 null을 반환하고, UI에서는 "-"로 표시한다.

## CVR (전환율)

CVR = sum(turn) / sum(clk) × 100

- turn: 전환수(완료수), clk: 클릭수
- 분모(clk)가 0이면 null
- 구현: `src/metrics.py`의 `calc_cvr()`

## CPA (전환단가)

CPA = sum(acost) / sum(turn)

- acost: 광고비, turn: 전환수
- 분모(turn)가 0이면 null
- 구현: `src/metrics.py`의 `calc_cpa()`

## 마진율

마진율 = (sum(acost) - sum(earn)) / sum(acost) × 100

- acost: 광고비, earn: 정산비(지급액)
- 분모(acost)가 0이면 null
- 구현: `src/metrics.py`의 `calc_margin_rate()`

## TOP 10 의존도

의존도 = sum(전환수 상위 10개 광고) / sum(전체 전환수) × 100

- 구현: `src/rules.py`의 `calc_dependency_ratio()`
- 의존도가 높을수록(예: 70% 이상) 소수 광고에 성과가 집중되어 리스크가 큼

## 집계 레벨

- `calc_all_kpis(df)`: 전체 단일 집계 → dict
- `calc_group_kpis(df, group_col)`: 그룹별 집계 (예: 매체별, 유형별)
- `calc_daily_kpis(df)`: 일자별 집계
- `calc_hourly_kpis(df)`: 시간대별 집계

## 전기 비교 (Period-over-Period)

`calc_change_rate(current, previous)`는 단순 차이(current - previous)를 반환한다.

| 현재 기간 | 비교 기간 |
| --- | --- |
| 최근 1일 | 전일 1일 |
| 최근 7일 | 직전 7일 |
| 사용자 지정 (N일) | 시작일 직전 N일 |
| 전체 | 비교 없음 |
