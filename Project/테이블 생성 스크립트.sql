-- 광고 참여 목록에 라벨을 붙인 테이블
-- 원본 광고참여목록 + 완료정보 + 비용정보 + 이상치 플래그 + 제거조건 플래그
-- 사실상 팩트 테이블의 역할을 함
-- ads_join_info_labeled
-- 클릭키(1행=1클릭) 기준 통합 라벨링 테이블
-- 원본 ads_join_info + 완료정보 + 비용정보 + 이상치 플래그 + 제거조건 플래그

CREATE OR REPLACE TABLE ads_join_info_labeled AS
WITH base AS (
    SELECT
        ji.*,
        CAST(ji.click_day AS DATE) AS click_day,
        -- 완료 테이블 조인
        r.rwd_idx,
        r.regdate,
        r.ctit,
        -- 실제 비용 우선, 없으면 원본 fallback
        COALESCE(r.show_cost, ji.adv_price)       AS show_cost,
        COALESCE(r.adv_cost, ji.contract_price)   AS adv_cost,
        COALESCE(r.earn_cost, ji.media_price)     AS earn_cost,
        COALESCE(r.rwd_cost, ji.reward_price)     AS rwd_cost,
        CASE WHEN r.rwd_idx IS NOT NULL THEN 1
            ELSE 0
        END AS is_completed,
        CAST(r.regdate AS DATE) AS reg_day,
        CASE WHEN r.rwd_idx IS NOT NULL
             AND CAST(ji.click_day AS DATE) = CAST(r.regdate AS DATE)
            THEN 1 ELSE 0
        END AS is_same_day_complete,
        CASE WHEN r.rwd_idx IS NOT NULL
             AND CAST(ji.click_day AS DATE) < CAST(r.regdate AS DATE)
            THEN 1 ELSE 0
        END AS is_delayed_complete,
        CASE WHEN r.rwd_idx IS NOT NULL
            THEN COALESCE(r.adv_cost, ji.contract_price)
               - COALESCE(r.earn_cost, ji.media_price)
            ELSE NULL
        END AS margin
    FROM ads_join_info ji
    LEFT JOIN ads_rwd r
        ON ji.click_key = r.click_key
),
daily_ads AS (
    -- 광고 x 날짜 클릭수
    SELECT
        ads_idx,
        click_day,
        COUNT(*) AS clicks
    FROM base
    GROUP BY ads_idx, click_day
),
iqr_base AS (
    -- 광고별 사분위수
    SELECT
        ads_idx,
        quantile_cont(clicks, 0.25) AS q1,
        quantile_cont(clicks, 0.75) AS q3
    FROM daily_ads
    GROUP BY ads_idx
),
daily_flag AS (
    -- anomaly flag 생성
    SELECT
        d.ads_idx,
        d.click_day,
        d.clicks,
        i.q1,
        i.q3,
        (i.q3 - i.q1) AS iqr,
        (i.q3 + (i.q3 - i.q1) * 1.5) AS upper_bound,
        CASE WHEN d.clicks > (i.q3 + (i.q3 - i.q1) * 1.5)
            THEN 1 ELSE 0
        END AS is_anomaly_day
    FROM daily_ads d
    LEFT JOIN iqr_base i
        ON d.ads_idx = i.ads_idx
)
SELECT
    b.*,
    COALESCE(f.is_anomaly_day, 0) AS is_anomaly_day,
    -- 제거조건1 : anomaly_day 이면서 수익 없음(null or 0)
    CASE WHEN COALESCE(f.is_anomaly_day,0) = 1
         AND (
                b.margin IS NULL
                OR b.margin = 0
             )
        THEN 1 ELSE 0
    END AS remove_cond1_flag,
    -- 제거조건2 : anomaly_day 전체 제거
    CASE WHEN COALESCE(f.is_anomaly_day,0) = 1
        THEN 1 ELSE 0
    END AS remove_cond2_flag
FROM base b
LEFT JOIN daily_flag f
    ON b.ads_idx = f.ads_idx
   AND b.click_day = f.click_day;

select click_day 
FROM ads_join_info_labeled
limit 5;


-- 메인퍼널 테이블 (remove_cond2 적용)
-- 메인퍼널 테이블 (remove_cond2 적용)
CREATE OR REPLACE TABLE main_funnel AS
SELECT
    a.*,
    l.ads_type,
    l.ads_category,
    l.ads_save_way,
    l.ads_reward_price,
    l.ads_rejoin_type,
    CASE strftime(CAST(a.click_day AS DATE), '%w')
        WHEN '0' THEN 'Sun'
        WHEN '1' THEN 'Mon'
        WHEN '2' THEN 'Tue'
        WHEN '3' THEN 'Wed'
        WHEN '4' THEN 'Thu'
        WHEN '5' THEN 'Fri'
        WHEN '6' THEN 'Sat'
    END AS weekday
FROM ads_join_info_labeled a
LEFT JOIN ads_list l
    ON a.ads_idx = l.ads_idx
WHERE a.remove_cond2_flag = 0;



-- 광고 성과 비교용 테이블(remove_cond1 적용)
CREATE OR REPLACE TABLE ads_outcome AS
SELECT 
    l.ads_idx,
    l.ads_name,
    l.ads_type,
    l.ads_category,
    l.ads_reward_price,
    l.ads_order,
    l.ads_rejoin_type,
    COUNT(DISTINCT a.click_key) AS click_cnt,
    COUNT(DISTINCT a.rwd_idx)   AS complete_cnt,
    AVG(a.ctit)                 AS avg_ctit,
    SUM(a.rwd_cost)             AS total_reward_cost
FROM ads_list l
LEFT JOIN ads_join_info_labeled a
    ON l.ads_idx = a.ads_idx
   AND a.remove_cond1_flag = 0
GROUP BY
    l.ads_idx,
    l.ads_name,
    l.ads_type,
    l.ads_category,
    l.ads_reward_price,
    l.ads_order,
    l.ads_rejoin_type;


-- 유저 x 일자 활동 테이블(remove_cond1 적용)
CREATE OR REPLACE TABLE tb_user_daily_activity AS
SELECT 
    a.dvc_idx, 
    a.click_day AS active_date,
    a.mda_idx,
    COUNT(a.click_key) AS click_cnt,
    COUNT(a.rwd_idx) AS complete_cnt,
    COUNT(DISTINCT a.ads_idx) AS unique_ads_cnt,
    MIN(a.click_date) AS first_click,
    MAX(a.click_date) AS last_click,
    AVG(a.ctit) AS mean_ctit,
    APPROX_QUANTILE(a.ctit, 0.5) AS median_ctit,
    MIN(a.ctit) AS min_ctit,
    MAX(a.ctit) AS max_ctit
FROM ads_join_info_labeled a
WHERE a.remove_cond1_flag = 0
GROUP BY
    a.dvc_idx,
    a.click_day,
    a.mda_idx;


-- 캠페인 날짜 캘린더 테이블 (remove_cond1 적용)
CREATE OR REPLACE TABLE sched AS
SELECT
    l.ads_idx,
    CAST(a.click_day AS DATE) AS click_date,
    l.ads_name,
    l.ads_type,
    l.ads_category,
    l.ads_sdate,
    l.ads_edate,
    CASE 
        WHEN l.ads_sdate IS NULL OR l.ads_sdate = '0000-00-00 00:00:00' 
        THEN NULL
        ELSE DATEDIFF(
            'day',
            CAST(l.ads_sdate AS DATE),
            CAST(a.click_day AS DATE)
        ) + 1
    END AS campaign_n_day,
    l.ads_day_cap,
    COUNT(DISTINCT a.click_key) AS click_cnt,
    COUNT(DISTINCT a.rwd_idx) AS complete_cnt,
    ROUND(
        SUM(CASE WHEN l.ads_day_cap = 'Y' THEN 1.0 ELSE 0 END) / COUNT(*) * 100,
        2
    ) AS day_cap_ratio
FROM ads_list l
LEFT JOIN ads_join_info_labeled a
    ON l.ads_idx = a.ads_idx
   AND a.remove_cond1_flag = 0
GROUP BY
    l.ads_idx,
    a.click_day,
    l.ads_name,
    l.ads_type,
    l.ads_category,
    l.ads_sdate,
    l.ads_edate,
    l.ads_day_cap;


-- 재무테이블_광고목록 (remove_cond1 버전)
CREATE OR REPLACE TABLE tb_재무테이블_광고목록_clean1 AS
SELECT
    a.rwd_idx,
    a.click_key,
    a.ads_idx,
    a.mda_idx,
    CAST(a.click_date AS TIMESTAMP) AS click_date,
    CAST(a.regdate AS TIMESTAMP) AS regdate,
    a.ctit,
    a.show_cost,
    a.adv_cost,
    a.earn_cost,
    a.rwd_cost,
    a.margin AS ive_margin,
    l.ads_name,
    l.ads_type,
    l.ads_category,
    l.ads_save_way,
    l.ads_order,
    l.ads_rejoin_type,
    l.ads_reward_price
FROM ads_join_info_labeled a
LEFT JOIN ads_list l
    ON a.ads_idx = l.ads_idx
WHERE a.rwd_idx IS NOT NULL
  AND a.remove_cond1_flag = 0
  
 
-- 재무테이블_광고목록(remove_cond2 버전)
CREATE OR REPLACE TABLE tb_재무테이블_광고목록_clean2 AS
SELECT
    a.rwd_idx,
    a.click_key,
    a.ads_idx,
    a.mda_idx,
    CAST(a.click_date AS TIMESTAMP) AS click_date,
    CAST(a.regdate AS TIMESTAMP) AS regdate,
    a.ctit,
    a.show_cost,
    a.adv_cost,
    a.earn_cost,
    a.rwd_cost,
    a.margin AS ive_margin,
    l.ads_name,
    l.ads_type,
    l.ads_category,
    l.ads_save_way,
    l.ads_order,
    l.ads_rejoin_type,
    l.ads_reward_price
FROM ads_join_info_labeled a
LEFT JOIN ads_list l
    ON a.ads_idx = l.ads_idx
WHERE a.rwd_idx IS NOT NULL
  AND a.remove_cond2_flag = 0; 
  


--파케이로 변환
COPY ads_outcome
TO 'C:\Users\milkl\OneDrive\문서\sparta_project\final_project\cleaned_tables\ads_outcome.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);

COPY main_funnel
TO 'C:\Users\milkl\OneDrive\문서\sparta_project\final_project\cleaned_tables\main_funnel.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);

COPY sched
TO 'C:\Users\milkl\OneDrive\문서\sparta_project\final_project\cleaned_tables\sched.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);

COPY tb_user_daily_activity
TO 'C:\Users\milkl\OneDrive\문서\sparta_project\final_project\cleaned_tables\tb_user_daily_activity.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);

COPY tb_재무테이블_광고목록_clean1
TO 'C:\Users\milkl\OneDrive\문서\sparta_project\final_project\cleaned_tables\finance_clean1.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);

COPY tb_재무테이블_광고목록_clean2
TO 'C:\Users\milkl\OneDrive\문서\sparta_project\final_project\cleaned_tables\finance_clean2.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);