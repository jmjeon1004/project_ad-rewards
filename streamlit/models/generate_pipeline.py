"""
ml_mart.ipynb의 핵심 로직을 추출하여 preprocessing_pipeline.joblib 생성.
OHE, TF-IDF, 피처 리스트 등 모델 추론에 필요한 fitted 객체를 저장한다.

실행: python models/generate_pipeline.py
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data') + os.sep
OUT  = os.path.dirname(os.path.abspath(__file__)) + os.sep
SEED = 42

print(f"DATA: {BASE}")
print(f"OUT:  {OUT}")

# ── 1. 데이터 로드 ──
print("\n[1] 데이터 로드 중...")
ad_outcome = pd.read_parquet(BASE + 'ad_outcome.parquet')
ad_attr    = pd.read_parquet(BASE + 'ad_attr_map.parquet')
finance1   = pd.read_parquet(BASE + 'finance_clean1.parquet')
finance2   = pd.read_parquet(BASE + 'finance_clean2.parquet')
sched      = pd.read_parquet(BASE + 'sched_clean.parquet')
ad_master  = pd.read_parquet(BASE + 'ad_master_clean.parquet')
ad_class   = pd.read_parquet(BASE + 'ive_ad_classification.parquet')
finance    = pd.concat([finance1, finance2], ignore_index=True)
print(f"  ad_outcome: {ad_outcome.shape}, ad_master: {ad_master.shape}")

# ── 2. 유효 광고 선정 ──
print("\n[2] 유효 광고 선정 (click_cnt >= 30)...")
valid = ad_outcome[ad_outcome['click_cnt'] >= 30].copy()
print(f"  전체: {len(ad_outcome):,} -> 유효: {len(valid):,}건")

fin_agg = finance.groupby('ads_idx').agg(
    total_margin   = ('ive_margin', 'sum'),
    complete_cnt_f = ('rwd_idx', 'count'),
    avg_ctit_fin   = ('ctit', 'mean'),
).reset_index()
valid = valid.merge(fin_agg, on='ads_idx', how='left')
valid = valid.drop(columns=['avg_ctit'], errors='ignore')
valid = valid.rename(columns={'avg_ctit_fin': 'avg_ctit'})

# ── 3. Raw 변수 가공 ──
print("\n[3] Raw 변수 가공...")
valid['total_margin']   = valid['total_margin'].fillna(0)
valid['complete_cnt_f'] = valid['complete_cnt_f'].fillna(0)
valid['log_margin']     = np.log1p(valid['total_margin'].clip(lower=0))
valid['log_complete']   = np.log1p(valid['complete_cnt_f'])
valid['is_ctit_null']   = valid['avg_ctit'].isna().astype(int)

# ── 4. Split ──
print("\n[4] Train/Val/Test 분할...")
valid['_strat'] = pd.qcut(valid['log_margin'], q=5, labels=False, duplicates='drop')
idx_all = valid.index.tolist()
idx_train, idx_temp = train_test_split(idx_all, test_size=0.4, random_state=SEED,
                                        stratify=valid.loc[idx_all, '_strat'])
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=SEED,
                                      stratify=valid.loc[idx_temp, '_strat'])
df_train_raw = valid.loc[idx_train].copy()
df_val_raw   = valid.loc[idx_val].copy()
df_test_raw  = valid.loc[idx_test].copy()
print(f"  Train: {len(df_train_raw):,}, Val: {len(df_val_raw):,}, Test: {len(df_test_raw):,}")

# ── 5. Train fit: ctit_median, GLOBAL_CVR ──
print("\n[5] Train fit (ctit_median, GLOBAL_CVR)...")
ctit_median = df_train_raw['avg_ctit'].median()
GLOBAL_CVR = df_train_raw['complete_cnt'].sum() / df_train_raw['click_cnt'].sum()
print(f"  ctit_median={ctit_median:.0f}초, GLOBAL_CVR={GLOBAL_CVR*100:.2f}%")

def apply_train_fits(df):
    df = df.copy()
    df['avg_ctit_imputed'] = df['avg_ctit'].fillna(ctit_median)
    df['log_ctit']         = np.log1p(df['avg_ctit_imputed'].clip(lower=0))
    df['smoothed_cvr']     = (df['complete_cnt'] + 10 * GLOBAL_CVR) / (df['click_cnt'] + 10) * 100
    return df

df_train_raw = apply_train_fits(df_train_raw)
df_val_raw   = apply_train_fits(df_val_raw)
df_test_raw  = apply_train_fits(df_test_raw)

# ── 6. GroupedECDFScorer ──
print("\n[6] GroupedECDFScorer fit...")

class GroupedECDFScorer:
    def __init__(self, group_col, value_cols, invert_cols=None,
                 min_group_size=30, fallback_min_size=10):
        self.group_col = group_col
        self.value_cols = list(value_cols)
        self.invert_cols = set(invert_cols or [])
        self.min_group_size = min_group_size
        self.fallback_min_size = fallback_min_size
        self.ecdf_ = {}
        self.global_ecdf_ = {}
        self.group_sizes_ = {}
        self.fitted_ = False

    def fit(self, df):
        for col in self.value_cols:
            arr = df[col].dropna().values
            self.global_ecdf_[col] = np.sort(arr)
        for g, sub in df.groupby(self.group_col):
            self.group_sizes_[g] = len(sub)
            if len(sub) >= self.min_group_size:
                for col in self.value_cols:
                    arr = sub[col].dropna().values
                    if len(arr) >= self.min_group_size:
                        self.ecdf_[(g, col)] = np.sort(arr)
        self.fitted_ = True
        return self

    def transform(self, df):
        assert self.fitted_
        n = len(df)
        out = pd.DataFrame(index=df.index)
        groups = df[self.group_col].values
        for col in self.value_cols:
            scores = np.full(n, 0.5)
            levels = np.zeros(n, dtype=int)
            vals = df[col].values
            for g in pd.unique(groups):
                mask = groups == g
                sub_vals = vals[mask]
                if (g, col) in self.ecdf_:
                    sorted_train = self.ecdf_[(g, col)]
                    lvl = 0
                elif len(self.global_ecdf_.get(col, [])) >= self.fallback_min_size:
                    sorted_train = self.global_ecdf_[col]
                    lvl = 1
                else:
                    scores[mask] = 0.5
                    levels[mask] = 2
                    continue
                valid_mask = ~pd.isna(sub_vals)
                sub_scores = np.full(mask.sum(), 0.5)
                sub_levels = np.full(mask.sum(), lvl, dtype=int)
                if valid_mask.any():
                    v = sub_vals[valid_mask].astype(float)
                    n_train = len(sorted_train)
                    if n_train > 1:
                        left  = np.searchsorted(sorted_train, v, side='left')
                        right = np.searchsorted(sorted_train, v, side='right')
                        avg_rank = (left + right) / 2
                        p = np.clip((avg_rank + 0.5) / n_train, 0.0, 1.0)
                    else:
                        p = np.full(len(v), 0.5)
                    if col in self.invert_cols:
                        p = 1 - p
                    sub_scores[valid_mask] = p
                sub_levels[~valid_mask] = 2
                sub_scores[~valid_mask] = 0.5
                scores[mask] = sub_scores
                levels[mask] = sub_levels
            out[f'score_{col}'] = scores
            out[f'fallback_{col}'] = levels
        return out

SCORE_VALUE_COLS = ['smoothed_cvr', 'log_margin', 'log_complete', 'log_ctit']
scorer = GroupedECDFScorer(
    group_col='analysis_ads_type_label',
    value_cols=SCORE_VALUE_COLS,
    invert_cols=['log_ctit'],
    min_group_size=30,
    fallback_min_size=10,
).fit(df_train_raw)
print(f"  그룹 ECDF 등록: {len(scorer.ecdf_) // len(SCORE_VALUE_COLS)}개 그룹")

# ── 7. 라벨 threshold ──
print("\n[7] 라벨 threshold 계산...")
SCORE_COL_MAP = {
    'score_smoothed_cvr': 'score_cvr',
    'score_log_margin':   'score_margin',
    'score_log_complete': 'score_complete',
    'score_log_ctit':     'score_ctit',
}

def apply_scorer(df):
    df = df.copy()
    score_df = scorer.transform(df)
    score_df = score_df.rename(columns=SCORE_COL_MAP)
    for c in score_df.columns:
        if c.startswith('score_') or c.startswith('fallback_'):
            df[c] = score_df[c].values
    df['quality_score_LABEL_ONLY'] = (
        df['score_margin']   * 0.35 +
        df['score_cvr']      * 0.30 +
        df['score_complete'] * 0.20 +
        df['score_ctit']     * 0.15
    ) * 100
    return df

df_train_raw = apply_scorer(df_train_raw)
THRESHOLD_TOP25 = df_train_raw['quality_score_LABEL_ONLY'].quantile(0.75)
print(f"  모델1 고성과 threshold: {THRESHOLD_TOP25:.2f}점")

# ── 8. 피처 전처리 ──
print("\n[8] 피처 전처리...")
NUMERIC_FEATS = [
    'ads_reward_price','ads_rejoin_type','ads_order',
    'ads_action_rule','ads_action_diff_flag','ads_day_cap',
    'reg_hour','reg_hour_band','reg_weekday_enc','reg_is_weekend',
    'ads_require_adid','action_target_cnt',
    'mentioned_media_cnt','target_media_cnt',
]
OHE_FEATS = [
    'analysis_ads_type_label','reward_band',
    'final_media','final_action','ads_os_type',
]
EARLY_FEATS = [
    'early_click','early_complete',
    'click_day1','click_day2','click_day3',
    'complete_day1','complete_day2','complete_day3',
    'click_trend',
]

# 피처 소스 데이터
attr = (ad_attr[['ads_idx','ads_reward_price','ads_rejoin_type','ads_order',
                  'ads_action_rule','ads_action_diff_flag','ads_day_cap']]
        .drop_duplicates('ads_idx', keep='last').copy())
attr['ads_day_cap'] = attr['ads_day_cap'].astype(int)

master_feat = (ad_master[['ads_idx','regdate','ads_os_type','ads_require_adid',
                           'action_target_cnt','mentioned_media_cnt','target_media_cnt']]
               .drop_duplicates('ads_idx', keep='first').copy())
master_feat['regdate'] = pd.to_datetime(master_feat['regdate'])
master_feat['reg_hour']        = master_feat['regdate'].dt.hour
master_feat['reg_weekday_enc'] = master_feat['regdate'].dt.dayofweek
master_feat['reg_is_weekend']  = master_feat['reg_weekday_enc'].isin([5,6]).astype(int)

def hour_band(h):
    if pd.isna(h): return np.nan
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3

master_feat['reg_hour_band']    = master_feat['reg_hour'].apply(hour_band)
master_feat['ads_require_adid'] = master_feat['ads_require_adid'].astype(int)
master_feat['ads_os_type']      = master_feat['ads_os_type'].astype(str)

class_feat = (ad_class[['ads_idx','final_media','final_action']]
              .drop_duplicates('ads_idx', keep='last').copy())

# 텍스트 준비
name_df    = ad_attr[['ads_idx','ads_name']].drop_duplicates('ads_idx', keep='last').copy()
summary_df = ad_master[['ads_idx','ads_summary']].drop_duplicates('ads_idx', keep='first').copy()
text_df = name_df.merge(summary_df, on='ads_idx', how='left')
text_df['ads_summary'] = text_df['ads_summary'].fillna('')
text_df['text_raw'] = (text_df['ads_name'].fillna('') + ' ' + text_df['ads_summary']).str.strip()

# 초기 3일 실적
sched2 = sched.copy()
sched2['click_date_dt'] = pd.to_datetime(sched2['click_date'])
sched2['ads_sdate_dt']  = pd.to_datetime(sched2['ads_sdate'])
sched2['elapsed_day']   = (sched2['click_date_dt'] - sched2['ads_sdate_dt']).dt.days
sched2 = sched2[sched2['elapsed_day'] >= 0].copy()

early3 = (sched2[sched2['elapsed_day'] < 3]
          .groupby('ads_idx').agg(
              early_click    = ('click_cnt', 'sum'),
              early_complete = ('complete_cnt', 'sum'),
          ).reset_index())

for day in [0, 1, 2]:
    day_agg = (sched2[sched2['elapsed_day'] == day]
               .groupby('ads_idx').agg(
                   **{f'click_day{day+1}':    ('click_cnt', 'sum'),
                      f'complete_day{day+1}': ('complete_cnt', 'sum')}
               ).reset_index())
    early3 = early3.merge(day_agg, on='ads_idx', how='left')

for col in ['click_day1','click_day2','click_day3',
            'complete_day1','complete_day2','complete_day3']:
    early3[col] = early3[col].fillna(0)
early3['click_trend'] = (early3['click_day3'] - early3['click_day1']) / (early3['click_day1'] + 1)

# ── 9. 마트 빌드 + OHE/TF-IDF fit ──
print("\n[9] 마트 빌드 + OHE/TF-IDF fit (train 기준)...")

# Build mart for train
def build_mart(df_split):
    mart = df_split[['ads_idx','analysis_ads_type_label','reward_band']].copy()
    mart = mart.merge(attr, on='ads_idx', how='left')
    mart = mart.merge(master_feat[['ads_idx','reg_hour','reg_weekday_enc','reg_is_weekend',
                                    'reg_hour_band','ads_os_type','ads_require_adid',
                                    'action_target_cnt','mentioned_media_cnt','target_media_cnt']],
                       on='ads_idx', how='left')
    mart = mart.merge(class_feat, on='ads_idx', how='left')
    early3_cols = ['ads_idx','early_click','early_complete',
                   'click_day1','click_day2','click_day3',
                   'complete_day1','complete_day2','complete_day3','click_trend']
    mart = mart.merge(early3[early3_cols], on='ads_idx', how='left')
    for col in ['early_click','early_complete','click_day1','click_day2','click_day3',
                'complete_day1','complete_day2','complete_day3','click_trend']:
        mart[col] = mart[col].fillna(0)
    mart = mart.merge(text_df[['ads_idx','text_raw']], on='ads_idx', how='left')
    mart['final_media']  = mart['final_media'].fillna('media_unknown')
    mart['final_action'] = mart['final_action'].fillna('action_unknown')
    mart['ads_os_type']  = mart['ads_os_type'].astype(str)
    return mart

mart_train = build_mart(df_train_raw)

# OHE fit on train
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(mart_train[OHE_FEATS].astype(str))
ohe_feature_names = ohe.get_feature_names_out(OHE_FEATS).tolist()
print(f"  OHE 피처: {len(ohe_feature_names)}개")

# TF-IDF fit on train
TFIDF_PARAMS = dict(analyzer='char_wb', ngram_range=(2,4),
                     max_features=50, min_df=2, sublinear_tf=True)
tfidf = TfidfVectorizer(**TFIDF_PARAMS)
tfidf.fit(mart_train['text_raw'].fillna('').tolist())
tfidf_feature_names = [f"tfidf_{n}" for n in tfidf.get_feature_names_out()]
print(f"  TF-IDF 피처: {len(tfidf_feature_names)}개")

# LabelEncoder fit on train (for ads_rejoin_type, ads_action_rule)
le_dict = {}
for col in ['ads_rejoin_type', 'ads_action_rule']:
    le = LabelEncoder()
    vals = mart_train[col].astype(str).tolist()
    if 'unknown' not in vals:
        vals.append('unknown')
    le.fit(vals)
    le_dict[col] = le
print(f"  LabelEncoder: {list(le_dict.keys())}")

# Build feature column lists
def build_X(mart, include_early=False):
    ohe_arr   = ohe.transform(mart[OHE_FEATS].astype(str))
    tfidf_arr = tfidf.transform(mart['text_raw'].fillna('').tolist()).toarray()
    parts = [
        mart[NUMERIC_FEATS].reset_index(drop=True),
        pd.DataFrame(ohe_arr,   columns=ohe_feature_names),
        pd.DataFrame(tfidf_arr, columns=tfidf_feature_names),
    ]
    if include_early:
        parts.append(mart[EARLY_FEATS].reset_index(drop=True))
    return pd.concat(parts, axis=1)

X_m1 = build_X(mart_train, include_early=False)
X_m2 = build_X(mart_train, include_early=True)
model1_feature_cols = list(X_m1.columns)
model2_feature_cols = list(X_m2.columns)
print(f"  Model1 피처: {len(model1_feature_cols)}개")
print(f"  Model2 피처: {len(model2_feature_cols)}개")

# ── 10. 저장 ──
print("\n[10] preprocessing_pipeline.joblib 저장...")
pipeline_artifacts = {
    'ohe':                 ohe,
    'tfidf':               tfidf,
    'le_dict':             le_dict,
    'ohe_feature_names':   ohe_feature_names,
    'tfidf_feature_names': tfidf_feature_names,
    'numeric_feats':       NUMERIC_FEATS,
    'ohe_feats':           OHE_FEATS,
    'early_feats':         EARLY_FEATS,
    'model1_feature_cols': model1_feature_cols,
    'model2_feature_cols': model2_feature_cols,
    'label_model1_threshold': float(THRESHOLD_TOP25),
    'global_cvr':          float(GLOBAL_CVR),
    'ctit_median':         float(ctit_median),
    'version':             'v5',
}
out_path = OUT + 'preprocessing_pipeline.joblib'
joblib.dump(pipeline_artifacts, out_path)
print(f"  저장 완료: {out_path}")
print(f"  포함 객체: {list(pipeline_artifacts.keys())}")
print("\nDONE")
