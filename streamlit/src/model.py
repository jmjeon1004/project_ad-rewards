"""
ML 모델 로딩, 피처 엔지니어링, 추론.

Model 1: 광고 품질 등급 (XGBoost, 50 피처) → S/A/B/C/D 등급
Model 2: 광고 조기부진 예측 (LightGBM, 101 피처) → decline_risk/normal/rule_based_review
"""
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL1_ARTIFACTS, MODEL2_ARTIFACTS, PIPELINE_PATH


# =====================================================================
# 1) 전처리 유틸 — models/app.py에서 이식 (학습 때와 동일해야 함)
# =====================================================================

def clean_col_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 영문+숫자+_로 정규화. 중복 시 suffix 추가."""
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
    """unseen 카테고리를 'unknown'으로 매핑 후 인코딩."""
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
    """학습 피처 순서·구성과 정확히 맞춤. 누락 컬럼은 0, 불필요 컬럼은 제거."""
    df = df.copy()
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0
    return df[feature_list]


# =====================================================================
# 2) 아티팩트 로딩 (앱 수명 동안 1회)
# =====================================================================

@st.cache_resource
def load_model1_artifacts() -> dict:
    d = MODEL1_ARTIFACTS
    model   = joblib.load(os.path.join(d, 'model.joblib'))
    le_dict = joblib.load(os.path.join(d, 'label_encoders.joblib'))
    with open(os.path.join(d, 'feature_list.json'), encoding='utf-8') as f:
        feats = json.load(f)
    ref_proba = np.load(os.path.join(d, 'ref_proba_sorted.npy'))
    with open(os.path.join(d, 'grade_info.json'), encoding='utf-8') as f:
        grade_info = json.load(f)
    with open(os.path.join(d, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return {
        'model': model, 'le_dict': le_dict, 'features': feats,
        'ref_proba_sorted': ref_proba, 'grade_info': grade_info, 'meta': meta,
    }


@st.cache_resource
def load_model2_artifacts() -> dict:
    d = MODEL2_ARTIFACTS
    model   = joblib.load(os.path.join(d, 'model.joblib'))
    le_dict = joblib.load(os.path.join(d, 'label_encoders.joblib'))
    with open(os.path.join(d, 'feature_list.json'), encoding='utf-8') as f:
        feats = json.load(f)
    with open(os.path.join(d, 'threshold_info.json'), encoding='utf-8') as f:
        thr = json.load(f)
    with open(os.path.join(d, 'rule_info.json'), encoding='utf-8') as f:
        rule = json.load(f)
    with open(os.path.join(d, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return {
        'model': model, 'le_dict': le_dict, 'features': feats,
        'threshold_info': thr, 'rule_info': rule, 'meta': meta,
    }


@st.cache_resource
def load_pipeline() -> dict:
    return joblib.load(PIPELINE_PATH)


# =====================================================================
# 3) 피처 엔지니어링 — raw parquet → 모델 입력 행렬
# =====================================================================

def _build_feature_base(
    ad_attr: pd.DataFrame,
    ad_master: pd.DataFrame,
    ad_class: pd.DataFrame,
    ad_outcome: pd.DataFrame,
) -> pd.DataFrame:
    """
    모든 유효 광고(is_valid_click10==1)에 대해
    Model 1/2 공통 기본 피처를 조립한다.
    """
    # 유효 광고 필터
    valid_ads = ad_outcome[ad_outcome['is_valid_click10'] == 1][['ads_idx']].copy()

    # attr 피처
    attr_cols = ['ads_idx', 'ads_name', 'ads_reward_price', 'ads_rejoin_type',
                 'ads_order', 'ads_action_rule', 'ads_action_diff_flag',
                 'ads_day_cap', 'analysis_ads_type_label', 'reward_band']
    attr = ad_attr[attr_cols].drop_duplicates('ads_idx', keep='last').copy()
    attr['ads_day_cap'] = attr['ads_day_cap'].astype(int)

    # master 피처
    master_cols = ['ads_idx', 'regdate', 'ads_os_type', 'ads_require_adid',
                   'action_target_cnt', 'mentioned_media_cnt', 'target_media_cnt',
                   'ads_summary']
    mf = ad_master[master_cols].drop_duplicates('ads_idx', keep='first').copy()
    mf['regdate'] = pd.to_datetime(mf['regdate'])
    mf['reg_hour']        = mf['regdate'].dt.hour
    mf['reg_weekday_enc'] = mf['regdate'].dt.dayofweek
    mf['reg_is_weekend']  = mf['reg_weekday_enc'].isin([5, 6]).astype(int)
    mf['reg_hour_band']   = mf['reg_hour'].apply(
        lambda h: 0 if h < 6 else (1 if h < 12 else (2 if h < 18 else 3))
    )
    mf['ads_require_adid'] = mf['ads_require_adid'].astype(int)
    mf['ads_os_type']      = mf['ads_os_type'].astype(str)

    # classification 피처
    cf = ad_class[['ads_idx', 'final_media', 'final_action']].drop_duplicates(
        'ads_idx', keep='last'
    ).copy()

    # JOIN
    df = valid_ads.merge(attr, on='ads_idx', how='left')
    df = df.merge(mf.drop(columns=['regdate']), on='ads_idx', how='left')
    df = df.merge(cf, on='ads_idx', how='left')

    # 결측 처리
    df['final_media']  = df['final_media'].fillna('media_unknown')
    df['final_action'] = df['final_action'].fillna('action_unknown')
    df['ads_os_type']  = df['ads_os_type'].fillna('0').astype(str)

    # 텍스트 (Model 2 TF-IDF용)
    df['text_raw'] = (df['ads_name'].fillna('') + ' ' + df['ads_summary'].fillna('')).str.strip()

    return df


def _compute_early_click(sched: pd.DataFrame) -> pd.DataFrame:
    """sched_clean에서 D+3(elapsed_day < 3) 클릭 합계를 계산."""
    s = sched.copy()
    s['click_date_dt'] = pd.to_datetime(s['click_date'])
    s['ads_sdate_dt']  = pd.to_datetime(s['ads_sdate'])
    s['elapsed_day']   = (s['click_date_dt'] - s['ads_sdate_dt']).dt.days
    s = s[s['elapsed_day'] >= 0]

    early3 = (s[s['elapsed_day'] < 3]
              .groupby('ads_idx')
              .agg(early_click=('click_cnt', 'sum'))
              .reset_index())
    return early3


# =====================================================================
# 4) 예측 함수
# =====================================================================

def predict_model1(feature_df: pd.DataFrame, art: dict, pipeline: dict) -> pd.DataFrame:
    """
    Model 1: 등록 시점 피처로 품질 등급 예측.
    반환: ads_idx, m1_proba, m1_score, m1_grade
    """
    df = feature_df.copy()
    ads_idx = df['ads_idx'].values

    # OHE 적용
    ohe = pipeline['ohe']
    ohe_feats = pipeline['ohe_feats']
    ohe_arr = ohe.transform(df[ohe_feats].astype(str))
    ohe_df = pd.DataFrame(ohe_arr, columns=pipeline['ohe_feature_names'], index=df.index)

    # 수치형 피처
    numeric_feats = pipeline['numeric_feats']

    # Label encoding (ads_rejoin_type, ads_action_rule)
    df = apply_label_encoders(df, art['le_dict'])

    # 조합
    X = pd.concat([
        df[numeric_feats].reset_index(drop=True),
        ohe_df.reset_index(drop=True),
    ], axis=1)

    # clean_col_names + align
    X = clean_col_names_unique(X)
    X = align_features(X, art['features'])

    # 예측
    proba = art['model'].predict_proba(X)[:, 1]

    # 0~100 점수 변환
    ref = art['ref_proba_sorted']
    score = np.searchsorted(ref, proba, side='right') / len(ref) * 100

    # 등급
    g = art['grade_info']
    if g['mode'] == '5grade_qcut':
        bins   = np.array(g['bins'])
        labels = g['labels']
        grade  = pd.cut(score, bins=bins, labels=labels, include_lowest=True).astype(str)
    else:
        q60, q20 = g['q60'], g['q20']
        grade = np.where(proba >= q60, 'A',
                  np.where(proba >= q20, 'B', 'C'))

    return pd.DataFrame({
        'ads_idx': ads_idx,
        'm1_proba': np.round(proba, 4),
        'm1_score': np.round(score, 1),
        'm1_grade': grade,
    })


def predict_model2(
    feature_df: pd.DataFrame,
    early_click_df: pd.DataFrame,
    art: dict,
    pipeline: dict,
    threshold_key: str = 'best_f1',
) -> pd.DataFrame:
    """
    Model 2: D+3 초기 실적으로 조기부진 예측.
    반환: ads_idx, m2_proba, m2_decision
    """
    df = feature_df.copy()
    df = df.merge(early_click_df, on='ads_idx', how='left')
    df['early_click'] = df['early_click'].fillna(0)

    ads_idx = df['ads_idx'].values
    min_click = art['rule_info']['min_early_click']
    apply_mask = df['early_click'] >= min_click

    proba_all = np.full(len(df), np.nan)

    if apply_mask.any():
        df_apply = df[apply_mask].copy()

        # OHE
        ohe = pipeline['ohe']
        ohe_feats = pipeline['ohe_feats']
        ohe_arr = ohe.transform(df_apply[ohe_feats].astype(str))
        ohe_df = pd.DataFrame(ohe_arr, columns=pipeline['ohe_feature_names'],
                               index=df_apply.index)

        # TF-IDF
        tfidf = pipeline['tfidf']
        tfidf_arr = tfidf.transform(df_apply['text_raw'].fillna('').tolist()).toarray()
        tfidf_df = pd.DataFrame(tfidf_arr, columns=pipeline['tfidf_feature_names'],
                                 index=df_apply.index)

        # Label encoding
        df_apply = apply_label_encoders(df_apply, art['le_dict'])

        # 수치형 + OHE + TF-IDF + early_click
        numeric_feats = pipeline['numeric_feats']
        X = pd.concat([
            df_apply[numeric_feats].reset_index(drop=True),
            ohe_df.reset_index(drop=True),
            tfidf_df.reset_index(drop=True),
            df_apply[['early_click']].reset_index(drop=True),
        ], axis=1)

        X = clean_col_names_unique(X)
        X = align_features(X, art['features'])

        proba_all[apply_mask.values] = art['model'].predict_proba(X)[:, 1]

    # threshold
    thr_map = {
        'default_05': art['threshold_info']['default_05'],
        'best_f1':    art['threshold_info']['best_f1'],
        **art['threshold_info'].get('recall_targets', {}),
    }
    thr = thr_map.get(threshold_key, 0.5)

    decision = np.where(
        np.isnan(proba_all), 'rule_based_review',
        np.where(proba_all >= thr, 'decline_risk', 'normal')
    )

    return pd.DataFrame({
        'ads_idx': ads_idx,
        'm2_proba': np.round(proba_all, 4),
        'm2_decision': decision,
    })


# =====================================================================
# 5) 메인 진입점 — 전체 광고 자동 스코어링
# =====================================================================

@st.cache_resource(show_spinner="ML 모델 스코어링 중...")
def score_all_ads(
    _ad_attr: pd.DataFrame,
    _ad_master: pd.DataFrame,
    _ad_class: pd.DataFrame,
    _ad_outcome: pd.DataFrame,
    _sched: pd.DataFrame,
) -> pd.DataFrame:
    """
    전체 유효 광고에 대해 Model 1 + Model 2 예측을 실행.
    반환 DataFrame: ads_idx, m1_proba, m1_score, m1_grade, m2_proba, m2_decision
    """
    # 아티팩트 로드
    m1_art = load_model1_artifacts()
    m2_art = load_model2_artifacts()
    pipeline = load_pipeline()

    # 피처 빌드
    feature_df = _build_feature_base(_ad_attr, _ad_master, _ad_class, _ad_outcome)
    early_click_df = _compute_early_click(_sched)

    # Model 1 — click_cnt >= 30 광고만 대상
    m1_valid_idx = set(_ad_outcome.loc[_ad_outcome['click_cnt'] >= 30, 'ads_idx'])
    m1_feature_df = feature_df[feature_df['ads_idx'].isin(m1_valid_idx)].copy()
    m1_result = predict_model1(m1_feature_df, m1_art, pipeline)

    # Model 2
    m2_result = predict_model2(feature_df, early_click_df, m2_art, pipeline)

    # 결합
    result = m1_result.merge(m2_result, on='ads_idx', how='outer')
    return result
