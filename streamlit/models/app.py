# =====================================================================
# app.py — Streamlit 배포 앱 골격
# 실행: streamlit run app.py
# =====================================================================
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# 1) 학습 노트북과 동일한 전처리 함수
#    노트북에서 쓰던 clean_col_names_unique / safe_transform을 그대로 옮긴다.
#    이 두 함수를 빠뜨리면 컬럼명·인코딩이 학습 때와 어긋난다.
# ---------------------------------------------------------------------
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
    - 누락된 컬럼은 0으로 채움 (필요 시 도메인 기본값으로 변경)
    - 학습에 없는 추가 컬럼은 버림
    - 컬럼 순서를 feature_list 순서로 강제
    """
    missing = [c for c in feature_list if c not in df.columns]
    for c in missing:
        df[c] = 0
    return df[feature_list]


# ---------------------------------------------------------------------
# 2) 산출물 로드 (st.cache_resource로 1회만)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model1(artifact_dir='model1_artifacts'):
    model    = joblib.load(f'{artifact_dir}/model.joblib')
    le_dict  = joblib.load(f'{artifact_dir}/label_encoders.joblib')
    with open(f'{artifact_dir}/feature_list.json', encoding='utf-8') as f:
        feats = json.load(f)
    ref_proba_sorted = np.load(f'{artifact_dir}/ref_proba_sorted.npy')
    with open(f'{artifact_dir}/grade_info.json', encoding='utf-8') as f:
        grade_info = json.load(f)
    with open(f'{artifact_dir}/metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    return {
        'model': model, 'le_dict': le_dict, 'features': feats,
        'ref_proba_sorted': ref_proba_sorted,
        'grade_info': grade_info, 'meta': meta,
    }


@st.cache_resource
def load_model2(artifact_dir='model2_artifacts'):
    model    = joblib.load(f'{artifact_dir}/model.joblib')
    le_dict  = joblib.load(f'{artifact_dir}/label_encoders.joblib')
    with open(f'{artifact_dir}/feature_list.json', encoding='utf-8') as f:
        feats = json.load(f)
    with open(f'{artifact_dir}/threshold_info.json', encoding='utf-8') as f:
        thr = json.load(f)
    with open(f'{artifact_dir}/rule_info.json', encoding='utf-8') as f:
        rule = json.load(f)
    with open(f'{artifact_dir}/metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    return {
        'model': model, 'le_dict': le_dict, 'features': feats,
        'threshold_info': thr, 'rule_info': rule, 'meta': meta,
    }


# ---------------------------------------------------------------------
# 3) 예측 함수
# ---------------------------------------------------------------------
def predict_model1(input_df: pd.DataFrame, art: dict):
    df = clean_col_names_unique(input_df)
    df = apply_label_encoders(df, art['le_dict'])
    X  = align_features(df, art['features'])

    proba = art['model'].predict_proba(X)[:, 1]

    # train+val 분포에 매핑해 0~100 점수
    ref = art['ref_proba_sorted']
    score = np.searchsorted(ref, proba, side='right') / len(ref) * 100

    # 등급
    g = art['grade_info']
    if g['mode'] == '5grade_qcut':
        bins   = np.array(g['bins'])
        labels = g['labels']
        grade  = pd.cut(score, bins=bins, labels=labels, include_lowest=True)
    else:
        q60, q20 = g['q60'], g['q20']
        grade = np.where(proba >= q60, 'A',
                  np.where(proba >= q20, 'B', 'C'))

    return pd.DataFrame({
        'proba': proba, 'score': np.round(score, 1), 'grade': grade,
    })


def predict_model2(input_df: pd.DataFrame, art: dict, threshold_key='best_f1'):
    df = clean_col_names_unique(input_df)

    # early_click 룰 적용
    min_click = art['rule_info']['min_early_click']
    apply_mask = df['early_click'] >= min_click

    df_apply = apply_label_encoders(df[apply_mask], art['le_dict'])
    X        = align_features(df_apply, art['features'])

    proba = np.full(len(df), np.nan)
    proba[apply_mask.values] = art['model'].predict_proba(X)[:, 1]

    # threshold 선택
    thr_map = {
        'default_05': art['threshold_info']['default_05'],
        'best_f1':    art['threshold_info']['best_f1'],
        **art['threshold_info']['recall_targets'],
    }
    thr = thr_map[threshold_key]

    pred = np.where(np.isnan(proba), 'rule_based_review',
              np.where(proba >= thr, 'decline_risk', 'normal'))

    return pd.DataFrame({
        'proba': proba, 'threshold': thr, 'decision': pred,
    })


# ---------------------------------------------------------------------
# 4) UI
# ---------------------------------------------------------------------
st.set_page_config(page_title='광고 품질 예측', layout='wide')
st.title('광고 품질 예측 시스템')

m1 = load_model1()
m2 = load_model2()

with st.sidebar:
    st.subheader('모델 정보')
    st.caption(f"Model 1: {m1['meta']['algorithm']} | test AUC {m1['meta']['test_auc']:.4f}")
    st.caption(f"Model 2: {m2['meta']['algorithm']} | test PR-AUC {m2['meta']['test_prauc']:.4f}")

tab1, tab2 = st.tabs(['Model 1 — 품질 등급', 'Model 2 — 조기부진'])

with tab1:
    uploaded = st.file_uploader('등록 시점 데이터 (parquet/csv)', type=['parquet', 'csv'], key='m1')
    if uploaded:
        df = pd.read_parquet(uploaded) if uploaded.name.endswith('.parquet') else pd.read_csv(uploaded)
        result = predict_model1(df, m1)
        st.dataframe(result.head(50))
        st.download_button('결과 CSV 다운로드', result.to_csv(index=False).encode('utf-8-sig'),
                            file_name='model1_result.csv')

with tab2:
    threshold_key = st.selectbox(
        'Threshold 선택',
        options=['best_f1', 'default_05', 'Recall >= 0.70', 'Recall >= 0.80'],
    )
    uploaded = st.file_uploader('D+3 시점 데이터 (parquet/csv)', type=['parquet', 'csv'], key='m2')
    if uploaded:
        df = pd.read_parquet(uploaded) if uploaded.name.endswith('.parquet') else pd.read_csv(uploaded)
        result = predict_model2(df, m2, threshold_key=threshold_key)
        st.dataframe(result.head(50))
        st.download_button('결과 CSV 다운로드', result.to_csv(index=False).encode('utf-8-sig'),
                            file_name='model2_result.csv')
