"""AI-agent RAG 검색 모듈 — 지식베이스(PRD/룰/ML 메타데이터) 임베딩 검색.

인덱스(models/rag_artifacts/index.parquet)는 scripts/build_rag_index.py로
미리 계산해둔 결과를 로드만 한다. 런타임에는 사용자 질문 임베딩 1회만 호출한다.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from src.config import RAG_INDEX_PATH

EMBED_MODEL = "gemini-embedding-001"
_SIMILARITY_THRESHOLD = 0.65


@st.cache_resource
def _get_client() -> genai.Client:
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    return genai.Client(api_key=api_key)


@st.cache_resource
def _load_index() -> tuple[pd.DataFrame, np.ndarray] | None:
    if not os.path.exists(RAG_INDEX_PATH):
        return None
    df = pd.read_parquet(RAG_INDEX_PATH)
    matrix = np.vstack(df["embedding"].to_numpy())
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    return df, matrix


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """질문과 관련된 지식베이스 청크를 상위 top_k개 반환한다.

    인덱스가 없거나 임베딩 호출이 실패하면 빈 리스트를 반환해
    AI-agent의 기존 데이터 기반 응답 흐름에 영향을 주지 않는다.
    """
    index = _load_index()
    if index is None:
        return []
    df, matrix = index

    try:
        client = _get_client()
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        q_vec = np.array(resp.embeddings[0].values)
    except Exception:
        return []

    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []
    q_vec = q_vec / q_norm

    scores = matrix @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_idx:
        score = float(scores[i])
        if score < _SIMILARITY_THRESHOLD:
            continue
        row = df.iloc[i]
        results.append({
            "doc": row["doc"],
            "section": row["section"],
            "text": row["text"],
            "score": score,
        })
    return results
