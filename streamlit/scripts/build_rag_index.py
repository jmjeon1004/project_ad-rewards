"""
data/rag_docs/*.md를 ## 헤딩 단위로 청크 분할하고, Gemini Embedding API로
임베딩을 계산해 models/rag_artifacts/index.parquet에 저장하는 스크립트.

문서가 바뀔 때마다 수동으로 재실행한다 (배포 파이프라인에는 포함하지 않음).

사용법: python scripts/build_rag_index.py
"""
import glob
import io
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAG_ARTIFACTS, RAG_DOCS_DIR, RAG_INDEX_PATH

load_dotenv()

from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"


def _chunk_doc(path: str) -> list[dict]:
    """## 헤딩 단위로 문서를 청크 분할한다. 첫 # 타이틀은 모든 청크의 prefix로 포함."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    doc_name = os.path.basename(path)
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1).strip() if title_match else doc_name

    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        heading_match = re.match(r"##\s+(.+)", section)
        section_name = heading_match.group(1).strip() if heading_match else doc_name
        chunks.append({
            "doc": doc_name,
            "section": section_name,
            "text": f"# {title}\n\n{section}",
        })
    return chunks


def main():
    paths = sorted(glob.glob(os.path.join(RAG_DOCS_DIR, "*.md")))
    if not paths:
        print(f"문서가 없습니다: {RAG_DOCS_DIR}")
        return

    chunks: list[dict] = []
    for path in paths:
        chunks.extend(_chunk_doc(path))
    print(f"{len(paths)}개 문서 → {len(chunks)}개 청크")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY가 설정되지 않았습니다 (.env 확인)")
    client = genai.Client(api_key=api_key)

    embeddings = []
    for chunk in chunks:
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=chunk["text"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings.append(resp.embeddings[0].values)
        print(f"  임베딩 완료: {chunk['doc']} › {chunk['section']}")

    df = pd.DataFrame(chunks)
    df["embedding"] = embeddings

    os.makedirs(RAG_ARTIFACTS, exist_ok=True)
    df.to_parquet(RAG_INDEX_PATH, index=False)
    print(f"저장 완료: {RAG_INDEX_PATH} (청크 {len(df)}개, 차원 {len(embeddings[0])})")


if __name__ == "__main__":
    main()
