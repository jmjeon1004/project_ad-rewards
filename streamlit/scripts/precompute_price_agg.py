"""
ads_join_info_labeled.parquet (741MB)에서 price_agg를 사전 집계하여
data/price_agg.parquet로 저장하는 스크립트.

사용법: python scripts/precompute_price_agg.py
"""
import os
import sys
import pandas as pd

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DATA_DIR, ANALYSIS_DATE_START, ANALYSIS_DATE_END

_DATE_START = pd.Timestamp(ANALYSIS_DATE_START).date()
_DATE_END = pd.Timestamp(ANALYSIS_DATE_END).date()

src_path = os.path.join(DATA_DIR, "ads_join_info_labeled.parquet")
out_path = os.path.join(DATA_DIR, "price_agg.parquet")

print(f"Reading {src_path} ...")
df = pd.read_parquet(
    src_path,
    columns=["ads_idx", "mda_idx", "click_date_only", "click_hour", "media_price", "adv_price"],
    filters=[
        ("click_date_only", ">=", _DATE_START),
        ("click_date_only", "<=", _DATE_END),
    ],
)
print(f"  Loaded {len(df):,} rows")

agg = df.groupby(["ads_idx", "mda_idx", "click_date_only", "click_hour"]).agg(
    media_price=("media_price", "sum"),
    adv_price=("adv_price", "sum"),
).reset_index()
print(f"  Aggregated to {len(agg):,} rows")

agg.to_parquet(out_path, index=False)
print(f"  Saved to {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")
