"""광고 상세 분석 페이지 — wireframe v1.9 기준 리디자인"""
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

_logo_b64 = base64.b64encode((Path(__file__).parent.parent / "assets" / "logo.png").read_bytes()).decode()

# ── CSS ──
st.markdown("""
<style>
    .block-container { padding-top: 3rem; }
    [data-testid="stSidebar"] { min-width: 200px; }
    [data-testid="stSidebarNavItems"] li:last-child { display: none; }

    :root {
        --bg: #FFFFFF;
        --bg-card: #FFFFFF;
        --bg-soft: #FFFFFF;
        --bg-band: #FFFFFF;
        --ink: #2D2A26;
        --ink-2: #5B5751;
        --ink-3: #8A867F;
        --line: #E0DBC9;
        --line-soft: #EAE5D3;
        --good: #6B9C6B;
        --good-deep: #4A7A4A;
        --danger: #C25E55;
        --danger-deep: #9A4842;
        --warn-deep: #E65100;
        --accent: #4A6BAA;
        --accent-soft: #DCE6F2;
        --opp: #4A7A4A;
        --opp-soft: #DDEAD9;
        --warn: #E65100;
        --warn-soft: #FBEFD0;
        --danger-soft: #F4DCDA;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { margin-top: 0; margin-bottom: 0; }
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div { background: #FFFFFF !important; }
    div[data-testid="stExpander"] { background: #FFFFFF !important; }
    div[data-testid="stExpanderDetails"] { background: #FFFFFF !important; }
    section[data-testid="stMain"] { background: #FFFFFF !important; }
    .stMainBlockContainer { background: #FFFFFF !important; }
    div[data-testid="stMainBlockContainer"] { background: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

from src.init import ensure_data_loaded
from src.metrics import calc_all_kpis
from src.charts import (
    make_media_cvr_bars,
)
from src.config import (
    COLORS, ML_GRADE_COLORS, ML_GRADE_LABELS,
    RISK_COLORS, RISK_LABELS,
    BADGE_HERO_CONFIG, BADGE_ENTRY_LABELS, DETAIL_BACK_LABELS,
    OPPORTUNITY_BADGE_COLORS, RISK_ACTION_BADGE_COLORS,
    MEDIA_NAME_MAP,
    fmt_pct, fmt_number, fmt_currency,
)
from src.detail_data import (
    get_model_metas, get_grade_info,
    get_media_cvr_comparison,
    get_click_day_ratio,
    build_hero_content, build_why_steps, get_kpi_contexts,
)

# ── 액션 모달 ──
MODAL_MESSAGES = {
    "approval": {"icon": "✓", "title": "결재 요청 완료", "text": "결재를 요청했습니다.\n결재선을 통해 검토가 진행됩니다."},
    "landing": {"icon": "✓", "title": "랜딩 점검 요청 완료", "text": "랜딩페이지 점검을 요청했습니다.\n결과는 메시지로 전달됩니다."},
    "creative": {"icon": "✓", "title": "소재 교체 요청 완료", "text": "소재 교체를 요청했습니다.\n담당자가 진행할 예정입니다."},
    "check": {"icon": "✓", "title": "소재 사전 점검 완료", "text": "소재 사전 점검을 등록했습니다."},
    "monitor": {"icon": "✓", "title": "모니터링 등록 완료", "text": "모니터링이 등록됐습니다.\n3일 후 자동 재평가됩니다."},
    "alert": {"icon": "✓", "title": "알림 설정 완료", "text": "알림이 설정됐습니다.\n조건 충족 시 알려드립니다."},
    "pause": {"icon": "⏸", "title": "광고 일시 중단 완료", "text": "광고가 일시 중단됐습니다.\n점검 후 재개 가능합니다."},
}

@st.dialog("확인")
def show_action_modal(action_key: str):
    msg = MODAL_MESSAGES.get(action_key, MODAL_MESSAGES["alert"])
    icon_bg = "#F4DCDA" if action_key == "pause" else "#DDEAD9"
    icon_color = "#9A4842" if action_key == "pause" else "#4A7A4A"
    st.markdown(f"""
    <div style="text-align:center; padding:16px 0;">
        <div style="width:56px; height:56px; border-radius:50%; background:{icon_bg}; color:{icon_color};
                    display:flex; align-items:center; justify-content:center; font-size:28px;
                    font-weight:700; margin:0 auto 16px;">{msg['icon']}</div>
        <div style="font-size:17px; font-weight:700; color:#2D2A26; margin-bottom:8px;">{msg['title']}</div>
        <div style="font-size:13px; color:#5B5751; line-height:1.55; white-space:pre-line;">{msg['text']}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("확인", use_container_width=True, key="modal_confirm"):
        st.rerun()


# ── 액션 카드 헬퍼 ──
_BADGE_CARD_STYLES = {
    "매체확장": {"border": "#4A6BAA", "bg": "#DCE6F2", "num_bg": "#4A6BAA", "num_color": "white"},
    "승급추진": {"border": "#2E7D32", "bg": "#E8F5E9", "num_bg": "#2E7D32", "num_color": "white"},
    "즉시조치": {"border": "#C25E55", "bg": "#F4DCDA", "num_bg": "#C25E55", "num_color": "white"},
    "우선검토": {"border": "#E65100", "bg": "#FBEFD0", "num_bg": "#E65100", "num_color": "#2D2A26"},
}

def _render_action_header(num: str, headline: str, why: str, badge: str, is_priority: bool = True):
    """액션 카드 헤더 HTML 렌더링."""
    style = _BADGE_CARD_STYLES.get(badge, _BADGE_CARD_STYLES["매체확장"])
    num_bg, num_color = style["num_bg"], style["num_color"]
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:28px 1fr; gap:14px; align-items:center;">
        <div style="width:24px; height:24px; background:{num_bg}; border-radius:50%; display:flex;
                    align-items:center; justify-content:center; font-weight:700; color:{num_color};
                    font-size:12px;">{num}</div>
        <div>
            <div style="font-size:13.5px; font-weight:700; color:#2D2A26;">{headline}</div>
            <div style="font-size:12px; color:#5B5751; line-height:1.5;">{why}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

ensure_data_loaded()


def _clear_detail_state():
    """광고 상세 관련 세션 상태를 모두 정리."""
    _ads = st.session_state.get("detail_ads_idx")
    for k in ["detail_ads_idx", "detail_badge", "detail_model",
              "detail_source", "_detail_prev_ads_idx"]:
        st.session_state.pop(k, None)
    if _ads:
        for _k in list(st.session_state.keys()):
            if _k.startswith(f"_detail_precomputed_{_ads}") or _k.startswith(f"_media_cvr_{_ads}"):
                del st.session_state[_k]


# ═══════════════════════════════════════════════════
# 데이터 로딩 & 세션 상태
# ═══════════════════════════════════════════════════
ads_idx = st.session_state.get("detail_ads_idx")
if not ads_idx:
    st.error("광고를 선택해주세요. ML 인사이트의 상세 버튼을 통해 진입하세요.")
    if st.button("← 전체 광고 현황"):
        st.switch_page("pages/01_전체_광고_현황_오버뷰.py")
    st.stop()

badge = st.session_state.get("detail_badge", "매체확장")
model_type = st.session_state.get("detail_model", "m1")
source = st.session_state.get("detail_source", "pages/01_전체_광고_현황_오버뷰.py")

base = st.session_state["base"]
ad_summary = st.session_state["ad_summary"]
model_scores = st.session_state["model_scores"]
sched = st.session_state["sched"]
ad_master = st.session_state["ad_master"]
today = st.session_state["today"]

# ── 광고별 데이터 사전 필터링 (1회만, 이후 재사용) ──
# 다른 광고의 이전 캐시 정리
_prev_detail_ads = st.session_state.get("_detail_prev_ads_idx")
if _prev_detail_ads is not None and _prev_detail_ads != ads_idx:
    for _k in list(st.session_state.keys()):
        if _k.startswith(f"_detail_precomputed_{_prev_detail_ads}") or _k.startswith(f"_media_cvr_{_prev_detail_ads}"):
            del st.session_state[_k]
    # 유형별 평균 캐시는 유지 (광고 무관)
st.session_state["_detail_prev_ads_idx"] = ads_idx

_detail_cache_key = f"_detail_precomputed_{ads_idx}"
if st.session_state.get(_detail_cache_key) is None:
    _ad_base = base[base["ads_idx"] == ads_idx]
    _ad_scores_df = model_scores[model_scores["ads_idx"] == ads_idx]
    _ad_sched = sched[sched["ads_idx"] == ads_idx]
    _ad_master_row = ad_master[ad_master["ads_idx"] == ads_idx]
    st.session_state[_detail_cache_key] = {
        "ad_base": _ad_base,
        "ad_scores_df": _ad_scores_df,
        "ad_sched": _ad_sched,
        "ad_master_row": _ad_master_row,
    }

_precomputed = st.session_state[_detail_cache_key]
ad_base = _precomputed["ad_base"]
ad_scores_df = _precomputed["ad_scores_df"]

if ad_base.empty:
    st.error(f"광고 #{ads_idx}의 성과 데이터가 없습니다.")
    if st.button("← 돌아가기"):
        _clear_detail_state()
        st.switch_page(source)
    st.stop()

info = ad_base.iloc[0]
ads_name = info.get("ads_name", f"광고 #{ads_idx}")
ad_type = info.get("analysis_ads_type_label", "-")
ad_category = info.get("category_name", "-")
current_media = info.get("final_media", "-")
current_media_kr = MEDIA_NAME_MAP.get(current_media, current_media)

# 모델 스코어
sc = ad_scores_df.iloc[0] if not ad_scores_df.empty else None

# 모델 메타, KPI, 운영 일수
m1_meta, m2_meta, threshold_info = get_model_metas()
grade_info = get_grade_info()
kpis = calc_all_kpis(ad_base)
ad_sched = _precomputed["ad_sched"]
d_plus = int(ad_sched["campaign_n_day"].max()) if not ad_sched.empty and not ad_sched["campaign_n_day"].isna().all() else None
d_plus_str = f"운영 {d_plus}일 차" if d_plus is not None else ""

# 승급추진 전용 파생값
s_cut = grade_info["bins"][4]  # S등급 cutpoint (80.7)
m1_score_val = float(sc.get("m1_score", 0)) if sc is not None else 0
grade_gap = max(0, s_cut - m1_score_val)
# a_grade_count 캐싱 (전체 model_scores 스캔 비용 절감)
if "_a_grade_count" not in st.session_state:
    st.session_state["_a_grade_count"] = int((model_scores["m1_grade"] == "A").sum()) if "m1_grade" in model_scores.columns else 0
a_grade_count = st.session_state["_a_grade_count"]

# 배지 config
hero_cfg = BADGE_HERO_CONFIG.get(badge, BADGE_HERO_CONFIG["매체확장"])
entry_label = BADGE_ENTRY_LABELS.get(badge, badge)
badge_color = (OPPORTUNITY_BADGE_COLORS.get(badge)
               or RISK_ACTION_BADGE_COLORS.get(badge, "#888"))

# ═══════════════════════════════════════════════════
# Section 1: 네비게이션 바
# ═══════════════════════════════════════════════════
nav_col1, nav_col2 = st.columns([3, 2])
with nav_col1:
    back_label = DETAIL_BACK_LABELS.get(model_type, "← 돌아가기")
    if st.button(back_label):
        # 돌아갈 페이지의 ML 탭을 m1→grade, m2→risk로 설정
        _file_to_page_key = {v: k for k, v in {
            "p1": "pages/01_전체_광고_현황_오버뷰.py",
            "p2": "pages/02_유형별_운영_탐색.py",
            "p3": "pages/03_매체별_운영_탐색.py",
            "p4": "pages/04_카테고리별_운영_탐색.py",
        }.items()}
        _src_page_key = _file_to_page_key.get(source)
        if _src_page_key:
            st.session_state[f"ml_tab_{_src_page_key}"] = "grade" if model_type == "m1" else "risk"
        _clear_detail_state()
        st.switch_page(source)
with nav_col2:
    if model_type == "m1":
        meta = m1_meta
        meta_text = f"M1 v1.0 · {meta.get('algorithm', 'XGBClassifier')} · AUC {meta.get('test_auc', 0):.3f}"
    else:
        meta = m2_meta
        meta_text = f"M2 v1.0 · {meta.get('algorithm', 'LightGBM')} · PR-AUC {meta.get('test_prauc', 0):.3f}"
    st.markdown(f"<div style='text-align:right; font-size:12px; color:#8A867F; padding:8px 0;'>{meta_text}</div>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# Section 2+3: 광고 헤더 + 히어로 카드 (단일 HTML 렌더링)
# ═══════════════════════════════════════════════════
reward = info.get("ads_reward_price", 0)
reward_str = f"보상 {int(reward):,}원" if pd.notna(reward) and reward > 0 else ""
rejoin = info.get("ads_rejoin_type", "")
rejoin_str = str(rejoin).upper() if pd.notna(rejoin) and rejoin else ""

# 매체 수 (매체확장용)
master_row = _precomputed["ad_master_row"]
media_count = None
if not master_row.empty and pd.notna(master_row.iloc[0].get("mentioned_media_cnt")):
    media_count = int(master_row.iloc[0]["mentioned_media_cnt"])

hero = build_hero_content(badge, kpis, sc, info, media_count, grade_info=grade_info)
tone = hero_cfg["tone"]
tone_soft = hero_cfg["tone_soft"]
icon = hero_cfg["icon"]
hero_border = "#E65100" if badge == "우선검토" else tone

# 광고 헤더 + 히어로를 하나의 st.html로 통합 (Streamlit 엘리먼트 수 절감)
_tag_style = "font-size:12px; padding:4px 12px; background:#fff; border:1px solid #E0DBC9; border-radius:100px; color:#5B5751;"
_extra_tags = ""
if reward_str:
    _extra_tags += f"<span style='{_tag_style}'>{reward_str}</span>"
if rejoin_str:
    _extra_tags += f"<span style='{_tag_style}'>{rejoin_str}</span>"

st.html(f"""
<div style="margin-bottom:6px;">
    <span style="font-size:12px; color:#8A867F;">광고 ID #{ads_idx} · {d_plus_str}</span>
    <span style="display:inline-block; margin-left:8px; padding:2px 10px; border-radius:100px;
                 font-size:11px; font-weight:700; background:{hero_cfg['tone_soft']};
                 color:{'#E65100' if badge == '우선검토' else hero_cfg['tone']};">{entry_label}</span>
</div>
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
    <h1 style="font-size:24px; font-weight:700; letter-spacing:-0.02em; margin:0; color:#2D2A26;">
        {ads_name}
    </h1>
    <img src="data:image/png;base64,{_logo_b64}" style="height:32px; object-fit:contain;" />
</div>
<div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:20px;">
    <span style="{_tag_style}">{ad_type}</span>
    <span style="{_tag_style}">{current_media_kr}</span>
    <span style="{_tag_style}">{ad_category}</span>
    {_extra_tags}
</div>
<div style="background:#fff; border-radius:14px; padding:22px 26px; margin-bottom:18px;
            border:2px solid {hero_border};">
    <div style="display:grid; grid-template-columns:56px 1fr; gap:22px; align-items:center;">
        <div style="width:56px; height:56px; border-radius:50%; display:flex; align-items:center;
                    justify-content:center; font-size:22px; font-weight:700;
                    background:{tone_soft}; color:{tone};">
            {icon}
        </div>
        <div>
            <div style="font-size:11.5px; color:#8A867F; margin-bottom:4px; text-transform:uppercase;
                        letter-spacing:0.06em; font-weight:600;">{hero['label']}</div>
            <div style="font-size:20px; font-weight:700; line-height:1.3; margin-bottom:6px;
                        letter-spacing:-0.01em; color:#2D2A26;">
                {hero['headline']}
            </div>
            <p style="font-size:13.5px; color:#5B5751; line-height:1.55; margin:0;">
                {hero['explain']}
            </p>
        </div>
    </div>
</div>
""")

# ═══════════════════════════════════════════════════
# Section 4: 모델 카드
# ═══════════════════════════════════════════════════
if sc is not None:
    if model_type == "m1":
        grade = sc.get("m1_grade", "-")
        m1_score = sc.get("m1_score", 0)
        grade_color = ML_GRADE_COLORS.get(grade, "#757575")
        grade_label = ML_GRADE_LABELS.get(grade, "")
        accent = "#2E7D32" if badge == "승급추진" else "#4A6BAA"

        st.html(f"""
        <div style="background:#fff; border:1px solid #E0DBC9; border-radius:12px;
                    padding:16px 20px; margin-bottom:18px; border-left:4px solid {accent};">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="width:24px; height:24px; border-radius:5px; background:{accent};
                             color:white; font-weight:700; font-size:11px; display:flex;
                             align-items:center; justify-content:center; font-family:monospace;">M1</span>
                <span style="font-size:13px; font-weight:700; color:#2D2A26;">품질 등급 예측</span>
                <span style="margin-left:auto; font-size:11px; color:#8A867F;">
                    {m1_meta.get('algorithm', 'XGBClassifier')} · AUC {m1_meta.get('test_auc', 0):.3f}
                </span>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="width:56px; height:56px; border-radius:8px; background:{grade_color};
                            color:white; font-weight:700; font-size:28px; display:flex;
                            align-items:center; justify-content:center; flex-shrink:0;">
                    {grade}
                </div>
                <div>
                    <div style="font-size:18px; font-weight:700; color:#2D2A26;">
                        {f"{grade} 등급 상위 (S 진입 임박)" if badge == "승급추진" else f"{grade} 등급 · {grade_label}"}
                    </div>
                    <div style="font-size:12.5px; color:#5B5751; line-height:1.5;">
                        점수 <strong>{m1_score:.1f}점</strong>{f" · S cutpoint <strong>{s_cut:.0f}점</strong>까지 {grade_gap:.1f}점 차이 · A 등급 {a_grade_count:,}건 중 상위" if badge == "승급추진" else ""}
                    </div>
                </div>
            </div>
        </div>
        """)
    else:
        proba = sc.get("m2_proba", 0)
        decision = sc.get("m2_decision", "-")
        risk_label = RISK_LABELS.get(decision, decision or "-")
        accent = "#E65100" if badge == "우선검토" else "#C25E55"
        proba_val = float(proba) if pd.notna(proba) else 0
        circle_bg = "#C25E55" if proba_val >= 0.7 else "#E65100"
        circle_txt_color = "white" if proba_val >= 0.7 else "#2D2A26"
        headline_color = "#9A4842" if proba_val >= 0.7 else "#E65100"
        thr_best_f1 = threshold_info.get("best_f1", 0.5)
        thr_diff = proba_val - thr_best_f1

        st.html(f"""
        <div style="background:#fff; border:1px solid #E0DBC9; border-radius:12px;
                    padding:16px 20px; margin-bottom:18px; border-left:4px solid {accent};">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="width:24px; height:24px; border-radius:5px; background:{accent};
                             color:white; font-weight:700; font-size:11px; display:flex;
                             align-items:center; justify-content:center; font-family:monospace;">M2</span>
                <span style="font-size:13px; font-weight:700; color:#2D2A26;">조기부진 예측</span>
                <span style="margin-left:auto; font-size:11px; color:#8A867F;">
                    {m2_meta.get('algorithm', 'LightGBM')} · PR-AUC {m2_meta.get('test_prauc', 0):.3f}
                </span>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="width:56px; height:56px; border-radius:50%; background:{circle_bg};
                            color:{circle_txt_color}; display:flex; flex-direction:column;
                            align-items:center; justify-content:center; font-weight:700; flex-shrink:0;">
                    <span style="font-size:18px; line-height:1;">{proba_val:.2f}</span>
                    <span style="font-size:8.5px; opacity:0.9;">{'위험' if proba_val >= 0.7 else '우선 검토'}</span>
                </div>
                <div>
                    <div style="font-size:18px; font-weight:700; color:{headline_color};">
                        {risk_label} ({decision})
                    </div>
                    <div style="font-size:12.5px; color:#5B5751; line-height:1.5;">
                        부진 확률 <strong>{proba_val:.2f}</strong> · threshold <strong>{thr_best_f1:.2f}</strong>
                        {'초과' if thr_diff > 0 else '미만'} (best_f1 {'+' if thr_diff >= 0 else ''}{thr_diff:.2f})
                    </div>
                </div>
            </div>
        </div>
        """)

# ═══════════════════════════════════════════════════
# Section 5: KPI 행
# ═══════════════════════════════════════════════════

# 데이터 부족 안내 배너
_missing_metrics = []
if kpis["margin_rate"] is None:
    _missing_metrics.append("마진율")
if _missing_metrics:
    _missing_str = ", ".join(_missing_metrics)
    st.html(f"""
    <div style="display:flex; align-items:center; gap:8px; padding:10px 14px;
                background:#FFF8E1; border:1px solid #FFE082; border-radius:8px;
                margin-bottom:12px;">
        <span style="font-size:15px;">⚠️</span>
        <span style="font-size:12.5px; color:#5B5751;">
            이 광고는 <b>{_missing_str}</b> 지표를 계산할 수 없습니다.
            광고비 데이터가 누락되어 있습니다.
        </span>
    </div>
    """)

kpi_contexts = get_kpi_contexts(kpis, base, ads_idx, ad_type)

if model_type == "m1":
    kpi_items = [
        ("클릭수", fmt_number(kpis["clk"]), kpi_contexts.get("clk", ("", ""))),
        ("완료수", fmt_number(kpis["turn"]), kpi_contexts.get("turn", ("", ""))),
        ("CVR", fmt_pct(kpis["cvr"]) if kpis["cvr"] is not None else "-", kpi_contexts.get("cvr", ("", ""))),
        ("CPA", fmt_currency(kpis["cpa"]) if kpis["cpa"] is not None else "-", kpi_contexts.get("cpa", ("", ""))),
        ("마진율", fmt_pct(kpis["margin_rate"]) if kpis["margin_rate"] is not None else "-", kpi_contexts.get("margin_rate", ("", ""))),
    ]
else:
    # M2 KPI: early_click, 완료수, CVR, CPA, 광고비
    early_sched = ad_sched[ad_sched["campaign_n_day"] < 3]
    early_click = int(early_sched["click_cnt"].sum()) if not early_sched.empty else 0
    early_ctx = ("≥ 10 PASS", "good") if early_click >= 10 else ("&lt; 10", "bad")
    turn_ctx = ("", "")
    if kpis["turn"] == 0:
        turn_ctx = ("완료 없음", "bad")

    kpi_items = [
        ("early_click", fmt_number(early_click), early_ctx),
        ("완료수", fmt_number(kpis["turn"]), turn_ctx),
        ("CVR", fmt_pct(kpis["cvr"]) if kpis["cvr"] is not None else "-", kpi_contexts.get("cvr", ("", ""))),
        ("CPA", fmt_currency(kpis["cpa"]) if kpis["cpa"] is not None else "-", kpi_contexts.get("cpa", ("", ""))),
        ("광고비", fmt_currency(kpis["acost"]), ("", "")),
    ]

# KPI 행 렌더링
kpi_html_items = []
for label, value, (ctx_text, ctx_cls) in kpi_items:
    ctx_color = "#4A7A4A" if ctx_cls == "good" else "#9A4842" if ctx_cls == "bad" else "#8A867F"
    ctx_style = f"font-weight:600;" if ctx_cls in ("good", "bad") else "font-style:italic;"
    ctx_html = f'<div style="font-size:10px; color:{ctx_color}; {ctx_style}">{ctx_text}</div>' if ctx_text else ""
    # 값에서 단위(원, %) 분리 → 단위는 작은 폰트로 렌더링
    if value.endswith("원"):
        num_part, unit_part = value[:-1], "원"
    elif value.endswith("%"):
        num_part, unit_part = value[:-1], "%"
    else:
        num_part, unit_part = value, ""
    unit_html = f'<span style="font-size:14px; font-weight:600;">{unit_part}</span>' if unit_part else ""
    kpi_html_items.append(
        f'<div style="padding:14px 16px; border-right:1px solid #EAE5D3;">'
        f'<div style="font-size:10.5px; color:#8A867F; margin-bottom:6px; font-weight:500;">{label}</div>'
        f'<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em; line-height:1;'
        f' margin-bottom:4px; color:#2D2A26;">{num_part}{unit_html}</div>'
        f'{ctx_html}'
        f'</div>'
    )

st.html(f"""
<div style="display:grid; grid-template-columns:repeat({len(kpi_items)}, 1fr); background:#fff;
            border:1px solid #E0DBC9; border-radius:12px; margin-bottom:18px; overflow:hidden;">
    {''.join(kpi_html_items)}
</div>
""")

# ═══════════════════════════════════════════════════
# Section 6: 액션 섹션
# ═══════════════════════════════════════════════════
section_title = "즉시 조치 액션" if badge == "즉시조치" else "검토 액션 (모니터링 권장)" if badge == "우선검토" else "우선 검토 액션"
st.markdown(f"""
<div style="font-size:16px; font-weight:700; margin:22px 0 10px; letter-spacing:-0.01em;
            color:#2D2A26;">
    {section_title}
</div>
""", unsafe_allow_html=True)

# ── 배지별 액션 카드 ──
card_style = _BADGE_CARD_STYLES.get(badge, _BADGE_CARD_STYLES["매체확장"])

if badge == "매체확장":
    # ── Primary: 매체 확장 검토 ──
    with st.container(border=True):
        _render_action_header(
            "1", "매체 확장 — 어떤 매체로 옮길까",
            f"CVR {fmt_pct(kpis.get('cvr'))}로 매우 우수 · 다른 매체에서도 잘 될 가능성 높음.",
            badge,
        )
        st.divider()

        media_cvr = get_media_cvr_comparison(ads_idx, base, ad_type, ad_category, current_media)
        if not media_cvr.empty:
            st.markdown(f"""
            <div style="background:#FFFFFF; border-radius:8px; padding:12px 16px; margin-bottom:12px;">
                <div style="font-size:12px; font-weight:700; color:#5B5751; margin-bottom:6px;
                            display:flex; justify-content:space-between;">
                    <span>매체별 평균 CVR 비교 — 같은 {ad_type}·{ad_category}</span>
                    <span style="font-size:10.5px; color:#8A867F; font-weight:500;">이 광고 vs 다른 매체 평균</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig = make_media_cvr_bars(media_cvr, current_media)
            st.plotly_chart(fig, use_container_width=True)

            # 확장 가능 매체 옵션 카드
            expand_options = media_cvr[~media_cvr["is_current"]].head(3)
            if not expand_options.empty:
                st.markdown("<div style='font-size:12px; font-weight:700; color:#2D2A26; margin-bottom:6px;'><strong>확장 가능 매체</strong></div>", unsafe_allow_html=True)
                opt_cols = st.columns(len(expand_options))
                for i, (_, opt) in enumerate(expand_options.iterrows()):
                    media_kr = MEDIA_NAME_MAP.get(opt["media"], opt["media"])
                    is_first = i == 0
                    border_color = "#4A6BAA" if is_first else "#E0DBC9"
                    bg_color = "#DCE6F2" if is_first else "#fff"
                    rank_bg = "#4A6BAA" if is_first else "#EFEBDF"
                    rank_color = "white" if is_first else "#5B5751"
                    with opt_cols[i]:
                        st.markdown(f"""
                        <div style="background:{bg_color}; border:1.5px solid {border_color};
                                    border-radius:8px; padding:12px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-weight:700; font-size:13px;">{media_kr}</span>
                                <span style="font-size:10px; background:{rank_bg}; color:{rank_color};
                                             padding:1px 7px; border-radius:100px; font-weight:600;">#{i+1}</span>
                            </div>
                            <div style="font-size:11px; color:#5B5751;">
                                CVR <strong>{opt['cvr']:.1f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # 버튼 행
        btn_cols = st.columns([3, 1])
        with btn_cols[1]:
            if st.button("결재 요청 →", key="btn_approval_expand", type="primary", use_container_width=True):
                show_action_modal("approval")

    # ── Secondary: 재평가 알림 ──
    with st.container(border=True):
        sec_cols = st.columns([4, 1])
        with sec_cols[0]:
            _render_action_header("2", "3일 후 재평가 알림", "매체 확장 진행 후 추적 위해 알림 설정.", badge, is_priority=False)
        with sec_cols[1]:
            if st.button("알림 설정", key="btn_alert_expand", use_container_width=True):
                show_action_modal("alert")

elif badge == "승급추진":
    # ── Primary: 사전 준비 ──
    with st.container(border=True):
        _render_action_header(
            "1", "S 진입 시 즉시 증액 가능하게 사전 준비",
            f"{grade_gap:.1f}점 차이로 S 진입 임박 — 진입 후 추가 노출 기회를 놓치지 않도록 미리 결재선 점검·확보 권장.",
            badge,
        )
        st.divider()

        # 등급 진행 바 (SVG)
        cur_grade = sc.get('m1_grade', 'A') if sc is not None else 'A'

        bins = grade_info["bins"]  # [0, 20.02, 40, 59.9, 80.7, 100]
        g_labels = grade_info["labels"]  # ["D","C","B","A","S"]
        g_colors = {"D": "#C25E55", "C": "#D88B5A", "B": "#C9A858", "A": "#6B9C6B", "S": "#4A6BAA"}

        # SVG 좌표 — viewBox 0 0 800 80, 바 영역 x=40~780 (width 740)
        bar_left, bar_right, bar_y, bar_h = 40, 780, 25, 30
        bar_w = bar_right - bar_left

        # 구간 rects
        rects_svg = ""
        labels_svg = ""
        for i, lbl in enumerate(g_labels):
            x1 = bar_left + (bins[i] / 100) * bar_w
            x2 = bar_left + (bins[i + 1] / 100) * bar_w
            w = x2 - x1
            cx = (x1 + x2) / 2
            rects_svg += f'<rect x="{x1:.1f}" y="{bar_y}" width="{w:.1f}" height="{bar_h}" fill="{g_colors[lbl]}" opacity="0.6"/>'
            labels_svg += f'<text x="{cx:.1f}" y="{bar_y + bar_h/2 + 5}" text-anchor="middle" font-size="12" fill="white" font-weight="700" font-family="Pretendard,sans-serif">{lbl}</text>'

        # 축 라벨
        display_bins = [0, 20, 40, 60, 80, 100]
        axis_svg = ""
        for i, db in enumerate(display_bins):
            x = bar_left + (bins[i] / 100) * bar_w
            color = "#4A6BAA" if db == 80 else "#8A867F"
            fw = "font-weight:700;" if db == 80 else ""
            axis_svg += f'<text x="{x:.1f}" y="72" text-anchor="middle" font-size="10" fill="{color}" style="{fw}" font-family="Pretendard,sans-serif">{db}</text>'

        # 마커 위치
        marker_x = bar_left + (m1_score_val / 100) * bar_w
        marker_color = g_colors.get(cur_grade, "#6B9C6B")

        chart_html = f"""
        <div style="background:#FFFFFF; border-radius:8px; padding:16px; font-family:Pretendard,sans-serif;">
            <div style="font-size:12px; font-weight:700; color:#5B5751; margin-bottom:10px;
                        display:flex; justify-content:space-between;">
                <span>M1 점수 위치 — 등급 분포</span>
                <span style="font-size:10.5px; color:#8A867F; font-weight:500;">현재 {m1_score_val:.1f}점 · {cur_grade} 등급 상위</span>
            </div>
            <svg width="100%" height="80" viewBox="0 0 820 80" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <clipPath id="bar-clip">
                        <rect x="{bar_left}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="6" ry="6"/>
                    </clipPath>
                </defs>
                <g clip-path="url(#bar-clip)">
                    {rects_svg}
                </g>
                {labels_svg}
                {axis_svg}
                <line x1="{marker_x:.1f}" y1="{bar_y - 6}" x2="{marker_x:.1f}" y2="{bar_y + bar_h + 1}" stroke="#2D2A26" stroke-width="2.5"/>
                <circle cx="{marker_x:.1f}" cy="{bar_y + bar_h / 2}" r="7" fill="white" stroke="{marker_color}" stroke-width="2.5"/>
                <text x="{marker_x:.1f}" y="{bar_y - 10}" text-anchor="middle" font-size="11" fill="#2D2A26" font-weight="700" font-family="Pretendard,sans-serif">현재 {m1_score_val:.1f}점</text>
            </svg>
            <div style="margin-top:8px; font-size:11px; color:#8A867F; font-style:italic;">
                ↑ {cur_grade} 등급 영역 내 상위 위치 · S 진입 cutpoint({s_cut:.0f}점)까지 {grade_gap:.1f}점 차이
            </div>
        </div>
        """
        import streamlit.components.v1 as _stc
        _stc.html(chart_html, height=170)

        # 버튼 행
        btn_cols = st.columns([2, 1])
        with btn_cols[1]:
            if st.button("알림 설정", key="btn_alert_rising", type="primary", use_container_width=True):
                show_action_modal("alert")

    # ── Secondary: S 진입 알림 ──
    with st.container(border=True):
        sec_cols = st.columns([4, 1])
        with sec_cols[0]:
            _render_action_header("2", "S 진입 시 알림 설정", "점수가 S 등급 이상 도달 시 자동 알림.", badge, is_priority=False)
        with sec_cols[1]:
            if st.button("알림 설정", key="btn_alert_rising2", use_container_width=True):
                show_action_modal("alert")

elif badge == "즉시조치":
    # 즉시조치 액션 1,2 붉은 계열 배경 스타일 (key 기반 CSS 타겟팅)
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"][data-key="danger_action_1"]) {
        background-color: #F4DCDA !important;
        border: 1.5px solid #C25E55 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"][data-key="danger_action_2"]) {
        background-color: #F4DCDA !important;
        border: 1.5px solid #C25E55 !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Primary: 소재/랜딩 점검 ──
    with st.container(key="danger_action_1", border=True):
        clk_total = kpis.get("clk", 0)
        turn_total = kpis.get("turn", 0)
        hdr_cols = st.columns([3, 1, 1])
        with hdr_cols[0]:
            _render_action_header(
                "1", "소재 또는 랜딩페이지 즉시 점검",
                f"클릭 {clk_total:,}건에 완료 {turn_total:,}건. 가장 빈번한 부진 원인은 소재·랜딩 정합성.",
                badge, is_priority=True,
            )
        with hdr_cols[1]:
            if st.button("랜딩 점검", key="btn_landing", use_container_width=True):
                show_action_modal("landing")
        with hdr_cols[2]:
            if st.button("소재 교체", key="btn_creative", type="primary", use_container_width=True):
                show_action_modal("creative")

    # ── Secondary: 일시 중단 ──
    with st.container(key="danger_action_2", border=True):
        sec_cols = st.columns([4, 1])
        with sec_cols[0]:
            _render_action_header(
                "2", "광고 일시 중단 검토",
                f"광고비 {fmt_currency(kpis.get('acost', 0))} 지출 · 추가 손실 방지.",
                badge, is_priority=True,
            )
        with sec_cols[1]:
            if st.button("일시 중단", key="btn_pause", type="primary", use_container_width=True):
                show_action_modal("pause")

else:  # 우선검토
    # ── Primary: 재평가 알림 ──
    with st.container(border=True):
        _render_action_header(
            "1", "3일 후 재평가 알림 — 추세 추적",
            "즉시 조치 단계 아님. 다음 3일 동안 지표 변화 추적.",
            badge,
        )
        st.divider()

        # sc에서 직접 가져와 model_scores 재필터링 방지
        proba = float(sc.get("m2_proba", 0)) if sc is not None and pd.notna(sc.get("m2_proba")) else 0
        best_f1 = threshold_info.get("best_f1", 0.5)
        immediate_thr = 0.7

        # threshold gauge (HTML)
        st.markdown(f"""
        <div style="background:#FFFFFF; border-radius:8px; padding:16px; margin-bottom:12px;">
            <div style="font-size:12px; font-weight:700; color:#5B5751; margin-bottom:10px;
                        display:flex; justify-content:space-between;">
                <span>부진 확률 위치</span>
                <span style="font-size:10.5px; color:#8A867F; font-weight:500;">
                    현재 {proba:.2f} · best_f1 {best_f1:.2f} · 즉시 조치 {immediate_thr:.1f}
                </span>
            </div>
            <div style="height:22px; background:#EFEBDF; border-radius:4px; overflow:hidden; position:relative; margin:8px 0;">
                <div style="position:absolute; left:0; top:0; bottom:0; width:{best_f1*100:.0f}%; background:#DDEAD9;"></div>
                <div style="position:absolute; left:{best_f1*100:.0f}%; top:0; bottom:0; width:{(immediate_thr-best_f1)*100:.0f}%; background:#FBEFD0;"></div>
                <div style="position:absolute; left:{immediate_thr*100:.0f}%; top:0; bottom:0; right:0; background:#F4DCDA;"></div>
                <div style="position:absolute; left:{best_f1*100:.0f}%; top:-2px; bottom:-2px; width:2px; background:#2D2A26;"></div>
                <div style="position:absolute; left:{immediate_thr*100:.0f}%; top:-2px; bottom:-2px; width:2px; background:#2D2A26;"></div>
                <div style="position:absolute; left:{proba*100:.0f}%; top:50%; width:14px; height:14px;
                            border-radius:50%; transform:translate(-50%, -50%); border:2px solid white; z-index:2;
                            background:{'#C25E55' if proba >= immediate_thr else '#E65100'};"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#8A867F;">
                <span>0.0 안전</span>
                <span style="color:#4A7A4A; font-weight:700;">{best_f1:.2f} ↑ best_f1</span>
                <span style="color:#9A4842; font-weight:700;">{immediate_thr:.1f} 즉시 조치</span>
                <span>1.0 위험</span>
            </div>
            <div style="margin-top:12px; font-size:11px; color:#8A867F; font-style:italic;">
                ↑ 우선 검토 영역({best_f1:.2f}~{immediate_thr:.2f})에 위치 · 즉시 조치({immediate_thr:.1f}+)와는 {immediate_thr - proba:.2f} 차이
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 추적 지표 테이블
        st.markdown("<div style='font-size:12px; font-weight:700; color:#2D2A26; margin-bottom:6px;'><strong>추적할 핵심 지표</strong> — 2개 이상 악화 시 즉시 조치로 전환</div>", unsafe_allow_html=True)

        cvr_val = kpis.get("cvr")
        cvr_str = f"{cvr_val:.1f}%" if cvr_val is not None else "-"
        click_day_ratio = get_click_day_ratio(ads_idx, ad_sched, _pre_filtered=True)
        click_day_str = f"{click_day_ratio:.2f}" if click_day_ratio is not None else "-"

        st.markdown(f"""
        <table style="width:100%; font-size:12px; border-collapse:collapse; background:#fff; border-radius:8px;">
            <thead>
                <tr style="border-bottom:1px solid #E0DBC9;">
                    <th style="text-align:left; padding:6px 8px; color:#8A867F; font-weight:700; font-size:10.5px;">지표</th>
                    <th style="text-align:right; padding:6px 8px; color:#8A867F; font-weight:700; font-size:10.5px;">현재</th>
                    <th style="text-align:right; padding:6px 8px; color:#8A867F; font-weight:700; font-size:10.5px;">악화</th>
                    <th style="text-align:right; padding:6px 8px; color:#8A867F; font-weight:700; font-size:10.5px;">회복</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px dashed #EAE5D3;">
                    <td style="padding:7px 8px;">부진 확률</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#E65100; font-weight:700;">{proba:.2f}</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#9A4842;">≥ {immediate_thr:.2f}</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#4A7A4A;">&lt; {best_f1:.2f}</td>
                </tr>
                <tr style="border-bottom:1px dashed #EAE5D3;">
                    <td style="padding:7px 8px;">CVR</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#E65100; font-weight:700;">{cvr_str}</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#9A4842;">&lt; 8%</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#4A7A4A;">&gt; 15%</td>
                </tr>
                <tr>
                    <td style="padding:7px 8px;">click_day1 → day3</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#E65100; font-weight:700;">{click_day_str}</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#9A4842;">&lt; 0.50</td>
                    <td style="text-align:right; padding:7px 8px; font-family:monospace; color:#4A7A4A;">&gt; 0.85</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        # 버튼 행
        btn_cols = st.columns([2, 1, 1])
        with btn_cols[1]:
            if st.button("소재 사전 점검", key="btn_check_review", use_container_width=True):
                show_action_modal("check")
        with btn_cols[2]:
            if st.button("모니터링 등록 →", key="btn_monitor_review", type="primary", use_container_width=True):
                show_action_modal("monitor")

# ═══════════════════════════════════════════════════
# Section 7: "왜?" 섹션
# ═══════════════════════════════════════════════════
why_steps = build_why_steps(badge, kpis, sc, info, grade_info=grade_info)

why_title_map = {
    "매체확장": "왜 매체 확장 후보로 분류됐나?",
    "승급추진": "왜 승급 추진 광고로 분류됐나?",
    "즉시조치": "왜 부진 위험으로 봤나?",
    "우선검토": "왜 우선 검토로 분류됐나?",
}
with st.expander(why_title_map.get(badge, f"왜 {badge}(으)로 분류됐나?")):
    for step in why_steps:
        if step["caveat"]:
            num_bg = "#FBEFD0"
            num_color = "#E65100"
            title_color = "#E65100"
        else:
            num_bg = "#EFEBDF"
            num_color = "#5B5751"
            title_color = "#2D2A26"

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:24px 1fr; gap:14px; padding:12px 0;
                    border-bottom:1px dashed #EAE5D3;">
            <div style="width:22px; height:22px; background:{num_bg}; border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        font-weight:700; font-size:11px; color:{num_color};">{step['num']}</div>
            <div>
                <div style="font-size:13px; font-weight:700; margin-bottom:4px; color:{title_color};">{step['title']}</div>
                <div style="font-size:12.5px; color:#5B5751; line-height:1.6;">{step['text']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

