"""공통 설정: 경로, 임계값, 가중치, 색상 등"""
import os

# ── 경로 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── 품질 점수 가중치 ──
QUALITY_WEIGHTS = {
    "profitability": 0.35,
    "conversion_quality": 0.30,
    "scale": 0.20,
    "speed": 0.15,
}

# ── 예측 등급 임계값 (기존 룰 기반) ──
GRADE_THRESHOLDS = {
    "A": 0.55,   # >= 0.55 → A (집행추천)
    "C": 0.15,   # < 0.15 → C (비권장)
}

# ── 모델 아티팩트 경로 ──
MODEL1_ARTIFACTS = os.path.join(MODELS_DIR, "model1_artifacts")
MODEL2_ARTIFACTS = os.path.join(MODELS_DIR, "model2_artifacts")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")

# ── AI-agent RAG 지식베이스 경로 ──
RAG_DOCS_DIR = os.path.join(DATA_DIR, "rag_docs")
RAG_ARTIFACTS = os.path.join(MODELS_DIR, "rag_artifacts")
RAG_INDEX_PATH = os.path.join(RAG_ARTIFACTS, "index.parquet")

# ── ML 모델 등급 색상 (S/A/B/C/D) ──
ML_GRADE_COLORS = {
    "S": "#1B5E20",  # 짙은 초록
    "A": "#2E7D32",  # 초록
    "B": "#F9A825",  # 노랑
    "C": "#E65100",  # 주황
    "D": "#C62828",  # 빨강
}

# ── 조기부진 위험도 색상 ──
RISK_COLORS = {
    "decline_risk": "#C62828",       # 빨강
    "normal": "#2E7D32",             # 초록
    "rule_based_review": "#9E9E9E",  # 회색
}

# ── ML 등급 라벨 ──
ML_GRADE_LABELS = {
    "S": "최우수",
    "A": "우수",
    "B": "보통",
    "C": "주의",
    "D": "개선필요",
}

RISK_LABELS = {
    "decline_risk": "부진위험",
    "normal": "정상",
    "rule_based_review": "판단보류",
}

# ── ML 인사이트 UI 상수 ──
ML_INSIGHT_BG = "#FFFFFF"
ML_INSIGHT_HEADER_BG = "#2C2C2C"

OPPORTUNITY_BADGE_COLORS = {
    "매체확장": "#1565C0",
    "승급추진": "#2E7D32",
}

RISK_ACTION_BADGE_COLORS = {
    "즉시조치": "#C62828",
    "우선검토": "#E65100",
}

# ── CTIT 소스 ──
CTIT_SOURCE = "ad_outcome"  # or "main_funnel"

# ── 분석 기간 (hourly_report 필터링용) ──
ANALYSIS_DATE_START = "2025-07-26"
ANALYSIS_DATE_END = "2025-08-25"

# ── 기본 기간 ──
DEFAULT_PERIOD = "전체"
PERIOD_OPTIONS = ["최근 1일", "최근 7일", "전체", "사용자 지정"]

# ── 운영 상태 임계값 ──
NORMAL_RATIO_GREEN = 80
NORMAL_RATIO_YELLOW = 60
DEPENDENCY_RED = 70      # TOP 10 의존도 ≥ 70% → 빨강 (분석 기간 일별 75%ile=76.5% 기반)
DEPENDENCY_YELLOW = 55   # TOP 10 의존도 ≥ 55% → 노랑 (분석 기간 일별 중앙값=60.4%, 25%ile=50.7% 기반)

# ── 최소 클릭수 필터 ──
MIN_CLICK_FILTER = 30

# ── final_action → 광고 유형 매핑 ──
ACTION_TYPE_MAP = {
    "action_participation": "참여형",
    "action_run": "실행형",
    "action_click": "클릭형",
    "action_purchase": "구매형",
    "action_view": "노출형",
    "action_install": "설치형",
    "action_signup": "가입형",
    "action_exposure": "노출형",
}

# ── final_media → 한글 매체명 매핑 ──
MEDIA_NAME_MAP = {
    "media_naver": "네이버",
    "media_app": "앱",
    "media_unknown": "기타",
    "media_web": "웹",
    "media_kakao": "카카오",
    "media_youtube": "유튜브",
    "media_instagram": "인스타그램",
    "media_facebook": "페이스북",
    "media_tiktok": "틱톡",
    "media_twitter_x": "트위터/X",
}

# ── 유형별 운영 흐름 대시보드: analysis_ads_type_label → 대분류 매핑 ──
# analysis_ads_type_label은 final_action 기반 파생 컬럼(ACTION_TYPE_MAP 참고)으로,
# 참여형·실행형·클릭형·구매형·노출형·설치형·가입형·기타 값만 갖는다.
TYPE_CATEGORY_MAP = {
    "설치형": "install", "가입형": "install",
    "클릭형": "click", "실행형": "click",
    "참여형": "engage", "기타": "engage",
    "구매형": "purchase", "노출형": "purchase",
}

# ── 대분류별 라벨 + 산점도 축 지표 ──
TYPE_CATEGORY_GROUPS = {
    "install": {
        "label": "설치형·가입형", "x_metric": "cvr", "y_metric": "cpa",
        "x_label": "CVR (%)", "y_label": "CPA (원)", "y_kpi_label": "평균 CPA",
    },
    "click": {
        "label": "클릭형·실행형", "x_metric": "turn", "y_metric": "cpa",
        "x_label": "완료수", "y_label": "CPA (원)", "y_kpi_label": "평균 CPA",
    },
    "engage": {
        "label": "참여형·기타", "x_metric": "turn", "y_metric": "margin_rate",
        "x_label": "완료수", "y_label": "마진율 (%)", "y_kpi_label": "평균 마진율",
    },
    "purchase": {
        "label": "구매형·노출형", "x_metric": "margin_rate", "y_metric": "cpa",
        "x_label": "마진율 (%)", "y_label": "CPA (원)", "y_kpi_label": "평균 CPA",
    },
}

# ── 지표가 낮을수록 좋은지 여부 (유형별 운영 흐름 사분면 분류용) ──
FLOW_LOWER_IS_BETTER = {"cpa": True, "cvr": False, "turn": False, "margin_rate": False}

# ── 유형별 운영 흐름 사분면 액션 색상/라벨 ──
FLOW_ACTION_COLORS = {
    "urgent": "#e34948",
    "stable": "#2a78d6",
    "expand": "#1baf7a",
    "hold": "#898781",
}
FLOW_ACTION_LABELS = {
    "urgent": "즉시조치",
    "stable": "안정운영",
    "expand": "확장후보",
    "hold": "보류",
}

# ── 색상 팔레트 ──
COLORS = {
    # 브랜드 기본 색상
    "primary": "#4A7C59",       # 진한 초록
    "primary_light": "#8FBC8F", # 연한 초록
    "secondary": "#FFFFFF",     # 카드/박스 배경
    "legend_bg": "#F5F3ED",     # KPI 범례 베이지 배경 (등급 기준 띠와 동일)
    "background": "#FFFFFF",    # 밝은 배경
    "card_bg": "#FFFFFF",       # 카드 배경
    "border": "#E8E4DC",        # 테두리

    # 상태 색상
    "positive": "#2E7D32",      # 양호/상승
    "negative": "#C62828",      # 위험/하락
    "warning": "#F9A825",       # 경고
    "info": "#1565C0",          # 정보
    "neutral": "#9CA3AF",       # 회색

    # 사분면 색상
    "quadrant_best": "#4CAF50",      # 최우수 (초록)
    "quadrant_expensive": "#FFC107",  # 비싼 우등 (노랑)
    "quadrant_low_eff": "#2196F3",   # 저효율 (파랑)
    "quadrant_loss": "#F44336",      # 손실 (빨강)

    # 등급 색상
    "grade_a": "#2E7D32",
    "grade_b": "#F9A825",
    "grade_c": "#C62828",

    # KPI 막대 색상
    "bar_highlight": "#284C76",
    "bar_dim": "#8DA9C3",

    # 평일/주말
    "weekday": "#8DA9C3",
    "weekend": "#284C76",

    # 평균선
    "avg_line": "#800020",
}

# ── 유형별 색상 ──
TYPE_COLORS = {
    "가입형": "#7B2D8E",
    "구매형": "#E8700A",
    "노출형": "#2196F3",
    "설치형": "#795548",
    "실행형": "#388E3C",
    "참여형": "#00838F",
    "클릭형": "#E53935",
    "기타": "#757575",
}

# ── 매체별 색상 ──
MEDIA_COLORS = {
    "네이버": "#2DB400",
    "앱": "#8E24AA",
    "유튜브": "#FF0000",
    "인스타그램": "#E1306C",
    "카카오": "#F9A825",
    "트위터/X": "#1DA1F2",
    "틱톡": "#00C9B7",
    "페이스북": "#4267B2",
    "웹": "#546E7A",
    "기타": "#757575",
}

# ── 카테고리별 색상 ──
CATEGORY_COLORS = {
    "간편미션-퀴즈": "#7B2D8E",
    "쇼핑-상품별": "#E8700A",
    "경험하기(CPA)": "#2DB400",
    "무료참여": "#2196F3",
    "유료참여": "#795548",
    "멀티보상(게임)": "#388E3C",
    "경험하기(CPI/CPE)": "#E53935",
    "앱(간편적립)": "#F9A825",
    "구독(간편적립)": "#5C6BC0",
    "금융(참여)": "#E1306C",
    "간편미션": "#00C9B7",
    "기타": "#757575",
}

# ── 칩 정렬 함수 ──
def sort_chips(items: list[str]) -> list[str]:
    """'전체' 맨 앞 + '기타' 맨 끝 + 나머지 가나다순"""
    others = sorted([x for x in items if x not in ("전체", "기타")])
    result = []
    if "전체" in items:
        result.append("전체")
    result.extend(others)
    if "기타" in items:
        result.append("기타")
    return result


# ── 상세 페이지 배지 설정 ──
BADGE_HERO_CONFIG = {
    "매체확장": {"icon": "↗", "tone": "#4A6BAA", "tone_soft": "#DCE6F2", "tone_deep": "#3A5A8A", "cta": "매체 확장 검토 →"},
    "승급추진": {"icon": "◐", "tone": "#2E7D32", "tone_soft": "#E8F5E9", "tone_deep": "#1B5E20", "cta": "예산 사전 확보 →"},
    "즉시조치": {"icon": "⚠", "tone": "#C25E55", "tone_soft": "#F4DCDA", "tone_deep": "#9A4842", "cta": "즉시 조치 →"},
    "우선검토": {"icon": "👁", "tone": "#E65100", "tone_soft": "#FFDAB3", "tone_deep": "#E65100", "cta": "모니터링 추가 →"},
}

BADGE_ENTRY_LABELS = {
    "매체확장": "M1 매체 확장 후보로 진입",
    "승급추진": "M1 승급 추진 대상으로 진입",
    "즉시조치": "M2 즉시 조치 필요로 진입",
    "우선검토": "M2 우선 검토로 진입",
}

DETAIL_BACK_LABELS = {
    "m1": "← M1 기회 광고 리스트로 돌아가기",
    "m2": "← M2 부진 위험 리스트로 돌아가기",
}


# ── 숫자 포맷 함수들 ──
def fmt_number(val, unit=""):
    """큰 숫자를 K/M으로 축약"""
    if val is None or (isinstance(val, float) and (val != val)):
        return "-"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M{unit}"
    elif abs(val) >= 1_000:
        return f"{val/1_000:.1f}K{unit}"
    else:
        return f"{val:,.0f}{unit}"


def fmt_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and (val != val)):
        return "-"
    return f"{val:.{decimals}f}%"


def fmt_currency(val):
    if val is None or (isinstance(val, float) and (val != val)):
        return "-"
    return f"{val:,.0f}원"


def fmt_won_man(val):
    """금액을 '만원' 단위로 포맷 (예: 672,300 → 67.2만원)"""
    if val is None or (isinstance(val, float) and (val != val)):
        return "-"
    return f"{val / 10_000:,.1f}만원"


