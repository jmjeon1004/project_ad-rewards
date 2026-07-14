"""아이브코리아 광고운영 최적화 대시보드 — 엔트리포인트"""
import streamlit as st

st.set_page_config(
    page_title="아이브코리아 광고운영 최적화 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pages = [
    st.Page("pages/01_전체_광고_현황_오버뷰.py", title="전체 광고 현황 오버뷰", default=True),
    st.Page("pages/02_유형별_운영_탐색.py", title="유형별 운영 탐색"),
    st.Page("pages/03_매체별_운영_탐색.py", title="매체별 운영 탐색"),
    st.Page("pages/04_카테고리별_운영_탐색.py", title="카테고리별 운영 탐색"),
    st.Page("pages/05_광고_상세.py", title="광고 상세", visibility="hidden"),
]

nav = st.navigation(pages)
nav.run()
