"""공통 필터 렌더링 (Page 2/3/4 상단 필터 영역)"""
import calendar as _cal
import datetime as _dt

import streamlit as st
import pandas as pd
from src.config import COLORS, sort_chips, PERIOD_OPTIONS, DEFAULT_PERIOD


# ---------------------------------------------------------------------------
# 커스텀 달력 CSS (popover 내부 버튼 스타일)
# ---------------------------------------------------------------------------

_CALENDAR_CSS = """<style>
/* ── 달력 popover 크기 제한 ── */
[data-testid="stPopover"] > div > div > div[data-testid="stPopoverBody"] {
    max-width: 340px !important;
    padding: 12px !important;
}
/* 블록 간 수직 간격 축소 */
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
    gap: 0.15rem !important;
}
/* 달력 셀 버튼 — 컴팩트 */
[data-testid="stPopoverBody"] [data-testid="stButton"] > button {
    padding: 2px 0 !important;
    min-height: 28px !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}
/* 네비게이션 버튼 — 배경 제거 */
[data-testid="stPopoverBody"] [data-testid="stButton"] > button[kind="secondary"] {
    border: none !important;
    background: transparent !important;
}
/* 수평 블록(columns) 간격 축소 */
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"] {
    gap: 0.2rem !important;
}
/* 달력 초기화 버튼 — 빨간색 (cal-reset-wrap 클래스로 한정) */
.cal-reset-wrap button[data-testid="stBaseButton-secondary"] {
    background-color: #D32F2F !important;
    color: white !important;
    border: 1px solid #B71C1C !important;
    font-weight: 600 !important;
}
.cal-reset-wrap button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #B71C1C !important;
    border-color: #8B0000 !important;
}
</style>"""


def render_page_header(title: str):
    """페이지 헤더 + 오버뷰 돌아가기 버튼"""
    st.image(str(__import__('pathlib').Path(__file__).parent.parent / "assets" / "logo.png"), width=120)
    col1, col2 = st.columns([9, 1])
    with col1:
        st.markdown(f"## {title}")
    with col2:
        st.markdown("<div style='text-align: right; padding-top: 12px;'>", unsafe_allow_html=True)
        if st.button("← 전체 광고 현황", key=f"back_{title}"):
            st.session_state["scroll_to_top3"] = True
            st.switch_page("pages/01_전체_광고_현황_오버뷰.py")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sync_all_toggle(selected, all_non_all, prev_selected):
    """'전체' 토글 동기화. 반환: 동기화된 선택 리스트.
    전체 해제 시 빈 리스트 허용 (적용 버튼에서 최소 1개 검증)."""
    had_all = "전체" in (prev_selected or [])
    has_all = "전체" in (selected or [])
    cur_non_all = [x for x in (selected or []) if x != "전체"]

    if has_all and not had_all:
        return ["전체"] + list(all_non_all)
    if had_all and not has_all:
        return []
    if had_all and has_all and len(cur_non_all) < len(all_non_all):
        return cur_non_all
    if not has_all and cur_non_all and set(cur_non_all) == set(all_non_all):
        return ["전체"] + list(all_non_all)

    return list(selected or [])


def _render_pills_with_all(label, items, page_key, suffix):
    """st.pills + '전체' 토글 렌더링. '전체' 제외한 선택 리스트 반환."""
    sorted_items = sort_chips(items)
    all_non_all = [x for x in sorted_items if x != "전체"]
    all_options = ["전체"] + all_non_all

    wk = f"{page_key}_pending_{suffix}"
    pk = f"{page_key}_prev_{suffix}"

    if wk not in st.session_state:
        applied = st.session_state.get(f"{page_key}_applied_{suffix}", list(items))
        if set(applied) >= set(items):
            init_val = list(all_options)
        else:
            init_val = [x for x in all_non_all if x in applied]
        st.session_state[wk] = init_val
        st.session_state[pk] = list(init_val)

    prev = st.session_state.get(pk, list(all_options))
    cur = list(st.session_state[wk])
    synced = _sync_all_toggle(cur, all_non_all, prev)
    if synced != cur:
        st.session_state[wk] = synced

    st.markdown(f"**{label}**")
    selected = st.pills(
        label, all_options,
        selection_mode="multi",
        default=all_options,
        key=wk,
        label_visibility="collapsed",
    )

    st.session_state[pk] = list(selected) if selected else []
    return [x for x in (selected or []) if x != "전체"]


def _render_date_picker(page_key, data_date_min, data_date_max):
    """커스텀 달력 팝업 (st.popover).
    3단계: 달력 선택 → 달력 적용 → 상세필터 적용."""

    st.markdown(_CALENDAR_CSS, unsafe_allow_html=True)

    min_d = data_date_min.date()
    max_d = data_date_max.date()

    # 달력 상태 초기화
    if f"{page_key}_cal_year" not in st.session_state:
        st.session_state[f"{page_key}_cal_year"] = max_d.year
    if f"{page_key}_cal_month" not in st.session_state:
        st.session_state[f"{page_key}_cal_month"] = max_d.month

    year = st.session_state[f"{page_key}_cal_year"]
    month = st.session_state[f"{page_key}_cal_month"]
    cal_start = st.session_state.get(f"{page_key}_cal_start")
    cal_end = st.session_state.get(f"{page_key}_cal_end")

    # 선택 상태 요약 (popover 바깥 표시)
    pending = st.session_state.get(f"{page_key}_pending_dates")

    st.markdown("**관측 시점**")

    # popover 트리거 라벨 (버전 카운터로 적용 시 popover 닫기)
    cal_ver = st.session_state.get(f"{page_key}_cal_ver", 0)
    if pending:
        ps, pe = pending
        trigger_label = f"📅  {ps.strftime('%Y-%m-%d')} ~ {pe.strftime('%Y-%m-%d')}"
    else:
        trigger_label = "📅  날짜 직접 선택"
    # cal_ver 가 바뀌면 popover key 가 달라져 닫힌 상태로 재생성
    with st.popover(trigger_label, key=f"{page_key}_calpop_{cal_ver}"):
        # ── 월 네비게이션: ◀ 07 | 2025-08 | 09 ▶ ──
        prev_m = month - 1 if month > 1 else 12
        prev_y = year if month > 1 else year - 1
        next_m = month + 1 if month < 12 else 1
        next_y = year if month < 12 else year + 1

        nc1, nc2, nc3 = st.columns([1, 2, 1])
        with nc1:
            if st.button(f"◀ {prev_m:02d}", key=f"{page_key}_cprev",
                         use_container_width=True):
                st.session_state[f"{page_key}_cal_year"] = prev_y
                st.session_state[f"{page_key}_cal_month"] = prev_m
                st.rerun()
        with nc2:
            st.markdown(
                f"<div style='text-align:center; font-size:15px; "
                f"font-weight:bold; padding:2px 0;'>{year}-{month:02d}</div>",
                unsafe_allow_html=True,
            )
        with nc3:
            if st.button(f"{next_m:02d} ▶", key=f"{page_key}_cnext",
                         use_container_width=True):
                st.session_state[f"{page_key}_cal_year"] = next_y
                st.session_state[f"{page_key}_cal_month"] = next_m
                st.rerun()

        # ── 요일 헤더 (한글) ──
        st.markdown(
            '<div style="display:grid; grid-template-columns:repeat(7,1fr); '
            'text-align:center; font-size:12px; color:#666; margin:4px 0 0; '
            'gap:2px;">'
            '<span>일</span><span>월</span><span>화</span><span>수</span>'
            '<span>목</span><span>금</span><span>토</span></div>',
            unsafe_allow_html=True,
        )

        # ── 날짜 그리드 ──
        c = _cal.Calendar(firstweekday=6)  # 일요일 시작
        weeks = c.monthdayscalendar(year, month)

        for week in weeks:
            cols = st.columns(7)
            for di, day in enumerate(week):
                with cols[di]:
                    if day == 0:
                        st.markdown(
                            "<div style='height:28px'></div>",
                            unsafe_allow_html=True,
                        )
                        continue

                    d = _dt.date(year, month, day)
                    disabled = d < min_d or d > max_d

                    is_start = cal_start is not None and d == cal_start
                    is_end = cal_end is not None and d == cal_end
                    in_range = (
                        cal_start is not None
                        and cal_end is not None
                        and cal_start < d < cal_end
                    )

                    btn_type = (
                        "primary" if (is_start or is_end or in_range)
                        else "secondary"
                    )

                    if st.button(
                        str(day),
                        key=f"{page_key}_d{year}{month:02d}{day:02d}",
                        disabled=disabled,
                        type=btn_type,
                        use_container_width=True,
                    ):
                        _handle_day_click(page_key, d)
                        st.rerun()

        # ── 힌트 ──
        st.markdown(
            '<p style="color:#999; font-size:12px; margin:12px 0 4px;">'
            '같은 날짜를 두 번 클릭하면 1일만 선택됩니다</p>',
            unsafe_allow_html=True,
        )

        # ── 상태 메시지 + 초기화/적용 ──
        if cal_start is None:
            status = "시작일을 선택하세요"
            can_apply = False
        elif cal_end is None:
            status = f"{cal_start.strftime('%m/%d')} ~ 종료일 선택"
            can_apply = False
        else:
            ndays = (cal_end - cal_start).days + 1
            status = (
                f"{cal_start.strftime('%m/%d')} ~ "
                f"{cal_end.strftime('%m/%d')} ({ndays}일)"
            )
            can_apply = True

        st.divider()
        bc1, bc2, bc3 = st.columns([3, 1, 1])
        with bc1:
            st.markdown(
                f'<div style="padding-top:6px; font-size:13px;">{status}</div>',
                unsafe_allow_html=True,
            )
        with bc2:
            st.markdown('<div class="cal-reset-wrap">', unsafe_allow_html=True)
            if st.button("초기화", key=f"{page_key}_creset",
                         use_container_width=True):
                for k in [f"{page_key}_cal_start", f"{page_key}_cal_end"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with bc3:
            if st.button("적용", key=f"{page_key}_capply", type="primary",
                         disabled=not can_apply, use_container_width=True):
                st.session_state[f"{page_key}_pending_dates"] = (
                    cal_start, cal_end,
                )
                st.session_state[f"{page_key}_cal_ver"] = cal_ver + 1
                st.rerun()

    # 달력 적용 완료 안내
    st.caption(
        f"데이터 보유: {data_date_min.strftime('%Y-%m-%d')} ~ "
        f"{data_date_max.strftime('%Y-%m-%d')}"
    )
    if pending:
        ps, pe = pending
        pdays = (pe - ps).days + 1
        st.info(
            f"선택 기간: {ps.strftime('%Y-%m-%d')} ~ {pe.strftime('%Y-%m-%d')} "
            f"({pdays}일) — 하단의 **[적용]** 버튼을 눌러 대시보드에 반영하세요."
        )

    st.divider()


def _handle_day_click(page_key, d):
    """달력 날짜 클릭 처리."""
    cal_start = st.session_state.get(f"{page_key}_cal_start")
    cal_end = st.session_state.get(f"{page_key}_cal_end")

    if cal_start is None or (cal_start is not None and cal_end is not None):
        # 첫 선택 또는 리셋
        st.session_state[f"{page_key}_cal_start"] = d
        st.session_state[f"{page_key}_cal_end"] = None
    elif d == cal_start:
        # 같은 날짜 두 번 클릭 → 1일 선택
        st.session_state[f"{page_key}_cal_end"] = d
    elif d < cal_start:
        # 시작일보다 이전 → 새 시작일
        st.session_state[f"{page_key}_cal_start"] = d
    else:
        # 시작일 이후 → 종료일 설정
        st.session_state[f"{page_key}_cal_end"] = d


def _handle_chip_remove(page_key, action_key, group_items, sub_filter_items):
    """칩 제거 처리: applied 업데이트 + pending/prev 삭제."""
    if action_key == "period":
        for k in [
            f"{page_key}_period",
            f"{page_key}_start", f"{page_key}_end",
            f"{page_key}_date_range", f"{page_key}_date_pick",
            f"{page_key}_cal_start", f"{page_key}_cal_end",
            f"{page_key}_cal_year", f"{page_key}_cal_month",
            f"{page_key}_pending_dates", f"{page_key}_applied_dates",
        ]:
            if k in st.session_state:
                del st.session_state[k]

    elif action_key == "hours":
        st.session_state[f"{page_key}_applied_hours"] = (0, 23)
        if f"{page_key}_pending_hours" in st.session_state:
            del st.session_state[f"{page_key}_pending_hours"]

    elif action_key.startswith("group_"):
        name = action_key[6:]
        applied = list(st.session_state.get(f"{page_key}_applied_groups", group_items))
        applied = [g for g in applied if g != name]
        if not applied:
            applied = list(group_items)
        st.session_state[f"{page_key}_applied_groups"] = applied
        for k in [f"{page_key}_pending_groups", f"{page_key}_prev_groups"]:
            if k in st.session_state:
                del st.session_state[k]

    elif action_key.startswith("sub_"):
        name = action_key[4:]
        applied = list(st.session_state.get(f"{page_key}_applied_sub", sub_filter_items or []))
        applied = [s for s in applied if s != name]
        if not applied:
            applied = list(sub_filter_items or [])
        st.session_state[f"{page_key}_applied_sub"] = applied
        for k in [f"{page_key}_pending_sub", f"{page_key}_prev_sub"]:
            if k in st.session_state:
                del st.session_state[k]


def _reset_all_filters(page_key):
    """모든 필터 상태 초기화."""
    suffixes = [
        "period", "start", "end",
        "date_range", "date_pick",
        "cal_start", "cal_end", "cal_year", "cal_month", "cal_ver",
        "pending_dates", "applied_dates",
        "applied_groups", "applied_hours", "applied_sub",
        "pending_groups", "pending_sub", "pending_hours",
        "prev_groups", "prev_sub",
        "groups", "hours", "sub",
    ]
    for suffix in suffixes:
        k = f"{page_key}_{suffix}"
        if k in st.session_state:
            del st.session_state[k]


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_filters(
    page_key: str,
    group_col: str,
    group_items: list[str],
    group_label: str,
    data_date_min: pd.Timestamp,
    data_date_max: pd.Timestamp,
    show_sub_filter: bool = False,
    sub_filter_items: list[str] | None = None,
    sub_filter_label: str = "",
):
    """상단 필터 영역 렌더링.
    반환: (period, selected_groups, hour_range, custom_dates, sub_selected)"""

    # ── 페이지 전환 감지 → 상세 필터 초기화 ──
    last_page = st.session_state.get("_last_page")
    if last_page is not None and last_page != page_key:
        _reset_all_filters(page_key)
    st.session_state["_last_page"] = page_key

    # ── Applied 상태 초기화 ──
    if f"{page_key}_applied_groups" not in st.session_state:
        st.session_state[f"{page_key}_applied_groups"] = list(group_items)
    if f"{page_key}_applied_hours" not in st.session_state:
        st.session_state[f"{page_key}_applied_hours"] = (0, 23)
    if show_sub_filter and sub_filter_items:
        if f"{page_key}_applied_sub" not in st.session_state:
            st.session_state[f"{page_key}_applied_sub"] = list(sub_filter_items)

    # ── 1행: 관측 기준 / 관측 시점 (즉시 적용) ──
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**관측 기준**")
        obs_basis = st.radio(
            "관측 기준", ["유형", "매체", "카테고리"],
            horizontal=True, label_visibility="collapsed",
            key=f"{page_key}_obs_basis",
            index=["유형", "매체", "카테고리"].index(
                {"analysis_ads_type_label": "유형", "final_media": "매체",
                 "category_name": "카테고리"}.get(group_col, "유형")
            ),
        )
    with c2:
        st.markdown("**관측 시점**")
        period = st.radio(
            "관측 시점", PERIOD_OPTIONS,
            horizontal=True, label_visibility="collapsed",
            key=f"{page_key}_period",
            index=PERIOD_OPTIONS.index(DEFAULT_PERIOD),
        )

    # ── 2행: 상세 필터 (expander) ──
    # "사용자 지정" 선택 시 자동 펼침
    with st.expander("상세 필터", expanded=(period == "사용자 지정")):

        # 사용자 지정 날짜 (3단계 적용)
        if period == "사용자 지정":
            _render_date_picker(page_key, data_date_min, data_date_max)

        # 그룹 기준 pills
        pending_groups = _render_pills_with_all(group_label, group_items, page_key, "groups")

        # 보조 필터 pills
        pending_sub = list(sub_filter_items) if sub_filter_items else []
        if show_sub_filter and sub_filter_items:
            pending_sub = _render_pills_with_all(sub_filter_label, sub_filter_items, page_key, "sub")

        # 시간대 슬라이더
        st.markdown("**관측 시간대**")
        pending_hours = st.slider(
            "시간대", 0, 23, (0, 23),
            key=f"{page_key}_pending_hours",
        )

        # ── 상세 필터 적용 버튼 ──
        can_apply = bool(pending_groups)
        if show_sub_filter and sub_filter_items:
            can_apply = can_apply and bool(pending_sub)

        st.markdown("---")

        # ── 변경 사항 감지 ──
        applied_groups = st.session_state.get(f"{page_key}_applied_groups", list(group_items))
        applied_hours = st.session_state.get(f"{page_key}_applied_hours", (0, 23))
        applied_dates = st.session_state.get(f"{page_key}_applied_dates")
        pending_dates_val = st.session_state.get(f"{page_key}_pending_dates")

        changes = []
        if set(pending_groups) != set(applied_groups):
            added = set(pending_groups) - set(applied_groups)
            removed = set(applied_groups) - set(pending_groups)
            parts = []
            if added:
                parts.append(f"+{', '.join(sorted(added))}")
            if removed:
                parts.append(f"-{', '.join(sorted(removed))}")
            changes.append(f"{group_label}: {' / '.join(parts)}")
        if pending_hours != applied_hours:
            changes.append(f"시간대: {applied_hours[0]}~{applied_hours[1]}시 → {pending_hours[0]}~{pending_hours[1]}시")
        if show_sub_filter and sub_filter_items:
            applied_sub = st.session_state.get(f"{page_key}_applied_sub", list(sub_filter_items))
            if set(pending_sub) != set(applied_sub):
                added_s = set(pending_sub) - set(applied_sub)
                removed_s = set(applied_sub) - set(pending_sub)
                parts_s = []
                if added_s:
                    parts_s.append(f"+{', '.join(sorted(added_s))}")
                if removed_s:
                    parts_s.append(f"-{', '.join(sorted(removed_s))}")
                changes.append(f"{sub_filter_label}: {' / '.join(parts_s)}")
        if pending_dates_val != applied_dates:
            if pending_dates_val:
                ps, pe = pending_dates_val
                changes.append(f"기간: {ps.strftime('%Y-%m-%d')} ~ {pe.strftime('%Y-%m-%d')}")

        has_changes = bool(changes)

        hint_col, apply_col = st.columns([4, 1])
        with hint_col:
            if has_changes:
                st.markdown(
                    '<div style="padding-top:4px; font-size:12px; color:#ff6b35; font-weight:500;">'
                    '⚠ 변경사항이 있습니다. [적용] 버튼을 눌러 반영하세요</div>',
                    unsafe_allow_html=True,
                )
        with apply_col:
            if st.button("적용", type="primary", key=f"{page_key}_apply",
                         use_container_width=True, disabled=not can_apply):
                # 그룹/시간대 적용
                st.session_state[f"{page_key}_applied_groups"] = (
                    pending_groups if pending_groups else list(group_items)
                )
                st.session_state[f"{page_key}_applied_hours"] = pending_hours
                if show_sub_filter and sub_filter_items:
                    st.session_state[f"{page_key}_applied_sub"] = (
                        pending_sub if pending_sub else list(sub_filter_items)
                    )
                # 날짜 적용 (pending_dates → applied_dates)
                pending_dates = st.session_state.get(f"{page_key}_pending_dates")
                if pending_dates:
                    st.session_state[f"{page_key}_applied_dates"] = pending_dates
                st.rerun()

        if not can_apply:
            st.caption("하나 이상의 항목을 선택해야 적용할 수 있습니다.")

    # ── 페이지 전환 처리 ──
    basis_map = {"유형": "02", "매체": "03", "카테고리": "04"}
    current_page = {"analysis_ads_type_label": "02", "final_media": "03",
                    "category_name": "04"}.get(group_col, "02")
    selected_page = basis_map.get(obs_basis, current_page)
    if selected_page != current_page:
        if f"{page_key}_obs_basis" in st.session_state:
            del st.session_state[f"{page_key}_obs_basis"]
        page_files = {
            "02": "pages/02_유형별_운영_탐색.py",
            "03": "pages/03_매체별_운영_탐색.py",
            "04": "pages/04_카테고리별_운영_탐색.py",
        }
        st.switch_page(page_files[selected_page])

    # ── 반환값: applied 상태에서 추출 ──
    applied_groups = st.session_state[f"{page_key}_applied_groups"]
    applied_hours = st.session_state[f"{page_key}_applied_hours"]
    applied_sub = st.session_state.get(f"{page_key}_applied_sub", sub_filter_items or [])

    custom_start = None
    custom_end = None
    if period == "사용자 지정":
        applied_dates = st.session_state.get(f"{page_key}_applied_dates")
        if applied_dates:
            custom_start, custom_end = applied_dates

    return period, applied_groups, applied_hours, (custom_start, custom_end), applied_sub


# ---------------------------------------------------------------------------
# 적용 중 칩 행 — 페이지에서 필터링 후 호출
# ---------------------------------------------------------------------------

def render_applied_chips(
    page_key: str,
    filtered_count: int,
    group_items: list[str],
    group_label: str,
    show_sub_filter: bool = False,
    sub_filter_items: list[str] | None = None,
    sub_filter_label: str = "",
):
    """적용 중 필터 칩 + 필터링된 데이터 수 + 필터 초기화 렌더링."""
    period = st.session_state.get(f"{page_key}_period", DEFAULT_PERIOD)
    applied_groups = st.session_state.get(f"{page_key}_applied_groups", group_items)
    applied_hours = st.session_state.get(f"{page_key}_applied_hours", (0, 23))
    applied_sub = st.session_state.get(f"{page_key}_applied_sub", sub_filter_items or [])

    # 비기본값 필터만 칩으로 생성
    chip_items: list[tuple[str, str]] = []
    if period != DEFAULT_PERIOD:
        if period == "사용자 지정":
            applied_dates = st.session_state.get(f"{page_key}_applied_dates")
            pending_dates = st.session_state.get(f"{page_key}_pending_dates")
            dates = applied_dates or pending_dates
            if dates:
                s, e = dates
                days = (e - s).days + 1
                chip_items.append((
                    "period",
                    f"기간·{s.strftime('%m-%d')} ~ {e.strftime('%m-%d')} ({days}일)",
                ))
            else:
                chip_items.append(("period", "기간·사용자 지정 (미선택)"))
        else:
            chip_items.append(("period", f"기간·{period}"))
    pending_hours = st.session_state.get(f"{page_key}_pending_hours", (0, 23))
    display_hours = applied_hours if applied_hours != (0, 23) else pending_hours
    if display_hours != (0, 23):
        chip_items.append(("hours", f"시간대·{display_hours[0]}-{display_hours[1]}시"))
    if set(applied_groups) != set(group_items):
        for g in applied_groups:
            chip_items.append((f"group_{g}", f"{group_label}·{g}"))
    if show_sub_filter and sub_filter_items and set(applied_sub) != set(sub_filter_items):
        for s in applied_sub:
            chip_items.append((f"sub_{s}", f"{sub_filter_label}·{s}"))

    # 필터 초기화 버튼 빨간색 + 컨테이너 배경
    st.markdown(
        "<style>"
        "[data-testid='stVerticalBlockBorderWrapper']:has( "
        "  button[data-testid='stBaseButton-primary']"
        ") > div:first-child {"
        "  background-color: #FFFFFF !important;"
        "}"
        "/* 필터 초기화 버튼(primary) — 빨간색 오버라이드 */"
        "[data-testid='stVerticalBlockBorderWrapper'] "
        "  button[data-testid='stBaseButton-primary'] {"
        "  background-color: #D32F2F !important;"
        "  color: white !important;"
        "  border: 1px solid #B71C1C !important;"
        "  font-weight: 600 !important;"
        "}"
        "[data-testid='stVerticalBlockBorderWrapper'] "
        "  button[data-testid='stBaseButton-primary']:hover {"
        "  background-color: #B71C1C !important;"
        "  border-color: #8B0000 !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        if not chip_items:
            c1, c2, c3 = st.columns([2.5, 4, 2])
            with c1:
                st.markdown(
                    '<div style="padding-top:4px;">'
                    '<span style="font-weight:600; font-size:13px; color:#222222;">적용 중:</span>'
                    '&nbsp;&nbsp;<span style="font-size:13px; color:#888;">기본값 (전체)</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div style="text-align:right; font-size:13px; color:#888; padding-top:4px;">'
                    f'필터링된 데이터 {filtered_count:,}건</div>',
                    unsafe_allow_html=True,
                )
            with c3:
                if st.button("필터 초기화", key=f"{page_key}_reset", type="primary", use_container_width=True):
                    _reset_all_filters(page_key)
                    st.rerun()
        else:
            # 인라인 HTML 칩 생성
            chips_html = ""
            for _, label in chip_items:
                chips_html += (
                    '<span style="display:inline-block; background:#E8E0D4; color:#333;'
                    ' border-radius:16px; padding:4px 12px; margin:2px 4px;'
                    ' font-size:13px; font-weight:500; white-space:nowrap;">'
                    f'{label}</span>'
                )

            c1, c2 = st.columns([8, 1.5])
            with c1:
                st.markdown(
                    '<div style="display:flex; align-items:center; flex-wrap:wrap; gap:2px; padding-top:2px;">'
                    '<span style="font-weight:600; font-size:13px; color:#222222; margin-right:8px; white-space:nowrap;">적용 중:</span>'
                    f'{chips_html}'
                    f'<span style="font-size:13px; color:#888; margin-left:auto; white-space:nowrap;">'
                    f'필터링된 데이터 {filtered_count:,}건</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("필터 초기화", key=f"{page_key}_reset", type="primary", use_container_width=True):
                    _reset_all_filters(page_key)
                    st.rerun()
