from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from core.config import NAVIGATION_PAGES
from core.i18n import (
    display_source_label,
    nav_label,
    render_language_selector,
    t,
)
from users import logout_user


def render_sidebar(source_label: str, frame: pd.DataFrame) -> str:
    username = st.session_state.get("username") or "guest"
    prepared_rows = len(frame)
    room_types = frame["room_type"].nunique() if "room_type" in frame.columns else 0
    neighborhoods = frame["neighbourhood_group"].nunique() if "neighbourhood_group" in frame.columns else 0
    source_text = display_source_label(source_label)

    with st.sidebar:
        st.markdown('<div class="sidebar-brandmark">Airbnb</div>', unsafe_allow_html=True)
        profile_cols = st.columns([0.78, 0.22])
        with profile_cols[0]:
            st.markdown(
                f"""
                <div class="sidebar-profile">
                    <div class="sidebar-profile__avatar" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                            <circle cx="10" cy="7" r="4"></circle>
                            <path d="M20 8v6"></path>
                            <path d="M23 11h-6"></path>
                        </svg>
                        <span class="sidebar-profile__dot"></span>
                    </div>
                    <div class="sidebar-profile__meta">
                        <strong>{escape(username)}</strong>
                        <span>{escape(t("sidebar.live_badge"))}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with profile_cols[1]:
            st.markdown('<div class="sidebar-language-cell"></div>', unsafe_allow_html=True)
            render_language_selector(key="sidebar_language", compact=True)
        st.markdown(
            f'<div class="sidebar-section-title">{escape(t("sidebar.navigate"))}</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            t("sidebar.navigate"),
            options=NAVIGATION_PAGES,
            format_func=nav_label,
            label_visibility="collapsed",
            width="stretch",
        )
        st.markdown(
            f"""
            <div class="sidebar-current">
                <span>{escape(t("sidebar.current_view"))}</span>
                <strong>{escape(nav_label(page))}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-panel sidebar-panel--metrics">
                <div class="sidebar-panel__header">
                    <div>
                        <div class="sidebar-kicker">{escape(t("sidebar.dataset_status"))}</div>
                        <div class="sidebar-panel__title">{escape(source_text)}</div>
                    </div>
                    <div class="sidebar-source-pill">{escape(t("common.live"))}</div>
                </div>
                <p class="sidebar-panel__subtle">{escape(t("sidebar.dataset_ready"))}</p>
                <div class="sidebar-metrics">
                    <div class="sidebar-metric">
                        <strong>{prepared_rows:,}</strong>
                        <span>{escape(t("sidebar.prepared_rows"))}</span>
                    </div>
                    <div class="sidebar-metric">
                        <strong>{neighborhoods:,}</strong>
                        <span>{escape(t("sidebar.neighborhood_groups"))}</span>
                    </div>
                    <div class="sidebar-metric">
                        <strong>{room_types:,}</strong>
                        <span>{escape(t("sidebar.room_types"))}</span>
                    </div>
                    <div class="sidebar-metric">
                        <strong>{escape(t("common.live"))}</strong>
                        <span>{escape(t("common.analysis_mode"))}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-action-note">
                <div class="sidebar-kicker">{escape(t("sidebar.session_actions"))}</div>
                <p>{escape(t("sidebar.logout_hint"))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(t("sidebar.logout"), type="secondary", use_container_width=True):
            logout_user()
            st.rerun()
    return page
