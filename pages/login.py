from __future__ import annotations

import streamlit as st

from core.i18n import render_language_selector, t
from pages.auth import (
    inject_auth_styles,
    render_auth_grid_card,
    render_auth_hero,
    render_auth_note_card,
    render_auth_section_label,
)
from users import login_user


def render_page() -> None:
    inject_auth_styles(
        background_css="""
            radial-gradient(circle at top left, rgba(201, 92, 54, 0.18), transparent 0 26rem),
            radial-gradient(circle at top right, rgba(31, 60, 91, 0.14), transparent 0 24rem),
            linear-gradient(180deg, #f8f3ec 0%, #f3eadf 55%, #eee2d2 100%)
        """,
        hero_gradient="linear-gradient(135deg, #1f3c5b 0%, #c95c36 100%)",
        accent_color="#c95c36",
        accent_surface="rgba(201, 92, 54, 0.12)",
    )

    hero_col, form_col = st.columns([1.1, 0.9], gap="large")

    with hero_col:
        render_auth_hero(
            badge=t("login.badge"),
            title=t("login.title"),
            body=t("login.body"),
            chips=[
                t("login.chip.charts"),
                t("login.chip.preprocessing"),
                t("login.chip.chatbot"),
            ],
        )
        render_auth_section_label(t("login.section"))
        card_cols = st.columns(3, gap="small")
        with card_cols[0]:
            render_auth_grid_card(
                eyebrow=t("login.card.price.eyebrow"),
                title=t("login.card.price.title"),
                body=t("login.card.price.body"),
            )
        with card_cols[1]:
            render_auth_grid_card(
                eyebrow=t("login.card.room.eyebrow"),
                title=t("login.card.room.title"),
                body=t("login.card.room.body"),
            )
        with card_cols[2]:
            render_auth_grid_card(
                eyebrow=t("login.card.demand.eyebrow"),
                title=t("login.card.demand.title"),
                body=t("login.card.demand.body"),
            )

    with form_col:
        render_auth_note_card(
            heading=t("login.note.title"),
            body=t("login.note.body"),
        )

        notice = st.session_state.get("auth_notice")
        if notice:
            st.success(notice)
            st.session_state["auth_notice"] = None

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(t("login.username"), placeholder="analyst01")
            password = st.text_input(t("login.password"), type="password")
            submitted = st.form_submit_button(t("login.submit"), use_container_width=True)

        if submitted:
            success, message = login_user(username, password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        render_auth_note_card(
            heading=t("login.switch.title"),
            body=t("login.switch.body"),
        )
        if st.button(t("login.switch.button"), key="switch_auth_register", use_container_width=True):
            st.session_state["auth_page"] = "register"
            st.rerun()
        st.markdown('<div class="auth-language-row"></div>', unsafe_allow_html=True)
        lang_cols = st.columns([0.28, 0.44, 0.28])
        with lang_cols[1]:
            render_language_selector(key="auth_language_login")
