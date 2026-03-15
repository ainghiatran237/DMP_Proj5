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
from users import register_user


def render_page() -> None:
    inject_auth_styles(
        background_css="""
            radial-gradient(circle at 12% 16%, rgba(31, 60, 91, 0.12), transparent 0 24rem),
            radial-gradient(circle at 88% 12%, rgba(216, 166, 93, 0.16), transparent 0 22rem),
            linear-gradient(180deg, #f7f1e7 0%, #f2e8d8 55%, #ebdfcd 100%)
        """,
        hero_gradient="linear-gradient(135deg, #1f3c5b 0%, #d8a65d 100%)",
        accent_color="#d08f34",
        accent_surface="rgba(216, 166, 93, 0.14)",
    )

    hero_col, form_col = st.columns([1.1, 0.9], gap="large")

    with hero_col:
        render_auth_hero(
            badge=t("register.badge"),
            title=t("register.title"),
            body=t("register.body"),
            chips=[
                t("register.chip.local"),
                t("register.chip.session"),
                t("register.chip.return"),
            ],
        )
        render_auth_section_label(t("register.section"))
        stage_cols = st.columns(3, gap="small")
        with stage_cols[0]:
            render_auth_grid_card(
                eyebrow="01",
                title=t("register.card.create.title"),
                body=t("register.card.create.body"),
            )
        with stage_cols[1]:
            render_auth_grid_card(
                eyebrow="02",
                title=t("register.card.open.title"),
                body=t("register.card.open.body"),
            )
        with stage_cols[2]:
            render_auth_grid_card(
                eyebrow="03",
                title=t("register.card.explore.title"),
                body=t("register.card.explore.body"),
            )

    with form_col:
        render_auth_note_card(
            heading=t("register.note.title"),
            body=t("register.note.body"),
            bullet_points=[
                t("register.rule.username"),
                t("register.rule.password"),
                t("register.rule.confirm"),
            ],
        )

        with st.form("register_form", clear_on_submit=False):
            username = st.text_input(t("login.username"), placeholder="analyst01")
            password = st.text_input(t("login.password"), type="password")
            confirm_password = st.text_input(t("register.confirm_password"), type="password")
            submitted = st.form_submit_button(t("register.submit"), use_container_width=True)

        if submitted:
            success, message = register_user(username, password, confirm_password)
            if success:
                st.session_state["auth_notice"] = message
                st.session_state["auth_page"] = "login"
                st.rerun()
            else:
                st.error(message)

        render_auth_note_card(
            heading=t("register.switch.title"),
            body=t("register.switch.body"),
        )
        if st.button(t("register.switch.button"), key="switch_auth_login", use_container_width=True):
            st.session_state["auth_page"] = "login"
            st.rerun()
        st.markdown('<div class="auth-language-row"></div>', unsafe_allow_html=True)
        lang_cols = st.columns([0.28, 0.44, 0.28])
        with lang_cols[1]:
            render_language_selector(key="auth_language_register")
