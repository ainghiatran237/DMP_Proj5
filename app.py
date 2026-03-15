from __future__ import annotations

import streamlit as st

from core.data import dataset_cache_key, load_airbnb_bundle
from core.i18n import get_app_title
from core.styles import inject_styles, set_page_config
from pages.chatbot import render_page as render_chatbot_page
from pages.conclusion import render_page as render_conclusion_page
from pages.data_raw import render_page as render_data_raw_page
from pages.eda import render_page as render_eda_page
from pages.login import render_page as render_login_page
from pages.overview import render_page as render_overview_page
from pages.preprocessing import render_page as render_preprocessing_page
from pages.register import render_page as render_register_page
from pages.sidebar import render_sidebar
from users import initialize_session_state


def main() -> None:
    initialize_session_state()
    set_page_config(get_app_title())
    inject_styles()

    if not st.session_state["authenticated"]:
        auth_page = st.session_state.get("auth_page", "login")
        if auth_page == "register":
            render_register_page()
        else:
            render_login_page()
        return

    raw_data, cleaned_data, source_label, report = load_airbnb_bundle(dataset_cache_key())
    page = render_sidebar(source_label, cleaned_data)

    if page == "overview":
        render_overview_page(cleaned_data, source_label)
    elif page == "data_raw":
        render_data_raw_page(raw_data, cleaned_data)
    elif page == "preprocessing":
        render_preprocessing_page(raw_data, cleaned_data, report)
    elif page == "eda":
        render_eda_page(cleaned_data)
    elif page == "conclusion":
        render_conclusion_page(cleaned_data)
    elif page == "chatbot":
        render_chatbot_page(cleaned_data)


if __name__ == "__main__":
    main()
