from __future__ import annotations

import pandas as pd
import streamlit as st

from core.i18n import t
from core.insights import insight_sentences


def render_page(frame: pd.DataFrame) -> None:
    st.title(t("conclusion.title"))
    st.caption(t("conclusion.caption"))

    for index, insight in enumerate(insight_sentences(frame), start=1):
        st.markdown(
            f"""
            <div class="surface-card" style="margin-bottom: 0.8rem;">
                <strong>{t("conclusion.insight_label", index=index)}</strong>
                <p class="hint-text" style="margin-bottom: 0;">{insight}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="surface-card">
            <strong>{t("conclusion.next_steps_title")}</strong>
            <p class="hint-text" style="margin-bottom: 0;">
                {t("conclusion.next_steps_body")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
