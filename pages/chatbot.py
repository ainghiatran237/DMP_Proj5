from __future__ import annotations

import pandas as pd
import streamlit as st

from core.i18n import t
from core.insights import answer_chat_question

def render_page(frame: pd.DataFrame) -> None:
    st.title(t("chatbot.title"))
    st.caption(t("chatbot.caption"))

    quick_prompts = [
        t("chatbot.quick.price"),
        t("chatbot.quick.room_mix"),
        t("chatbot.quick.expensive_area"),
        t("chatbot.quick.availability"),
    ]

    quick_prompt = None
    prompt_cols = st.columns(4)
    for column, prompt_text in zip(prompt_cols, quick_prompts):
        if column.button(prompt_text, use_container_width=True):
            quick_prompt = prompt_text

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input(t("chatbot.input"))
    final_prompt = user_prompt or quick_prompt

    if final_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        assistant_reply = answer_chat_question(final_prompt, frame)
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
