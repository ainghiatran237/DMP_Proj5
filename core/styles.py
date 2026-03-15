from __future__ import annotations

import streamlit as st

from core.config import APP_TITLE


def set_page_config(page_title: str | None = None) -> None:
    st.set_page_config(
        page_title=page_title or APP_TITLE,
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: clamp(3.5rem, 8vh, 5rem);
                padding-bottom: 2rem;
            }
            .stButton > button[kind="tertiary"] {
                min-height: 2.9rem;
                padding: 0.5rem 1rem;
                border-radius: 999px;
                border: 1px solid rgba(120, 126, 140, 0.28);
                background: rgba(255, 255, 255, 0.72);
                color: #6a7180;
                box-shadow: 0 6px 18px rgba(31, 60, 91, 0.08);
                font-weight: 600;
                white-space: nowrap;
            }
            .stButton > button[kind="tertiary"] > div {
                align-items: center;
                justify-content: center;
                gap: 0.48rem;
            }
            .stButton > button[kind="tertiary"]:hover {
                border-color: rgba(201, 92, 54, 0.25);
                background: rgba(255, 255, 255, 0.92);
                color: #4e5664;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
            [data-testid="stSidebarCollapseButton"] {
                display: none;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f3f3f1 0%, #ecece9 100%);
                border-right: 1px solid rgba(30, 36, 48, 0.06);
                min-width: 18.5rem !important;
                max-width: 18.5rem !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                width: 18.5rem !important;
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 0.95rem;
                padding-bottom: 1.4rem;
            }
            .sidebar-brandmark {
                margin: 0.1rem 0 0.95rem;
                color: #e35b16;
                text-align: center;
                font-size: 2.85rem;
                font-weight: 800;
                line-height: 0.95;
                letter-spacing: -0.04em;
            }
            .sidebar-profile {
                display: flex;
                align-items: center;
                gap: 0.9rem;
                margin: 0.15rem 0 1.05rem;
                padding: 0.15rem 0.15rem 0.15rem 0.05rem;
            }
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.sidebar-profile):has(.sidebar-language-cell) {
                align-items: center;
            }
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.sidebar-profile):has(.sidebar-language-cell) > div[data-testid="column"] {
                display: flex;
                align-items: center;
            }
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.sidebar-profile):has(.sidebar-language-cell) > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
                display: flex;
                gap: 0rem;
                width: 100%;
                max-width: 100%;
                height: 100%;
                min-width: 1rem;
                flex-flow: column;
                flex: 1 1 0%;
                align-items: stretch;
                justify-content: flex-start;
            }
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.sidebar-profile):has(.sidebar-language-cell) > div[data-testid="column"]:last-child {
                justify-content: center;
            }
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.sidebar-profile):has(.sidebar-language-cell) > div[data-testid="column"]:last-child .stButton {
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .sidebar-profile__avatar {
                position: relative;
                width: 3.1rem;
                height: 3.1rem;
                min-width: 3.1rem;
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(180deg, #79a5a4 0%, #5f8f8d 100%);
                box-shadow: 0 10px 22px rgba(95, 143, 141, 0.22);
                color: #eef7f6;
                border: 2px solid rgba(255, 255, 255, 0.8);
            }
            .sidebar-profile__avatar svg {
                width: 1.45rem;
                height: 1.45rem;
            }
            .sidebar-profile__dot {
                position: absolute;
                right: 0.14rem;
                bottom: 0.12rem;
                width: 0.64rem;
                height: 0.64rem;
                border-radius: 999px;
                background: #42d680;
                border: 2px solid #f3f3f1;
            }
            .sidebar-profile__meta strong {
                display: block;
                color: #1f2a3c;
                font-size: 1.08rem;
                line-height: 1.15;
            }
            .sidebar-profile__meta span {
                display: block;
                margin-top: 0.22rem;
                color: #f1670f;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .sidebar-language-cell {
                display: none;
            }
            .sidebar-panel {
                background: rgba(255, 255, 255, 0.84);
                border: 1px solid rgba(30, 36, 48, 0.08);
                border-radius: 18px;
                padding: 1rem;
                margin-bottom: 0.95rem;
                box-shadow: 0 10px 24px rgba(31, 60, 91, 0.08);
                backdrop-filter: blur(10px);
            }
            .sidebar-kicker,
            .sidebar-section-title {
                color: #7a8594;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .sidebar-panel__header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 0.85rem;
            }
            .sidebar-panel__title {
                margin-top: 0.35rem;
                color: #273246;
                font-size: 1.02rem;
                font-weight: 700;
                line-height: 1.2;
            }
            .sidebar-panel p {
                margin: 0.45rem 0 0;
                color: #697385;
                line-height: 1.55;
                font-size: 0.88rem;
            }
            .sidebar-panel__subtle {
                margin-top: 0.55rem !important;
            }
            .sidebar-source-pill {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.34rem 0.68rem;
                border-radius: 999px;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                white-space: nowrap;
            }
            .sidebar-source-pill {
                background: rgba(86, 98, 116, 0.08);
                border: 1px solid rgba(86, 98, 116, 0.08);
                color: #566274;
            }
            .sidebar-metrics {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.55rem;
                margin-top: 0.85rem;
            }
            .sidebar-metric {
                background: #f8f8f7;
                border: 1px solid rgba(30, 36, 48, 0.06);
                border-radius: 14px;
                padding: 0.72rem 0.76rem;
            }
            .sidebar-metric strong {
                display: block;
                color: #2b3649;
                font-size: 1.02rem;
                line-height: 1.1;
            }
            .sidebar-metric span {
                display: block;
                margin-top: 0.28rem;
                color: #808999;
                font-size: 0.77rem;
                line-height: 1.35;
            }
            .sidebar-section-title {
                margin: 0.2rem 0 0.65rem;
            }
            .sidebar-current {
                margin-top: 0.9rem;
                margin-bottom: 0.95rem;
                padding: 0.9rem 1rem;
                background: rgba(255, 255, 255, 0.84);
                border: 1px solid rgba(30, 36, 48, 0.08);
                border-radius: 16px;
                box-shadow: 0 8px 18px rgba(31, 60, 91, 0.06);
            }
            .sidebar-current span {
                display: block;
                color: #7a8594;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .sidebar-current strong {
                display: block;
                margin-top: 0.34rem;
                color: #273246;
                font-size: 1rem;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] > div {
                gap: 0.28rem;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
                appearance: none;
                -webkit-appearance: none;
                position: absolute;
                opacity: 0;
                width: 1px;
                height: 1px;
                pointer-events: none;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label {
                position: relative;
                align-items: center !important;
                gap: 0;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 16px;
                padding: 0.88rem 0.95rem 0.88rem 3rem;
                margin-bottom: 0.06rem;
                transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label > div:last-child {
                min-width: 0;
                width: 100%;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label::before {
                content: "◫";
                position: absolute;
                left: 0.92rem;
                top: 50%;
                transform: translateY(-50%);
                width: 1.4rem;
                height: 1.4rem;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 10px;
                background: rgba(31, 60, 91, 0.08);
                color: #596678;
                font-size: 0.82rem;
                font-weight: 700;
                line-height: 1;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-of-type {
                display: none !important;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
                transform: translateX(2px);
                background: rgba(255, 255, 255, 0.6);
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
                background: #f1670f;
                box-shadow: 0 10px 22px rgba(241, 103, 15, 0.28);
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label p {
                margin: 0;
                color: #5b6778;
                font-size: 1.03rem;
                font-weight: 700;
                line-height: 1.15;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
            [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) label,
            [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) span {
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked)::before {
                background: rgba(255, 255, 255, 0.18);
                color: #ffffff;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:nth-of-type(2)::before {
                content: "≣";
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:nth-of-type(3)::before {
                content: "◇";
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:nth-of-type(4)::before {
                content: "◔";
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:nth-of-type(5)::before {
                content: "✓";
                font-size: 0.9rem;
            }
            [data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:nth-of-type(6)::before {
                content: "✦";
            }
            [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                min-height: 3rem;
                border-radius: 16px;
                border: 1px solid rgba(31, 60, 91, 0.12);
                background: rgba(255, 255, 255, 0.88);
                color: #324055;
                font-weight: 700;
                box-shadow: 0 12px 24px rgba(31, 60, 91, 0.08);
            }
            [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                border-color: rgba(241, 103, 15, 0.18);
                box-shadow: 0 14px 26px rgba(31, 60, 91, 0.1);
            }
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
                width: auto !important;
                min-width: 0 !important;
                max-width: none !important;
                height: auto !important;
                min-height: 0 !important;
                max-height: none !important;
                padding: 0.15rem !important;
                border: none !important;
                background: transparent !important;
                color: #273246 !important;
                box-shadow: none !important;
                font-size: 2.25rem !important;
                margin-inline: auto !important;
            }
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"] > div {
                align-items: center;
                justify-content: center;
                gap: 0 !important;
            }
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"] > div p {
                display: none !important;
            }
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
                border: none !important;
                background: transparent !important;
                color: #111827 !important;
                box-shadow: none !important;
            }
            .sidebar-action-note {
                margin: 0.2rem 0 0.7rem;
                padding: 0.85rem 0.95rem;
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(30, 36, 48, 0.08);
                box-shadow: 0 8px 18px rgba(31, 60, 91, 0.06);
            }
            .sidebar-action-note p {
                margin: 0.45rem 0 0;
                color: #697385;
                line-height: 1.52;
                font-size: 0.86rem;
            }
            .hero {
                background: linear-gradient(135deg, #1f3c5b 0%, #c95c36 100%);
                color: #f8f6f1;
                border-radius: 18px;
                padding: 1.5rem 1.6rem;
                margin-bottom: 1rem;
                box-shadow: 0 16px 40px rgba(31, 60, 91, 0.18);
            }
            .hero h1 {
                font-size: 2rem;
                margin: 0;
            }
            .hero p {
                margin: 0.6rem 0 0;
                max-width: 52rem;
            }
            .surface-card {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(30, 36, 48, 0.08);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                backdrop-filter: blur(10px);
            }
            .hint-text {
                color: #5d6472;
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
