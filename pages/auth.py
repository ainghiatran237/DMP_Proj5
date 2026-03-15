from __future__ import annotations

from html import escape

import streamlit as st


def inject_auth_styles(
    *,
    background_css: str,
    hero_gradient: str,
    accent_color: str,
    accent_surface: str,
) -> None:
    st.markdown(
        f"""
        <style>
            .block-container {{
                max-width: 1220px;
                padding-top: clamp(6.25rem, 14vh, 8.75rem);
                padding-bottom: 3rem;
            }}
            .stApp {{
                background: {background_css};
            }}
            .auth-hero-card,
            .auth-grid-card,
            .auth-note-card {{
                animation: auth-fade-up 0.48s ease both;
            }}
            .auth-hero-card {{
                position: relative;
                overflow: hidden;
                background: {hero_gradient};
                color: #f8f3ec;
                border-radius: 24px;
                padding: 1.55rem 1.6rem;
                box-shadow: 0 18px 42px rgba(31, 60, 91, 0.16);
            }}
            .auth-hero-card::after {{
                content: "";
                position: absolute;
                right: -3rem;
                bottom: -4rem;
                width: 11rem;
                height: 11rem;
                border-radius: 999px;
                background: rgba(248, 243, 236, 0.12);
            }}
            .auth-badge {{
                display: inline-block;
                padding: 0.38rem 0.78rem;
                border-radius: 999px;
                background: rgba(248, 243, 236, 0.15);
                border: 1px solid rgba(248, 243, 236, 0.12);
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .auth-hero-card h1 {{
                margin: 0.95rem 0 0.7rem;
                font-size: 2.3rem;
                line-height: 1.08;
                color: #f8f3ec;
                max-width: 12ch;
            }}
            .auth-hero-card p {{
                margin: 0;
                max-width: 38rem;
                line-height: 1.62;
                color: rgba(248, 243, 236, 0.9);
            }}
            .auth-chip-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.05rem;
            }}
            .auth-chip {{
                padding: 0.45rem 0.78rem;
                border-radius: 999px;
                background: rgba(248, 243, 236, 0.12);
                border: 1px solid rgba(248, 243, 236, 0.12);
                font-size: 0.88rem;
            }}
            .auth-section-label {{
                margin: 1.05rem 0 0.72rem;
                color: #596474;
                font-size: 0.8rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .auth-grid-card {{
                min-height: 9rem;
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(30, 36, 48, 0.08);
                border-radius: 20px;
                padding: 1rem;
                box-shadow: 0 12px 28px rgba(31, 60, 91, 0.08);
                backdrop-filter: blur(10px);
            }}
            .auth-grid-card__eyebrow {{
                display: inline-block;
                margin-bottom: 0.55rem;
                padding: 0.32rem 0.6rem;
                border-radius: 999px;
                background: {accent_surface};
                color: {accent_color};
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.05em;
            }}
            .auth-grid-card strong {{
                display: block;
                margin-bottom: 0.4rem;
                color: #1f3c5b;
                font-size: 1rem;
            }}
            .auth-grid-card p {{
                margin: 0;
                color: #5d6472;
                line-height: 1.56;
                font-size: 0.93rem;
            }}
            .auth-note-card {{
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(30, 36, 48, 0.08);
                border-radius: 20px;
                padding: 1rem 1.05rem;
                margin-bottom: 0.8rem;
                box-shadow: 0 12px 28px rgba(31, 60, 91, 0.08);
                color: #5d6472;
                line-height: 1.6;
            }}
            .auth-note-card h3,
            .auth-note-card h4 {{
                margin: 0 0 0.35rem;
                color: #1e2430;
            }}
            .auth-note-card p {{
                margin: 0;
            }}
            .auth-note-list {{
                margin: 0.8rem 0 0;
                padding-left: 1rem;
                color: #5d6472;
            }}
            .auth-note-list li {{
                margin-bottom: 0.35rem;
            }}
            .stButton > button {{
                min-height: 2.7rem;
                border-radius: 999px;
            }}
            .stButton > button[kind="tertiary"] {{
                min-height: 2.9rem;
                padding: 0.5rem 1rem;
                border-radius: 999px;
                border: 1px solid rgba(120, 126, 140, 0.26);
                background: rgba(255, 255, 255, 0.78);
                color: #6a7180;
                box-shadow: 0 8px 18px rgba(31, 60, 91, 0.08);
                font-weight: 600;
                white-space: nowrap;
            }}
            .stButton > button[kind="tertiary"] > div {{
                align-items: center;
                justify-content: center;
                gap: 0.48rem;
            }}
            .stButton > button[kind="tertiary"]:hover {{
                border-color: rgba(201, 92, 54, 0.24);
                background: rgba(255, 255, 255, 0.96);
            }}
            .auth-language-row {{
                margin-top: 0.65rem;
            }}
            @keyframes auth-fade-up {{
                from {{
                    opacity: 0;
                    transform: translateY(14px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            @media (max-width: 900px) {{
                .auth-hero-card h1 {{
                    font-size: 1.95rem;
                    max-width: none;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth_hero(*, badge: str, title: str, body: str, chips: list[str]) -> None:
    chips_html = "".join(f'<div class="auth-chip">{escape(chip)}</div>' for chip in chips)
    st.markdown(
        f"""
        <div class="auth-hero-card">
            <div class="auth-badge">{escape(badge)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(body)}</p>
            <div class="auth-chip-row">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_section_label(label: str) -> None:
    st.markdown(
        f'<div class="auth-section-label">{escape(label)}</div>',
        unsafe_allow_html=True,
    )


def render_auth_grid_card(*, eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="auth-grid-card">
            <div class="auth-grid-card__eyebrow">{escape(eyebrow)}</div>
            <strong>{escape(title)}</strong>
            <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_note_card(*, heading: str, body: str, bullet_points: list[str] | None = None) -> None:
    bullets_html = ""
    if bullet_points:
        bullet_items = "".join(f"<li>{escape(item)}</li>" for item in bullet_points)
        bullets_html = f'<ul class="auth-note-list">{bullet_items}</ul>'

    tag = "h3" if bullet_points else "h4" if heading.endswith("?") else "h3"
    st.markdown(
        f"""
        <div class="auth-note-card">
            <{tag}>{escape(heading)}</{tag}>
            <p>{escape(body)}</p>
            {bullets_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
