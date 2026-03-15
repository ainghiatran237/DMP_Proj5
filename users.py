from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

import streamlit as st

from core.i18n import DEFAULT_LANGUAGE, t

USERS_FILE = Path("data/users.json")


def initialize_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "chat_history": [],
        "auth_page": "login",
        "auth_notice": None,
        "language": DEFAULT_LANGUAGE,
        "raw_df": None,
        "raw_df_name": None,
        "processed_df": None,
        "preprocessing_before_df": None,
        "processing_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def _ensure_users_file() -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("{}", encoding="utf-8")


def _load_users() -> dict[str, dict[str, str]]:
    _ensure_users_file()
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8")) or {}
    except json.JSONDecodeError:
        return {}


def _save_users(users: dict[str, dict[str, str]]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"{salt}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(_hash_password(password, salt), stored_hash)


def register_user(username: str, password: str, confirm_password: str) -> tuple[bool, str]:
    normalized_username = _normalize_username(username)
    if len(normalized_username) < 3:
        return False, t("auth.error.username_short")
    if " " in normalized_username:
        return False, t("auth.error.username_spaces")
    if len(password) < 6:
        return False, t("auth.error.password_short")
    if password != confirm_password:
        return False, t("auth.error.password_mismatch")

    users = _load_users()
    if normalized_username in users:
        return False, t("auth.error.username_exists")

    users[normalized_username] = {
        "username": username.strip(),
        "password_hash": _hash_password(password),
    }
    _save_users(users)
    return True, t("auth.notice.account_created")


def login_user(username: str, password: str) -> tuple[bool, str]:
    normalized_username = _normalize_username(username)
    users = _load_users()
    user_record = users.get(normalized_username)

    if not user_record or not _verify_password(password, user_record["password_hash"]):
        return False, t("auth.error.invalid_login")

    st.session_state["authenticated"] = True
    st.session_state["username"] = user_record["username"]
    st.session_state["chat_history"] = []
    st.session_state["auth_page"] = "login"
    st.session_state["auth_notice"] = None
    st.session_state["raw_df"] = None
    st.session_state["raw_df_name"] = None
    st.session_state["processed_df"] = None
    st.session_state["preprocessing_before_df"] = None
    st.session_state["processing_report"] = None
    return True, t("auth.notice.welcome_back", username=user_record["username"])


def logout_user() -> None:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["chat_history"] = []
    st.session_state["auth_page"] = "login"
    st.session_state["auth_notice"] = None
    st.session_state["raw_df"] = None
    st.session_state["raw_df_name"] = None
    st.session_state["processed_df"] = None
    st.session_state["preprocessing_before_df"] = None
    st.session_state["processing_report"] = None
