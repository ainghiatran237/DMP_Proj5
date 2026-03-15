from __future__ import annotations

import pandas as pd


def format_currency(value: float | int | None, fallback: str = "N/A") -> str:
    if value is None or pd.isna(value):
        return fallback
    return f"${value:,.0f}"
