from __future__ import annotations

import random
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core.config import DATASET_PATH, SAMPLE_SOURCE_LABEL


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(column).strip().lower())).strip("_")
        for column in normalized.columns
    ]

    aliases = {
        "neighbourhood_group": ["neighborhood_group"],
        "neighbourhood": ["neighborhood"],
        "number_of_reviews": ["reviews", "review_count"],
        "availability_365": ["availability", "available_days"],
    }
    rename_map: dict[str, str] = {}
    for canonical, options in aliases.items():
        if canonical in normalized.columns:
            continue
        for option in options:
            if option in normalized.columns:
                rename_map[option] = canonical
                break

    return normalized.rename(columns=rename_map)


def coerce_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_sample_dataset(rows: int = 240) -> pd.DataFrame:
    rng = random.Random(42)
    area_profiles = {
        "Manhattan": {
            "neighbourhoods": ["Midtown", "Chelsea", "Harlem", "SoHo"],
            "base_price": 220,
        },
        "Brooklyn": {
            "neighbourhoods": ["Williamsburg", "Bushwick", "Park Slope", "DUMBO"],
            "base_price": 165,
        },
        "Queens": {
            "neighbourhoods": ["Astoria", "Flushing", "Long Island City"],
            "base_price": 145,
        },
        "Bronx": {
            "neighbourhoods": ["Mott Haven", "Riverdale", "Fordham"],
            "base_price": 110,
        },
        "Staten Island": {
            "neighbourhoods": ["St. George", "Great Kills", "Tottenville"],
            "base_price": 125,
        },
    }
    room_profiles = {
        "Entire home/apt": {"multiplier": 1.35, "review_bias": 1.05},
        "Private room": {"multiplier": 0.82, "review_bias": 0.92},
        "Shared room": {"multiplier": 0.58, "review_bias": 0.78},
        "Hotel room": {"multiplier": 1.12, "review_bias": 1.18},
    }

    records: list[dict[str, object]] = []
    for listing_id in range(1, rows + 1):
        neighbourhood_group = rng.choice(list(area_profiles))
        neighbourhood = rng.choice(area_profiles[neighbourhood_group]["neighbourhoods"])
        room_type = rng.choices(
            population=list(room_profiles),
            weights=[0.48, 0.31, 0.09, 0.12],
            k=1,
        )[0]
        base_price = area_profiles[neighbourhood_group]["base_price"]
        room_factor = room_profiles[room_type]["multiplier"]
        volatility = rng.uniform(0.75, 1.35)
        price = round(base_price * room_factor * volatility, 2)
        review_bias = room_profiles[room_type]["review_bias"]
        reviews = max(0, int(rng.gauss(52 * review_bias, 18)))
        availability = max(0, min(365, int(rng.gauss(190, 75))))
        records.append(
            {
                "id": listing_id,
                "name": f"{neighbourhood} stay {listing_id}",
                "host_name": f"Host {rng.randint(10, 99)}",
                "neighbourhood_group": neighbourhood_group,
                "neighbourhood": neighbourhood,
                "room_type": room_type,
                "price": f"${price:,.2f}",
                "number_of_reviews": reviews,
                "reviews_per_month": round(max(0.1, rng.gauss(2.4, 1.1)), 2),
                "review_rate_number": rng.randint(3, 5),
                "minimum_nights": max(1, int(rng.gauss(4, 2))),
                "availability_365": availability,
                "last_review": date(2025, 1, 1) + timedelta(days=rng.randint(0, 380)),
            }
        )

    return pd.DataFrame(records)


def build_missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum().reset_index()
    missing.columns = ["column", "missing_values"]
    missing["missing_pct"] = (missing["missing_values"] / len(frame) * 100).round(2) if len(frame) else 0.0
    return missing.sort_values(["missing_values", "column"], ascending=[False, True])


def preprocess_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = normalize_columns(frame)
    missing_before = build_missing_table(cleaned)
    rows_before = len(cleaned)

    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed:
        cleaned = cleaned.drop_duplicates()

    numeric_columns = [
        "price",
        "service_fee",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
    ]
    for column in numeric_columns:
        if column not in cleaned.columns:
            continue
        if column in {"price", "service_fee"}:
            cleaned[column] = coerce_currency(cleaned[column])
        else:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "last_review" in cleaned.columns:
        cleaned["last_review"] = pd.to_datetime(cleaned["last_review"], errors="coerce")

    for column, fill_value in (
        ("neighbourhood_group", "Unknown"),
        ("neighbourhood", "Unknown"),
        ("room_type", "Unknown"),
    ):
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].fillna(fill_value)

    for column in ("number_of_reviews", "reviews_per_month", "availability_365"):
        if column not in cleaned.columns:
            continue
        fill_value = 0 if column != "availability_365" else cleaned[column].median()
        fill_value = 0 if pd.isna(fill_value) else fill_value
        cleaned[column] = cleaned[column].fillna(fill_value)

    removed_invalid_price = 0
    if "price" in cleaned.columns:
        valid_price_mask = cleaned["price"].notna() & (cleaned["price"] > 0)
        removed_invalid_price = int((~valid_price_mask).sum())
        cleaned = cleaned.loc[valid_price_mask].copy()

    cleaned = cleaned.reset_index(drop=True)
    missing_after = build_missing_table(cleaned)

    report = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "invalid_price_removed": removed_invalid_price,
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    return cleaned, report


def dataset_cache_key() -> str:
    if DATASET_PATH.exists():
        return str(DATASET_PATH.stat().st_mtime_ns)
    return "sample"


@st.cache_data(show_spinner=False)
def load_airbnb_bundle(_cache_key: str) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object]]:
    if DATASET_PATH.exists():
        raw_data = pd.read_csv(DATASET_PATH)
        source_label = str(DATASET_PATH)
    else:
        raw_data = build_sample_dataset()
        source_label = SAMPLE_SOURCE_LABEL

    cleaned_data, report = preprocess_data(raw_data)
    return raw_data, cleaned_data, source_label, report
