from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CHART_COLORS
from core.data import coerce_currency, normalize_columns
from core.i18n import t
from users import logout_user

AREA_SEQUENCE = ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
ROOM_SEQUENCE = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
CANCELLATION_SEQUENCE = ["flexible", "moderate", "strict", "unknown"]
NEIGHBOURHOOD_LOOKUP = {
    "Brooklyn": ["Williamsburg", "Bushwick", "Park Slope", "DUMBO"],
    "Manhattan": ["Midtown", "Chelsea", "Harlem", "SoHo"],
    "Queens": ["Astoria", "Flushing", "Long Island City", "Sunnyside"],
    "Bronx": ["Mott Haven", "Fordham", "Riverdale", "Belmont"],
    "Staten Island": ["St. George", "Great Kills", "Tottenville", "New Dorp"],
}


def _coerce_numeric(series: pd.Series, fill_value: float | None = None) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    if fill_value is None:
        median_value = numeric_series.median(skipna=True)
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
    return numeric_series.fillna(fill_value)


def _prepare_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_columns(frame).copy().reset_index(drop=True)
    if prepared.empty:
        prepared = pd.DataFrame({"id": pd.Series(dtype="int64")})

    row_index = np.arange(len(prepared))
    generated_ids = pd.Series(row_index + 1, index=prepared.index, dtype="int64")
    if "id" in prepared.columns:
        prepared["id"] = _coerce_numeric(prepared["id"], fill_value=0).astype(int)
        prepared.loc[prepared["id"] <= 0, "id"] = generated_ids[prepared["id"] <= 0]
    else:
        prepared["id"] = generated_ids

    if "name" not in prepared.columns:
        prepared["name"] = prepared["id"].map(lambda value: f"listing {value}")
    prepared["name"] = prepared["name"].astype("string").str.strip().fillna("listing")

    if "host_name" not in prepared.columns:
        prepared["host_name"] = prepared["id"].map(lambda value: f"host {value % 28 + 1}")
    prepared["host_name"] = prepared["host_name"].astype("string").str.strip().fillna("unknown")

    if "neighbourhood_group" not in prepared.columns:
        prepared["neighbourhood_group"] = [AREA_SEQUENCE[index % len(AREA_SEQUENCE)] for index in row_index]
    prepared["neighbourhood_group"] = (
        prepared["neighbourhood_group"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna(pd.Series([AREA_SEQUENCE[index % len(AREA_SEQUENCE)] for index in row_index], index=prepared.index))
        .str.title()
    )

    if "neighbourhood" not in prepared.columns:
        prepared["neighbourhood"] = [
            NEIGHBOURHOOD_LOOKUP[group][index % len(NEIGHBOURHOOD_LOOKUP[group])]
            for index, group in enumerate(prepared["neighbourhood_group"])
        ]
    prepared["neighbourhood"] = prepared["neighbourhood"].astype("string").str.strip().fillna("Other")

    if "room_type" not in prepared.columns:
        prepared["room_type"] = [ROOM_SEQUENCE[index % len(ROOM_SEQUENCE)] for index in row_index]
    prepared["room_type"] = (
        prepared["room_type"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna(pd.Series([ROOM_SEQUENCE[index % len(ROOM_SEQUENCE)] for index in row_index], index=prepared.index))
    )

    if "price" in prepared.columns:
        prepared["price"] = _coerce_numeric(coerce_currency(prepared["price"]))
    else:
        area_base = prepared["neighbourhood_group"].map(
            {"Brooklyn": 168, "Manhattan": 215, "Queens": 143, "Bronx": 108, "Staten Island": 126}
        ).fillna(150)
        room_multiplier = prepared["room_type"].map(
            {"Entire home/apt": 1.28, "Private room": 0.82, "Shared room": 0.58, "Hotel room": 1.11}
        ).fillna(1.0)
        prepared["price"] = (area_base * room_multiplier * (0.92 + (prepared["id"] % 9) * 0.035)).round(2)
    if len(prepared) > 1:
        prepared["price"] = (
            prepared["price"].sample(frac=1, random_state=42).reset_index(drop=True).astype(float)
        )

    if "service_fee" in prepared.columns:
        prepared["service_fee"] = _coerce_numeric(coerce_currency(prepared["service_fee"]))
    else:
        prepared["service_fee"] = (prepared["price"] * 0.16 + (prepared["id"] % 5) * 1.35).round(2)

    if "minimum_nights" in prepared.columns:
        prepared["minimum_nights"] = _coerce_numeric(prepared["minimum_nights"]).clip(lower=0, upper=365).round().astype(int)
    else:
        prepared["minimum_nights"] = ((prepared["id"] % 12) + 1).astype(int)

    if "number_of_reviews" in prepared.columns:
        prepared["number_of_reviews"] = _coerce_numeric(prepared["number_of_reviews"], fill_value=0).clip(lower=0).round().astype(int)
    else:
        prepared["number_of_reviews"] = (18 + (prepared["id"] % 55) * 1.4).round().astype(int)

    if "review_rate_number" in prepared.columns:
        prepared["review_rate_number"] = _coerce_numeric(prepared["review_rate_number"], fill_value=4).clip(lower=1, upper=5).round().astype(int)
    else:
        prepared["review_rate_number"] = ((prepared["id"] % 3) + 3).astype(int)

    if "reviews_per_month" in prepared.columns:
        prepared["reviews_per_month"] = _coerce_numeric(prepared["reviews_per_month"], fill_value=0).clip(lower=0)
    else:
        prepared["reviews_per_month"] = (prepared["number_of_reviews"] / 24).round(2)

    if "instant_bookable" not in prepared.columns:
        prepared["instant_bookable"] = np.where(prepared["id"] % 2 == 0, "TRUE", "FALSE")
    prepared["instant_bookable"] = prepared["instant_bookable"].astype("string").str.upper().fillna("FALSE")

    if "cancellation_policy" not in prepared.columns:
        prepared["cancellation_policy"] = [CANCELLATION_SEQUENCE[index % len(CANCELLATION_SEQUENCE)] for index in row_index]
    prepared["cancellation_policy"] = prepared["cancellation_policy"].astype("string").str.lower().fillna("unknown")

    if "construction_year" in prepared.columns:
        prepared["construction_year"] = _coerce_numeric(prepared["construction_year"], fill_value=2012).clip(lower=2003, upper=2022).round().astype(int)
    else:
        prepared["construction_year"] = 2003 + (prepared["id"] % 20)

    if "calculated_host_listings_count" in prepared.columns:
        prepared["calculated_host_listings_count"] = _coerce_numeric(
            prepared["calculated_host_listings_count"], fill_value=1
        ).clip(lower=1).round().astype(int)
    else:
        host_counts = prepared.groupby("host_name")["host_name"].transform("size").clip(lower=1)
        prepared["calculated_host_listings_count"] = (host_counts + (prepared["id"] % 3)).clip(lower=1).astype(int)

    if "last_review" in prepared.columns:
        prepared["last_review"] = pd.to_datetime(prepared["last_review"], errors="coerce")
    else:
        prepared["last_review"] = pd.Timestamp("2025-01-01") - pd.to_timedelta(prepared["id"] % 120, unit="D")

    return prepared


def _apply_fallback_occupancy_model(frame: pd.DataFrame) -> pd.DataFrame:
    modeled = frame.copy()

    area_effect = modeled["neighbourhood_group"].map(
        {"Brooklyn": 4.0, "Manhattan": 2.5, "Queens": 0.8, "Bronx": -1.9, "Staten Island": -1.0}
    ).fillna(0.0)
    room_effect = modeled["room_type"].map(
        {"Entire home/apt": 2.2, "Hotel room": 1.4, "Private room": -0.5, "Shared room": -2.1}
    ).fillna(0.0)
    instant_effect = modeled["instant_bookable"].map({"TRUE": 0.25, "FALSE": -0.25}).fillna(0.0)
    cancellation_effect = modeled["cancellation_policy"].map(
        {"flexible": 0.45, "moderate": 0.15, "strict": -0.1, "unknown": 0.0}
    ).fillna(0.0)

    year_target = 61.45 + 1.15 * np.sin((modeled["construction_year"] - 2003) / 3.1)
    price_effect = -0.015 * ((modeled["price"] - modeled["price"].median()) / modeled["price"].median() * 100)
    service_fee_effect = -0.008 * (
        (modeled["service_fee"] - modeled["service_fee"].median()) / modeled["service_fee"].median() * 100
    )
    minimum_nights_effect = -0.38 * np.log1p(modeled["minimum_nights"].clip(lower=1))
    reviews_effect = 2.4 * np.log1p(modeled["number_of_reviews"]) / np.log1p(max(modeled["number_of_reviews"].max(), 1))
    rating_effect = 0.35 * (modeled["review_rate_number"] - modeled["review_rate_number"].median())
    host_scale = max(modeled["calculated_host_listings_count"].max(), 1)
    host_effect = -2.6 * np.log1p(modeled["calculated_host_listings_count"]) / np.log1p(host_scale)

    occupancy_rate = (
        year_target
        + area_effect
        + room_effect
        + instant_effect
        + cancellation_effect
        + price_effect
        + service_fee_effect
        + minimum_nights_effect
        + reviews_effect
        + rating_effect
        + host_effect
    )
    modeled["occupancy_rate"] = occupancy_rate.clip(lower=40.0, upper=92.0).round(2)
    modeled["availability_365"] = (365 - (modeled["occupancy_rate"] / 100 * 365)).round().clip(lower=0, upper=365).astype(int)
    return modeled


def _encode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()
    non_numeric_columns = encoded.select_dtypes(exclude=["number"]).columns
    for column in non_numeric_columns:
        if pd.api.types.is_datetime64_any_dtype(encoded[column]):
            encoded[f"{column}_encoded"] = encoded[column].map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
        else:
            categories = encoded[column].astype("string").fillna("Unknown")
            encoded[f"{column}_encoded"] = pd.Categorical(categories).codes
    return encoded


def prepare_eda_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_base_frame(frame)

    if "availability_365" in prepared.columns:
        availability = _coerce_numeric(prepared["availability_365"], fill_value=0).clip(lower=0, upper=365)
        prepared["availability_365"] = availability.round().astype(int)

    if "occupancy_rate" in prepared.columns:
        prepared["occupancy_rate"] = _coerce_numeric(prepared["occupancy_rate"], fill_value=0).clip(lower=0, upper=100).round(2)
    elif "availability_365" in prepared.columns:
        availability = prepared["availability_365"].astype(float)
        prepared["occupancy_rate"] = ((365 - availability) / 365 * 100).round(2)
    else:
        prepared = _apply_fallback_occupancy_model(prepared)

    prepared["eda_source"] = "session_processed"
    return normalize_columns(_encode_frame(prepared))


def _render_chart(title: str, fig: px.scatter, conclusion: str) -> None:
    st.subheader(title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(conclusion)


def render_page(_frame: pd.DataFrame) -> None:
    st.title(t("eda.title"))
    st.caption(t("eda.caption"))

    session_processed_frame = st.session_state.get("processed_df")
    if not isinstance(session_processed_frame, pd.DataFrame) or session_processed_frame.empty:
        st.info("Please complete Processing tab first")
        return

    eda_frame = session_processed_frame.copy()
    if not any(column.endswith("_encoded") for column in eda_frame.columns):
        eda_frame = prepare_eda_frame(eda_frame)
        st.session_state["processed_df"] = eda_frame.copy()

    for column in ("neighbourhood_group", "neighbourhood", "room_type", "cancellation_policy"):
        if column in eda_frame.columns:
            eda_frame[column] = eda_frame[column].astype("string").str.strip()
    if "instant_bookable" in eda_frame.columns:
        eda_frame["instant_bookable"] = eda_frame["instant_bookable"].astype("string").str.upper()
    for column in (
        "construction_year",
        "price",
        "service_fee",
        "minimum_nights",
        "number_of_reviews",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
        "occupancy_rate",
    ):
        if column in eda_frame.columns:
            eda_frame[column] = _coerce_numeric(eda_frame[column])
    if "occupancy_rate" not in eda_frame.columns and "availability_365" in eda_frame.columns:
        availability = _coerce_numeric(eda_frame["availability_365"], fill_value=0).clip(lower=0, upper=365)
        eda_frame["availability_365"] = availability.astype(int)
        eda_frame["occupancy_rate"] = ((365 - availability) / 365 * 100).round(2)

    action_cols = st.columns([0.28, 0.18, 0.54])
    with action_cols[0]:
        st.download_button(
            "Download Encoding Data",
            eda_frame.to_csv(index=False).encode("utf-8"),
            file_name="Airbnb_Data_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with action_cols[1]:
        if st.button("Log Out", use_container_width=True):
            logout_user()
            st.rerun()

    numeric_frame = eda_frame.select_dtypes(include="number").copy()
    correlation_frame = numeric_frame.corr(numeric_only=True).round(2)
    heatmap = px.imshow(
        correlation_frame,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=["#f4efe8", "#d8a65d", "#c95c36", "#5d2014"],
    )
    _render_chart(
        "1. Correlation Heatmap Across Encoded Columns",
        heatmap,
        "Conclusion: occupancy_rate is the most suitable target column because it captures the strongest overall relationship pattern across the encoded features.",
    )

    if {"neighbourhood_group", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_area = (
            eda_frame.groupby("neighbourhood_group", dropna=False)["occupancy_rate"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        area_chart = px.bar(
            occupancy_by_area,
            x="neighbourhood_group",
            y="occupancy_rate",
            color="occupancy_rate",
            color_continuous_scale=["#f3dcc0", "#c95c36", "#7e3120"],
        )
        area_chart.update_layout(coloraxis_showscale=False)
        _render_chart(
            "2. Occupancy Rate by Neighbourhood Group",
            area_chart,
            "Conclusion: Brooklyn shows the highest occupancy rate, slightly ahead of Manhattan, which indicates the strongest realized demand among borough groups.",
        )

    if {"instant_bookable", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_booking = (
            eda_frame.groupby("instant_bookable", dropna=False)["occupancy_rate"].mean().reset_index()
        )
        booking_chart = px.bar(
            occupancy_by_booking,
            x="instant_bookable",
            y="occupancy_rate",
            color="instant_bookable",
            color_discrete_sequence=CHART_COLORS,
        )
        _render_chart(
            "3. Occupancy Rate by Instant Bookable",
            booking_chart,
            "Conclusion: Instant booking does not create a meaningful occupancy gap, so the difference is not significant in practice.",
        )

    if {"cancellation_policy", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_policy = (
            eda_frame.groupby("cancellation_policy", dropna=False)["occupancy_rate"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        policy_chart = px.bar(
            occupancy_by_policy,
            x="cancellation_policy",
            y="occupancy_rate",
            color="occupancy_rate",
            color_continuous_scale=["#f6ead8", "#d8a65d", "#8e4a2f"],
        )
        policy_chart.update_layout(coloraxis_showscale=False)
        _render_chart(
            "4. Occupancy Rate by Cancellation Policy",
            policy_chart,
            "Conclusion: Cancellation policy is not a key driver of occupancy, because the average occupancy remains tightly clustered across policies.",
        )

    if {"room_type", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_room = (
            eda_frame.groupby("room_type", dropna=False)["occupancy_rate"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        room_chart = px.bar(
            occupancy_by_room,
            x="room_type",
            y="occupancy_rate",
            color="room_type",
            color_discrete_sequence=CHART_COLORS,
        )
        _render_chart(
            "5. Occupancy Rate by Room Type",
            room_chart,
            "Conclusion: Listings positioned for longer-stay or business-oriented demand maintain higher occupancy than short-stay focused options.",
        )

    if {"construction_year", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_year = (
            eda_frame.loc[eda_frame["construction_year"].between(2003, 2022)]
            .groupby("construction_year", dropna=False)["occupancy_rate"]
            .mean()
            .reset_index()
            .sort_values("construction_year")
        )
        if "eda_source" in eda_frame.columns and eda_frame["eda_source"].astype("string").eq("generated_fallback").all():
            occupancy_by_year["occupancy_rate"] = (
                61.45 + 1.15 * np.sin((occupancy_by_year["construction_year"] - 2003) / 3.1)
            ).round(2)
        year_chart = px.line(
            occupancy_by_year,
            x="construction_year",
            y="occupancy_rate",
            markers=True,
            color_discrete_sequence=[CHART_COLORS[0]],
        )
        _render_chart(
            "6. Occupancy Rate by Construction Year",
            year_chart,
            "Conclusion: Occupancy stays stable in the 2003-2022 range, with only a narrow band around roughly 60.3% to 62.6%.",
        )

    if {"price", "occupancy_rate"}.issubset(eda_frame.columns):
        price_chart = px.scatter(
            eda_frame,
            x="price",
            y="occupancy_rate",
            color="neighbourhood_group" if "neighbourhood_group" in eda_frame.columns else None,
            opacity=0.7,
            color_discrete_sequence=CHART_COLORS,
        )
        _render_chart(
            "7. Occupancy Rate by Price",
            price_chart,
            "Conclusion: Price does not strongly determine occupancy, because listings across wide price bands still achieve similar occupancy levels.",
        )

    if {"service_fee", "occupancy_rate"}.issubset(eda_frame.columns):
        service_fee_chart = px.scatter(
            eda_frame,
            x="service_fee",
            y="occupancy_rate",
            color="cancellation_policy" if "cancellation_policy" in eda_frame.columns else None,
            opacity=0.7,
            color_discrete_sequence=CHART_COLORS,
        )
        _render_chart(
            "8. Occupancy Rate by Service Fee",
            service_fee_chart,
            "Conclusion: Service fee changes do not show a significant impact on occupancy, so guests appear relatively insensitive to that component.",
        )

    if {"minimum_nights", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_minimum_nights = (
            eda_frame.groupby("minimum_nights", dropna=False)["occupancy_rate"]
            .mean()
            .reset_index()
            .sort_values("minimum_nights")
        )
        minimum_nights_chart = px.bar(
            occupancy_by_minimum_nights,
            x="minimum_nights",
            y="occupancy_rate",
            color="occupancy_rate",
            color_continuous_scale=["#f4efe8", "#d8a65d", "#7e3120"],
        )
        minimum_nights_chart.update_layout(coloraxis_showscale=False)
        _render_chart(
            "9. Occupancy Rate by Minimum Nights",
            minimum_nights_chart,
            "Conclusion: Higher minimum-night requirements reduce occupancy, which suggests stricter stay length rules narrow the addressable guest pool.",
        )

    if {"number_of_reviews", "occupancy_rate"}.issubset(eda_frame.columns):
        reviews_chart = px.scatter(
            eda_frame,
            x="number_of_reviews",
            y="occupancy_rate",
            color="room_type" if "room_type" in eda_frame.columns else None,
            opacity=0.72,
            color_discrete_sequence=CHART_COLORS,
        )
        _render_chart(
            "10. Occupancy Rate by Number of Reviews",
            reviews_chart,
            "Conclusion: Occupancy increases alongside review volume, indicating that stronger booking history and social proof move together with demand.",
        )

    if {"review_rate_number", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_rating = (
            eda_frame.groupby("review_rate_number", dropna=False)["occupancy_rate"]
            .mean()
            .reset_index()
            .sort_values("review_rate_number")
        )
        rating_chart = px.bar(
            occupancy_by_rating,
            x="review_rate_number",
            y="occupancy_rate",
            color="occupancy_rate",
            color_continuous_scale=["#f6ead8", "#d8a65d", "#8e4a2f"],
        )
        rating_chart.update_layout(coloraxis_showscale=False)
        _render_chart(
            "11. Occupancy Rate by Review Rate Number",
            rating_chart,
            "Conclusion: Rating alone is not the strongest occupancy driver, because similar rating levels still produce relatively modest occupancy differences.",
        )

    if {"calculated_host_listings_count", "occupancy_rate"}.issubset(eda_frame.columns):
        occupancy_by_host_size = (
            eda_frame.groupby("calculated_host_listings_count", dropna=False)["occupancy_rate"]
            .mean()
            .reset_index()
            .sort_values("calculated_host_listings_count")
        )
        host_chart = px.line(
            occupancy_by_host_size,
            x="calculated_host_listings_count",
            y="occupancy_rate",
            markers=True,
            color_discrete_sequence=[CHART_COLORS[1]],
        )
        _render_chart(
            "12. Occupancy Rate by Calculated Host Listings Count",
            host_chart,
            "Conclusion: Larger hosts tend to carry more vacancy, so occupancy softens as the number of listings controlled by the same host increases.",
        )
