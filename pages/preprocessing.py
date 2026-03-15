from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CHART_COLORS
from core.data import coerce_currency, normalize_columns
from core.i18n import localize_dataframe_for_display, t
from pages.data_raw import build_missing_table, build_numeric_profile_frame, inject_page_navigation

NEIGHBOURHOOD_GROUP_FIXES = {
    "brookln": "brooklyn",
    "brookyn": "brooklyn",
    "manhatan": "manhattan",
    "manhatten": "manhattan",
}
NEIGHBOURHOOD_GROUP_DUMMY_COLUMNS = ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island", "Other"]
ROOM_TYPE_DUMMY_COLUMNS = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
CANCELLATION_POLICY_ENCODING = {"flexible": 0, "moderate": 1, "strict": 2, "unknown": -1}
ROOM_TYPE_CANONICAL_MAP = {
    "Entire Home/Apt": "Entire home/apt",
    "Private Room": "Private room",
    "Shared Room": "Shared room",
    "Hotel Room": "Hotel room",
}


def _clean_string(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _fill_with_median(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    median_value = series.median(skipna=True)
    return series.fillna(fallback if pd.isna(median_value) else median_value)


def _clean_listing_name(series: pd.Series) -> pd.Series:
    cleaned = _clean_string(series)
    cleaned = (
        cleaned.str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    return cleaned.replace({"": pd.NA})


def _clean_host_name(series: pd.Series) -> pd.Series:
    return _clean_string(series).str.lower().fillna("unknown")


def _clean_neighbourhood_group(series: pd.Series) -> pd.Series:
    cleaned = _clean_string(series).str.lower().replace(NEIGHBOURHOOD_GROUP_FIXES)
    return cleaned.fillna("other").str.title()


def _add_encoded_columns(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()

    verified_series = pd.Series("unconfirmed", index=encoded.index, dtype="string")
    if "host_identity_verified" in encoded.columns:
        verified_series = _clean_string(encoded["host_identity_verified"]).str.lower().fillna("unconfirmed")
    encoded["host_identity_verified_encoded"] = verified_series.eq("verified").astype("Int64")

    instant_series = pd.Series("FALSE", index=encoded.index, dtype="string")
    if "instant_bookable" in encoded.columns:
        instant_series = _clean_string(encoded["instant_bookable"]).str.upper().fillna("FALSE")
    encoded["instant_bookable_encoded"] = instant_series.map({"TRUE": 1, "FALSE": 0}).fillna(0).astype("Int64")

    group_dummies = pd.DataFrame(0, index=encoded.index, columns=NEIGHBOURHOOD_GROUP_DUMMY_COLUMNS, dtype="Int64")
    if "neighbourhood_group" in encoded.columns:
        group_series = _clean_neighbourhood_group(encoded["neighbourhood_group"])
        current_group_dummies = pd.get_dummies(group_series).reindex(columns=NEIGHBOURHOOD_GROUP_DUMMY_COLUMNS, fill_value=0)
        group_dummies = current_group_dummies.astype("Int64")
    encoded = pd.concat([encoded, group_dummies], axis=1)

    room_dummies = pd.DataFrame(0, index=encoded.index, columns=ROOM_TYPE_DUMMY_COLUMNS, dtype="Int64")
    if "room_type" in encoded.columns:
        room_series = _clean_string(encoded["room_type"]).str.title().replace(ROOM_TYPE_CANONICAL_MAP)
        encoded["room_type"] = room_series.fillna("Private room")
        current_room_dummies = pd.get_dummies(encoded["room_type"]).reindex(columns=ROOM_TYPE_DUMMY_COLUMNS, fill_value=0)
        room_dummies = current_room_dummies.astype("Int64")
    encoded = pd.concat([encoded, room_dummies], axis=1)

    policy_series = pd.Series("unknown", index=encoded.index, dtype="string")
    if "cancellation_policy" in encoded.columns:
        policy_series = _clean_string(encoded["cancellation_policy"]).str.lower().fillna("unknown")
    encoded["cancellation_policy_encoded"] = policy_series.map(CANCELLATION_POLICY_ENCODING).fillna(-1).astype("Int64")

    neighbourhood_series = pd.Series("Other", index=encoded.index, dtype="string")
    if "neighbourhood" in encoded.columns:
        neighbourhood_series = _clean_string(encoded["neighbourhood"]).fillna("Other")
    encoded["neighbourhood_encoded"] = pd.Series(
        pd.Categorical(neighbourhood_series).codes,
        index=encoded.index,
        dtype="Int64",
    )

    return encoded


def _prepare_null_comparison(before_frame: pd.DataFrame, after_frame: pd.DataFrame) -> pd.DataFrame:
    before_health = build_missing_table(before_frame)[["column name", "null count"]].rename(
        columns={"null count": "Before Processing"}
    )
    after_health = build_missing_table(after_frame)[["column name", "null count"]].rename(
        columns={"null count": "After Processing"}
    )
    comparison = before_health.merge(after_health, on="column name", how="outer").fillna(0)
    comparison["max_null"] = comparison[["Before Processing", "After Processing"]].max(axis=1)
    comparison = comparison.sort_values(["max_null", "column name"], ascending=[False, True])
    long_frame = comparison.drop(columns="max_null").melt(
        id_vars="column name",
        var_name="stage",
        value_name="null count",
    )
    long_frame["column name"] = pd.Categorical(
        long_frame["column name"],
        categories=comparison["column name"].tolist()[::-1],
        ordered=True,
    )
    return long_frame


def _prepare_boxplot_comparison(before_frame: pd.DataFrame, after_frame: pd.DataFrame) -> pd.DataFrame:
    stages: list[pd.DataFrame] = []
    for stage_name, frame in (("Before Processing", before_frame), ("After Processing", after_frame)):
        numeric_profile = build_numeric_profile_frame(frame)
        if numeric_profile.empty:
            continue
        melted = numeric_profile.melt(var_name="column name", value_name="value").dropna()
        if melted.empty:
            continue
        stages.append(melted.assign(stage=stage_name))
    if not stages:
        return pd.DataFrame(columns=["column name", "value", "stage"])
    return pd.concat(stages, ignore_index=True)


@st.cache_data(show_spinner=False)
def run_processing_pipeline(
    raw_frame: pd.DataFrame,
    fallback_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    normalized = normalize_columns(raw_frame)
    before_frame = normalized.copy()
    processed = normalized.copy()
    fallback_processed = normalize_columns(fallback_frame).copy() if fallback_frame is not None else normalized.copy()

    processed = processed.drop(columns=["country", "country_code", "house_rules", "license"], errors="ignore")

    missing_coordinate_rows = 0
    if {"lat", "long"}.issubset(processed.columns):
        processed["lat"] = pd.to_numeric(processed["lat"], errors="coerce")
        processed["long"] = pd.to_numeric(processed["long"], errors="coerce")
        rows_before_coordinates = len(processed)
        processed = processed.loc[processed["lat"].notna() & processed["long"].notna()].copy()
        missing_coordinate_rows = rows_before_coordinates - len(processed)

    if "id" in processed.columns:
        processed["id"] = _clean_string(processed["id"]).astype("string")
    if "name" in processed.columns:
        processed["name"] = _clean_listing_name(processed["name"])
    if "host_id" in processed.columns:
        processed["host_id"] = _clean_string(processed["host_id"]).astype("string")
    if "host_identity_verified" in processed.columns:
        processed["host_identity_verified"] = _clean_string(processed["host_identity_verified"]).fillna("unconfirmed")
    if "host_name" in processed.columns:
        processed["host_name"] = _clean_host_name(processed["host_name"])
    if "neighbourhood_group" in processed.columns:
        processed["neighbourhood_group"] = _clean_neighbourhood_group(processed["neighbourhood_group"])
    if "neighbourhood" in processed.columns:
        processed["neighbourhood"] = _clean_string(processed["neighbourhood"]).fillna("Other")
    if "instant_bookable" in processed.columns:
        processed["instant_bookable"] = _clean_string(processed["instant_bookable"]).str.upper().fillna("FALSE")
    if "cancellation_policy" in processed.columns:
        processed["cancellation_policy"] = _clean_string(processed["cancellation_policy"]).str.lower().fillna("unknown")
    if "room_type" in processed.columns:
        processed["room_type"] = _clean_string(processed["room_type"]).str.title().replace(ROOM_TYPE_CANONICAL_MAP)
    if "construction_year" in processed.columns:
        construction_year = pd.to_numeric(processed["construction_year"], errors="coerce")
        processed["construction_year"] = _fill_with_median(construction_year).round().astype("Int64")
    if "price" in processed.columns:
        price = _fill_with_median(coerce_currency(processed["price"]))
        processed["price"] = price.astype(float)
    if "service_fee" in processed.columns:
        service_fee = _fill_with_median(coerce_currency(processed["service_fee"]))
        processed["service_fee"] = service_fee.astype(float)
    if "minimum_nights" in processed.columns:
        minimum_nights = pd.to_numeric(processed["minimum_nights"], errors="coerce")
        minimum_nights = minimum_nights.where(minimum_nights.between(0, 365))
        processed["minimum_nights"] = _fill_with_median(minimum_nights).round().astype("Int64")
    if "number_of_reviews" in processed.columns:
        number_of_reviews = pd.to_numeric(processed["number_of_reviews"], errors="coerce").fillna(0)
        processed["number_of_reviews"] = number_of_reviews.round().astype("Int64")
    if "last_review" in processed.columns:
        processed["last_review"] = pd.to_datetime(processed["last_review"], errors="coerce")
    if "reviews_per_month" in processed.columns:
        reviews_per_month = pd.to_numeric(processed["reviews_per_month"], errors="coerce").fillna(0)
        processed["reviews_per_month"] = reviews_per_month.astype(float)
    if "review_rate_number" in processed.columns:
        review_rate_number = pd.to_numeric(processed["review_rate_number"], errors="coerce")
        processed["review_rate_number"] = _fill_with_median(review_rate_number).round().astype("Int64")
    if "calculated_host_listings_count" in processed.columns:
        host_listing_count = pd.to_numeric(processed["calculated_host_listings_count"], errors="coerce").fillna(1)
        processed["calculated_host_listings_count"] = host_listing_count.round().astype("Int64")
    if "availability_365" in processed.columns:
        availability = pd.to_numeric(processed["availability_365"], errors="coerce").clip(lower=0, upper=365).fillna(0)
        processed["availability_365"] = availability.round().astype("Int64")
        processed["occupancy_rate"] = ((365 - processed["availability_365"].astype(float)) / 365) * 100

    duplicates_removed = 0
    if "id" in processed.columns:
        rows_before_dedup = len(processed)
        processed = processed.drop_duplicates(subset=["id"], keep="first").copy()
        duplicates_removed = rows_before_dedup - len(processed)
    else:
        rows_before_dedup = len(processed)
        processed = processed.drop_duplicates().copy()
        duplicates_removed = rows_before_dedup - len(processed)

    processed = processed.reset_index(drop=True)
    if processed.empty:
        processed = fallback_processed.copy()

    processed = _add_encoded_columns(processed)

    report = {
        "rows_before": len(before_frame),
        "rows_after": len(processed),
        "columns_after": processed.shape[1],
        "duplicates_removed": duplicates_removed,
        "missing_coordinate_rows": missing_coordinate_rows,
    }
    return before_frame, processed, report


def render_page(raw_frame: pd.DataFrame, cleaned_frame: pd.DataFrame, _report: dict[str, object]) -> None:
    st.title(t("prep.title"))
    st.caption(t("prep.caption"))

    session_raw_frame = st.session_state.get("raw_df")
    uploaded_file_name = st.session_state.get("raw_df_name")
    has_uploaded_raw = isinstance(session_raw_frame, pd.DataFrame) and bool(uploaded_file_name)

    run_preprocessing = st.button(
        "Run Preprocessing",
        type="primary",
        use_container_width=True,
        disabled=not has_uploaded_raw,
    )

    if not has_uploaded_raw:
        st.warning("Please upload a CSV file in the Data Raw tab first")
        return

    stored_processed_frame = st.session_state.get("processed_df")
    stored_before_frame = st.session_state.get("preprocessing_before_df")
    stored_processing_report = st.session_state.get("processing_report")
    has_processed_data = (
        isinstance(stored_processed_frame, pd.DataFrame)
        and isinstance(stored_before_frame, pd.DataFrame)
        and isinstance(stored_processing_report, dict)
    )
    if not run_preprocessing and not has_processed_data:
        st.info("Click 'Run Preprocessing' to clean the raw data")
        return

    if run_preprocessing:
        source_frame = session_raw_frame.copy()
        before_frame, processed_frame, processing_report = run_processing_pipeline(source_frame)
        st.session_state["preprocessing_before_df"] = before_frame.copy()
        st.session_state["processing_report"] = processing_report
        st.session_state["processed_df"] = processed_frame.copy()
        stored_before_frame = before_frame
        stored_processed_frame = processed_frame
        stored_processing_report = processing_report
        st.success("Data cleaned successfully!")
    else:
        before_frame = stored_before_frame.copy()
        processed_frame = stored_processed_frame.copy()
        processing_report = stored_processing_report

    processed_csv = processed_frame.to_csv(index=False).encode("utf-8")

    action_cols = st.columns([0.2, 0.3, 0.5])
    with action_cols[0]:
        if st.button("EDA Data", type="primary", use_container_width=True):
            inject_page_navigation("eda")
    with action_cols[1]:
        st.download_button(
            "Download Processed Data",
            processed_csv,
            file_name="airbnb_processed_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    size_metrics = st.columns(2)
    size_metrics[0].metric("Processed Records", f"{processing_report['rows_after']:,}")
    size_metrics[1].metric("Processed Columns", f"{processing_report['columns_after']:,}")

    metrics = st.columns(4)
    metrics[0].metric(t("prep.metric.rows_before"), f"{processing_report['rows_before']:,}")
    metrics[1].metric(t("prep.metric.rows_after"), f"{processing_report['rows_after']:,}")
    metrics[2].metric(t("prep.metric.duplicates_removed"), f"{processing_report['duplicates_removed']:,}")
    metrics[3].metric("Rows Removed (Lat/Long Null)", f"{processing_report['missing_coordinate_rows']:,}")

    st.markdown(
        f"""
        <div class="surface-card">
            <strong>{t("prep.workflow_title")}</strong>
            <p class="hint-text">
                {t("prep.workflow_body")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Health of Processed Data")
    processed_health = build_missing_table(processed_frame)
    st.dataframe(
        localize_dataframe_for_display(processed_health),
        use_container_width=True,
        hide_index=True,
        height=620,
    )

    null_comparison = _prepare_null_comparison(before_frame, processed_frame)
    null_chart = px.bar(
        null_comparison,
        x="null count",
        y="column name",
        color="stage",
        orientation="h",
        barmode="group",
        title="Null Values Before vs After Processing",
        color_discrete_sequence=[CHART_COLORS[1], CHART_COLORS[0]],
    )
    null_chart.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(null_chart, use_container_width=True)

    boxplot_comparison = _prepare_boxplot_comparison(before_frame, processed_frame)
    if not boxplot_comparison.empty:
        boxplot = px.box(
            boxplot_comparison,
            x="value",
            y="column name",
            color="stage",
            orientation="h",
            title="Numerical Columns Before vs After Processing",
            color_discrete_sequence=[CHART_COLORS[1], CHART_COLORS[0]],
        )
        boxplot.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            legend_title_text="",
            height=max(420, boxplot_comparison["column name"].nunique() * 70),
        )
        st.plotly_chart(boxplot, use_container_width=True)

    sample_col, type_col = st.columns(2, gap="large")
    with sample_col:
        st.subheader(t("prep.preview"))
        st.dataframe(localize_dataframe_for_display(processed_frame.head(20)), use_container_width=True, height=320)

    with type_col:
        dtypes_frame = processed_frame.dtypes.reset_index()
        dtypes_frame.columns = ["column", "dtype"]
        st.subheader(t("prep.schema"))
        st.dataframe(localize_dataframe_for_display(dtypes_frame), use_container_width=True, hide_index=True, height=320)
