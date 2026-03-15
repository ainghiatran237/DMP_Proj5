from __future__ import annotations

import json
from textwrap import dedent

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from core.config import CHART_COLORS
from core.i18n import localize_dataframe_for_display, nav_label, t, translate_room_type

COLUMN_REFERENCE_ROWS = [
    {"column": "id", "original_dtype": "object", "meaning": "Unique listing identifier", "handling": "Keep as unique ID."},
    {"column": "NAME", "original_dtype": "object", "meaning": "Listing title shown to users", "handling": "Check missing values and keep as text."},
    {"column": "host id", "original_dtype": "object", "meaning": "Unique host identifier", "handling": "Keep as unique ID."},
    {"column": "host_identity_verified", "original_dtype": "object", "meaning": "Whether the host identity is verified", "handling": "Convert True/False to 1/0."},
    {"column": "host name", "original_dtype": "object", "meaning": "Host display name", "handling": "Check missing values and keep as text."},
    {"column": "neighbourhood group", "original_dtype": "object", "meaning": "NYC borough or high-level area", "handling": "Fix typos and map invalid values to 'Unknown'."},
    {"column": "neighbourhood", "original_dtype": "object", "meaning": "Local neighborhood name", "handling": "Check missing values and keep as categorical text."},
    {"column": "lat", "original_dtype": "object", "meaning": "Latitude coordinate", "handling": "Convert to float64 and validate NYC range."},
    {"column": "long", "original_dtype": "object", "meaning": "Longitude coordinate", "handling": "Convert to float64 and validate NYC range."},
    {"column": "country", "original_dtype": "object", "meaning": "Country name", "handling": "Standardize country naming."},
    {"column": "country code", "original_dtype": "object", "meaning": "Country code", "handling": "Validate allowed values."},
    {"column": "instant_bookable", "original_dtype": "object", "meaning": "Whether guests can book instantly", "handling": "Convert True/False to 1/0."},
    {"column": "cancellation_policy", "original_dtype": "object", "meaning": "Cancellation rule for the listing", "handling": "Fill missing values."},
    {"column": "room type", "original_dtype": "object", "meaning": "Accommodation type", "handling": "Encode as category or label values for ML."},
    {"column": "Construction year", "original_dtype": "object", "meaning": "Year the property was built", "handling": "Convert to int64."},
    {"column": "price", "original_dtype": "object", "meaning": "Listing price", "handling": "Clean currency, remove outliers, and add log transform."},
    {"column": "service fee", "original_dtype": "object", "meaning": "Platform or service fee", "handling": "Clean currency and store as nullable Int64."},
    {"column": "minimum nights", "original_dtype": "int64", "meaning": "Minimum nights required to book", "handling": "Replace values <= 0 and fill with median."},
    {"column": "number of reviews", "original_dtype": "int64", "meaning": "Total review count", "handling": "Fill missing with 0 or mean."},
    {"column": "last review", "original_dtype": "object", "meaning": "Date of the latest review", "handling": "Convert with pd.to_datetime."},
    {"column": "reviews per month", "original_dtype": "float64", "meaning": "Review frequency per month", "handling": "Fill missing with 0 or mean."},
    {"column": "review rate number", "original_dtype": "int64", "meaning": "Review score or rating number", "handling": "Fill missing with 0 or mean."},
    {"column": "calculated host listings count", "original_dtype": "int64", "meaning": "Listings managed by the same host", "handling": "Handle missing values."},
    {"column": "availability 365", "original_dtype": "int64", "meaning": "Available days in one year", "handling": "Limit to 0-365 and fill missing with 365."},
    {"column": "house_rules", "original_dtype": "object", "meaning": "Free-text house rules", "handling": "Drop as long text."},
    {"column": "license", "original_dtype": "object", "meaning": "License information", "handling": "Drop because the field has no usable data."},
]

PREPROCESSING_PIPELINE_STEPS = [
    {
        "title": "Step 1. Remove columns",
        "code": dedent(
            """
            df = df.drop(columns=["country", "country code", "house_rules", "license"], errors="ignore")
            """
        ).strip(),
    },
    {
        "title": "Step 2. Remove rows where `lat` or `long` is null",
        "code": dedent(
            """
            df = df.dropna(subset=["lat", "long"])
            """
        ).strip(),
    },
    {
        "title": "Step 3. `id` - convert to string and check duplicates",
        "code": dedent(
            """
            df["id"] = df["id"].astype("string")
            duplicate_ids = df["id"].duplicated().sum()
            """
        ).strip(),
    },
    {
        "title": "Step 4. `NAME` - strip whitespace, remove emoji/special chars, lowercase",
        "code": dedent(
            """
            df["NAME"] = (
                df["NAME"].astype("string")
                .str.strip()
                .str.replace(r"[^\\w\\s]", "", regex=True)
                .str.lower()
            )
            """
        ).strip(),
    },
    {
        "title": "Step 5. `host id` - convert to object",
        "code": dedent(
            """
            df["host id"] = df["host id"].astype("object")
            """
        ).strip(),
    },
    {
        "title": "Step 6. `host_identity_verified` - fill null = `unconfirmed`",
        "code": dedent(
            """
            df["host_identity_verified"] = df["host_identity_verified"].fillna("unconfirmed")
            """
        ).strip(),
    },
    {
        "title": "Step 7. `host name` - strip, fill null = `unknown`, lowercase",
        "code": dedent(
            """
            df["host name"] = (
                df["host name"].astype("string")
                .str.strip()
                .fillna("unknown")
                .str.lower()
            )
            """
        ).strip(),
    },
    {
        "title": "Step 8. `neighbourhood group` - strip, fix typos, fill null = `Other`",
        "code": dedent(
            """
            replacements = {
                "brookln": "brooklyn",
                "brookyn": "brooklyn",
                "manhatan": "manhattan",
                "manhatten": "manhattan",
            }
            df["neighbourhood group"] = (
                df["neighbourhood group"].astype("string")
                .str.strip()
                .str.lower()
                .replace(replacements)
                .fillna("other")
            )
            """
        ).strip(),
    },
    {
        "title": "Step 9. `neighbourhood` - fill null = `Other`",
        "code": dedent(
            """
            df["neighbourhood"] = df["neighbourhood"].fillna("Other")
            """
        ).strip(),
    },
    {
        "title": "Step 10. `instant_bookable` - fill null = `FALSE`",
        "code": dedent(
            """
            df["instant_bookable"] = df["instant_bookable"].fillna("FALSE")
            """
        ).strip(),
    },
    {
        "title": "Step 11. `cancellation_policy` - fill null = `unknown`",
        "code": dedent(
            """
            df["cancellation_policy"] = df["cancellation_policy"].fillna("unknown")
            """
        ).strip(),
    },
    {
        "title": "Step 12. `room type` - strip and Title Case",
        "code": dedent(
            """
            df["room type"] = df["room type"].astype("string").str.strip().str.title()
            """
        ).strip(),
    },
    {
        "title": "Step 13. `Construction year` - convert to int and fill null = median",
        "code": dedent(
            """
            construction_year = pd.to_numeric(df["Construction year"], errors="coerce")
            df["Construction year"] = construction_year.fillna(construction_year.median()).round().astype("Int64")
            """
        ).strip(),
    },
    {
        "title": "Step 14. `price` - remove `$` and comma, convert float64, fill null = median",
        "code": dedent(
            """
            price = (
                df["price"].astype("string")
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            price = pd.to_numeric(price, errors="coerce")
            df["price"] = price.fillna(price.median()).astype("float64")
            """
        ).strip(),
    },
    {
        "title": "Step 15. `service fee` - remove `$` and comma, convert float64, fill null = median",
        "code": dedent(
            """
            service_fee = (
                df["service fee"].astype("string")
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            service_fee = pd.to_numeric(service_fee, errors="coerce")
            df["service fee"] = service_fee.fillna(service_fee.median()).astype("float64")
            """
        ).strip(),
    },
    {
        "title": "Step 16. `minimum nights` - remove values > 365 or < 0, fill null = median",
        "code": dedent(
            """
            minimum_nights = pd.to_numeric(df["minimum nights"], errors="coerce")
            minimum_nights = minimum_nights.where(minimum_nights.between(0, 365))
            df["minimum nights"] = minimum_nights.fillna(minimum_nights.median())
            """
        ).strip(),
    },
    {
        "title": "Step 17. `number of reviews` - fill null = 0",
        "code": dedent(
            """
            df["number of reviews"] = pd.to_numeric(df["number of reviews"], errors="coerce").fillna(0)
            """
        ).strip(),
    },
    {
        "title": "Step 18. `last review` - convert to datetime, null = empty",
        "code": dedent(
            """
            df["last review"] = pd.to_datetime(df["last review"], errors="coerce")
            df["last review"] = df["last review"].dt.strftime("%Y-%m-%d").fillna("")
            """
        ).strip(),
    },
    {
        "title": "Step 19. `reviews per month` - fill null = 0",
        "code": dedent(
            """
            df["reviews per month"] = pd.to_numeric(df["reviews per month"], errors="coerce").fillna(0)
            """
        ).strip(),
    },
    {
        "title": "Step 20. `review rate number` - fill null = median",
        "code": dedent(
            """
            review_rate = pd.to_numeric(df["review rate number"], errors="coerce")
            df["review rate number"] = review_rate.fillna(review_rate.median())
            """
        ).strip(),
    },
    {
        "title": "Step 21. `calculated host listings count` - fill null = 1",
        "code": dedent(
            """
            host_listings = pd.to_numeric(df["calculated host listings count"], errors="coerce")
            df["calculated host listings count"] = host_listings.fillna(1)
            """
        ).strip(),
    },
    {
        "title": "Step 22. `availability 365` - clamp [0, 365], fill null = 0",
        "code": dedent(
            """
            availability = pd.to_numeric(df["availability 365"], errors="coerce").clip(lower=0, upper=365)
            df["availability 365"] = availability.fillna(0)
            """
        ).strip(),
    },
    {
        "title": "Step 23. Feature Engineering - create `occupancy_rate`",
        "code": dedent(
            """
            df["occupancy_rate"] = ((365 - df["availability 365"]) / 365) * 100
            """
        ).strip(),
    },
]


def _coerce_numeric_series(column_name: str, series: pd.Series) -> pd.Series | None:
    if column_name in {"id", "host id", "host_id"} or pd.api.types.is_bool_dtype(series):
        return None

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
    else:
        normalized = (
            series.astype("string")
            .str.strip()
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        numeric_series = pd.to_numeric(normalized, errors="coerce")

    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return None

    if numeric_series.notna().sum() / non_null_count < 0.8:
        return None

    return numeric_series


def build_numeric_profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns: dict[str, pd.Series] = {}
    for column in frame.columns:
        numeric_series = _coerce_numeric_series(column, frame[column])
        if numeric_series is not None and numeric_series.notna().any():
            numeric_columns[column] = numeric_series
    return pd.DataFrame(numeric_columns, index=frame.index)


def build_missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_profile = build_numeric_profile_frame(frame)
    health_rows: list[dict[str, object]] = []

    for column in frame.columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        row: dict[str, object] = {
            "column name": column,
            "data type": str(series.dtype),
            "null count": null_count,
            "null %": round((null_count / len(frame) * 100), 2) if len(frame) else 0.0,
            "min": pd.NA,
            "max": pd.NA,
            "mean": pd.NA,
            "median": pd.NA,
            "std": pd.NA,
        }

        if column in numeric_profile.columns:
            numeric_series = numeric_profile[column].dropna()
            if not numeric_series.empty:
                row.update(
                    {
                        "min": round(float(numeric_series.min()), 2),
                        "max": round(float(numeric_series.max()), 2),
                        "mean": round(float(numeric_series.mean()), 2),
                        "median": round(float(numeric_series.median()), 2),
                        "std": round(float(numeric_series.std()), 2) if len(numeric_series) > 1 else 0.0,
                    }
                )

        health_rows.append(row)

    return pd.DataFrame(health_rows)


def inject_page_navigation(target_page: str) -> None:
    target_label = nav_label(target_page)
    components.html(
        f"""
        <script>
        const targetLabel = {json.dumps(target_label)};
        const clickSidebarOption = () => {{
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) {{
                return false;
            }}

            const candidates = Array.from(sidebar.querySelectorAll('label, label *, [role="radiogroup"] *'));
            const match = candidates.find((node) => (node.textContent || '').trim() === targetLabel);
            const clickable = match ? (match.closest('label') || match) : null;
            if (!clickable) {{
                return false;
            }}

            clickable.click();
            return true;
        }};

        if (!clickSidebarOption()) {{
            let attempts = 0;
            const timer = window.setInterval(() => {{
                attempts += 1;
                if (clickSidebarOption() || attempts >= 20) {{
                    window.clearInterval(timer);
                }}
            }}, 150);
        }}
        </script>
        """,
        height=0,
    )


def build_column_reference_table() -> pd.DataFrame:
    return pd.DataFrame(COLUMN_REFERENCE_ROWS)


def _find_matching_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    candidate_lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        matched = candidate_lookup.get(candidate.strip().lower())
        if matched is not None:
            return matched
    return None


def build_dtype_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    dtype_counts = frame.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["dtype", "column_count"]
    return dtype_counts.sort_values(["column_count", "dtype"], ascending=[False, True]).reset_index(drop=True)


def build_quick_stats_table(frame: pd.DataFrame) -> pd.DataFrame:
    stat_rows: list[dict[str, object]] = []
    quick_stat_columns = {
        "price": ["price"],
        "minimum_nights": ["minimum_nights", "minimum nights"],
        "availability_365": ["availability_365", "availability 365"],
        "number_of_reviews": ["number_of_reviews", "number of reviews"],
    }

    for label, candidates in quick_stat_columns.items():
        column_name = _find_matching_column(frame, candidates)
        if column_name is None:
            continue
        numeric_series = _coerce_numeric_series(column_name, frame[column_name])
        if numeric_series is None:
            continue
        numeric_series = numeric_series.dropna()
        if numeric_series.empty:
            continue
        stat_rows.append(
            {
                "Column": label,
                "Min": round(float(numeric_series.min()), 2),
                "Max": round(float(numeric_series.max()), 2),
                "Mean": round(float(numeric_series.mean()), 2),
            }
        )

    return pd.DataFrame(stat_rows)


def filter_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    filter_cols = st.columns(4)

    if "neighbourhood_group" in frame.columns:
        area_options = sorted(frame["neighbourhood_group"].dropna().astype(str).unique().tolist())
        selected_areas = filter_cols[0].multiselect(t("raw.filter.neighborhood_group"), area_options)
        if selected_areas:
            filtered = filtered[filtered["neighbourhood_group"].isin(selected_areas)]

    if "room_type" in frame.columns:
        room_options = sorted(frame["room_type"].dropna().astype(str).unique().tolist())
        selected_rooms = filter_cols[1].multiselect(
            t("raw.filter.room_type"),
            room_options,
            format_func=translate_room_type,
        )
        if selected_rooms:
            filtered = filtered[filtered["room_type"].isin(selected_rooms)]

    if "price" in frame.columns and not filtered.empty and not filtered["price"].dropna().empty:
        min_price = float(filtered["price"].min())
        max_price = float(filtered["price"].max())
        if min_price != max_price:
            price_range = filter_cols[2].slider(
                t("raw.filter.price_range"),
                min_price,
                max_price,
                (min_price, max_price),
            )
            filtered = filtered[filtered["price"].between(price_range[0], price_range[1])]

    if "number_of_reviews" in frame.columns and not filtered.empty and not filtered["number_of_reviews"].dropna().empty:
        review_cap = int(filtered["number_of_reviews"].max())
        review_threshold = filter_cols[3].slider(t("raw.filter.minimum_reviews"), 0, review_cap, 0)
        filtered = filtered[filtered["number_of_reviews"] >= review_threshold]

    return filtered


def render_page(raw_frame: pd.DataFrame, cleaned_frame: pd.DataFrame) -> None:
    st.title(t("raw.title"))
    st.caption(t("raw.caption"))

    if st.button("Process Data", type="primary"):
        inject_page_navigation("preprocessing")

    session_raw_frame = st.session_state.get("raw_df")
    session_raw_name = st.session_state.get("raw_df_name")
    audit_frame = session_raw_frame.copy() if isinstance(session_raw_frame, pd.DataFrame) else raw_frame.copy()
    uploaded_file = st.file_uploader(
        t("raw.upload.label"),
        type=["csv"],
        help=t("raw.upload.help"),
    )
    if uploaded_file is not None:
        try:
            audit_frame = pd.read_csv(uploaded_file)
            st.session_state["raw_df"] = audit_frame.copy()
            st.session_state["raw_df_name"] = uploaded_file.name
            st.session_state["processed_df"] = None
            st.session_state["preprocessing_before_df"] = None
            st.session_state["processing_report"] = None
            st.caption(t("raw.source.uploaded", file_name=uploaded_file.name))
        except Exception as exc:
            audit_frame = raw_frame
            st.error(t("raw.upload.error", error=str(exc)))
    elif isinstance(session_raw_frame, pd.DataFrame):
        st.caption(t("raw.source.uploaded", file_name=session_raw_name or "uploaded CSV"))
    else:
        st.session_state["raw_df"] = raw_frame.copy()
        st.session_state["raw_df_name"] = None
        st.caption(t("raw.source.default"))

    metric_cols = st.columns(2)
    metric_cols[0].metric(t("raw.metric.rows"), f"{len(audit_frame):,}")
    metric_cols[1].metric(t("raw.metric.columns"), f"{audit_frame.shape[1]:,}")

    tab_summary_overview, tab_preview, tab_missing, tab_pipeline, tab_cleaned = st.tabs(
        [
            "Overview",
            t("raw.tab.preview"),
            t("raw.tab.missing"),
            t("raw.tab.pipeline"),
            t("raw.tab.cleaned"),
        ]
    )

    with tab_summary_overview:
        overview_metrics = st.columns(4)
        neighbourhood_group_column = _find_matching_column(audit_frame, ["neighbourhood group", "neighbourhood_group"])
        neighbourhood_group_count = (
            f"{audit_frame[neighbourhood_group_column].dropna().nunique():,}"
            if neighbourhood_group_column is not None
            else "N/A"
        )
        overview_metrics[0].metric("Total Rows", f"{len(audit_frame):,}")
        overview_metrics[1].metric("Total Columns", f"{audit_frame.shape[1]:,}")
        overview_metrics[2].metric("Total Missing Values", f"{int(audit_frame.isna().sum().sum()):,}")
        overview_metrics[3].metric("Neighbourhood Groups", neighbourhood_group_count)

        dtype_col, stats_col = st.columns(2, gap="large")
        with dtype_col:
            st.subheader("Data Types Distribution")
            dtype_distribution = build_dtype_distribution_table(audit_frame)
            dtype_chart = px.bar(
                dtype_distribution,
                x="column_count",
                y="dtype",
                orientation="h",
                color="column_count",
                color_continuous_scale=["#f3dcc0", "#c95c36", "#7e3120"],
            )
            dtype_chart.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(dtype_chart, use_container_width=True)

        with stats_col:
            st.subheader("Key Columns Quick Stats")
            quick_stats = build_quick_stats_table(audit_frame)
            if quick_stats.empty:
                st.info("No supported numeric columns found in the uploaded data.")
            else:
                st.dataframe(quick_stats, use_container_width=True, hide_index=True, height=320)

        st.subheader("Column Reference Table")
        overview_reference = build_column_reference_table().rename(
            columns={
                "column": "Column",
                "original_dtype": "Original Dtype",
                "meaning": "Meaning",
                "handling": "How to Handle",
            }
        )
        st.dataframe(
            overview_reference,
            use_container_width=True,
            hide_index=True,
            height=620,
        )

    with tab_preview:
        st.subheader(t("raw.preview.title"))
        st.dataframe(localize_dataframe_for_display(audit_frame.head(50)), use_container_width=True, height=360)

    with tab_missing:
        st.subheader(t("raw.missing.title"))
        health_table = build_missing_table(audit_frame)
        st.dataframe(
            localize_dataframe_for_display(health_table),
            use_container_width=True,
            hide_index=True,
            height=620,
        )

        null_chart = px.bar(
            health_table.sort_values(["null count", "column name"], ascending=[False, True]),
            x="null count",
            y="column name",
            orientation="h",
            color="null %",
            title="Null Values by Column",
            color_continuous_scale=["#f3dcc0", "#c95c36", "#7e3120"],
        )
        null_chart.update_layout(
            coloraxis_colorbar_title_text="Null %",
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(null_chart, use_container_width=True)

        numeric_profile = build_numeric_profile_frame(audit_frame)
        if not numeric_profile.empty:
            melted_numeric = numeric_profile.melt(var_name="column name", value_name="value").dropna()
            if not melted_numeric.empty:
                boxplot = px.box(
                    melted_numeric,
                    x="value",
                    y="column name",
                    color="column name",
                    orientation="h",
                    title="Numerical Column Distribution",
                    color_discrete_sequence=CHART_COLORS,
                )
                boxplot.update_layout(
                    showlegend=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=max(420, len(numeric_profile.columns) * 70),
                )
                st.plotly_chart(boxplot, use_container_width=True)

    with tab_pipeline:
        st.subheader(t("raw.pipeline.title"))
        st.caption(t("raw.pipeline.caption"))
        for step in PREPROCESSING_PIPELINE_STEPS:
            with st.expander(step["title"]):
                st.code(step["code"], language="python")

    with tab_cleaned:
        st.subheader(t("raw.cleaned.title"))
        filtered = filter_dataframe(cleaned_frame)
        st.metric(t("raw.metric.filtered_rows"), f"{len(filtered):,}")
        st.dataframe(localize_dataframe_for_display(filtered), use_container_width=True, height=420)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("raw.download"),
            csv_data,
            file_name=t("raw.download_filename"),
            mime="text/csv",
        )
