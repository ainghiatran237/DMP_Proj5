from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CHART_COLORS, SAMPLE_SOURCE_LABEL
from core.formatting import format_currency
from core.i18n import display_source_label, localize_dataframe_for_display, t, translate_room_type
from core.insights import insight_sentences


def render_page(frame: pd.DataFrame, source_label: str) -> None:
    source_text = display_source_label(source_label)
    st.markdown(
        f"""
        <div class="hero">
            <h1>{t("app.title")}</h1>
            <p>
                {t("overview.hero_body", source=source_text)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_label == SAMPLE_SOURCE_LABEL:
        st.info(t("overview.sample_info"))

    metric_cols = st.columns(4)
    price_series = frame["price"] if "price" in frame.columns else pd.Series(dtype=float)
    reviews_series = frame["number_of_reviews"] if "number_of_reviews" in frame.columns else pd.Series(dtype=float)
    availability_series = frame["availability_365"] if "availability_365" in frame.columns else pd.Series(dtype=float)

    metric_cols[0].metric(t("overview.metric.listings"), f"{len(frame):,}")
    metric_cols[1].metric(
        t("overview.metric.median_price"),
        format_currency(price_series.median() if not price_series.empty else None, fallback=t("common.na")),
    )
    metric_cols[2].metric(
        t("overview.metric.avg_reviews"),
        f"{reviews_series.mean():.1f}" if not reviews_series.empty else t("common.na"),
    )
    metric_cols[3].metric(
        t("overview.metric.avg_availability"),
        t("common.days_suffix", value=f"{availability_series.mean():.0f}")
        if not availability_series.empty
        else t("common.na"),
    )

    chart_col, mix_col = st.columns([1.25, 0.75], gap="large")
    with chart_col:
        if "price" in frame.columns and not frame["price"].empty:
            histogram = px.histogram(
                frame,
                x="price",
                nbins=35,
                title=t("overview.chart.price_distribution"),
                color_discrete_sequence=[CHART_COLORS[1]],
            )
            histogram.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(histogram, use_container_width=True)

    with mix_col:
        if "room_type" in frame.columns:
            room_mix = frame["room_type"].value_counts().reset_index()
            room_mix.columns = ["room_type", "count"]
            room_mix["room_type_label"] = room_mix["room_type"].map(translate_room_type)
            donut = px.pie(
                room_mix,
                names="room_type_label",
                values="count",
                hole=0.52,
                title=t("overview.chart.room_mix"),
                color_discrete_sequence=CHART_COLORS,
            )
            donut.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
            st.plotly_chart(donut, use_container_width=True)

    left_col, right_col = st.columns([0.85, 1.15], gap="large")
    with left_col:
        st.subheader(t("overview.key_insights"))
        for insight in insight_sentences(frame):
            st.markdown(f"- {insight}")

    with right_col:
        if {"neighbourhood_group", "price"}.issubset(frame.columns):
            area_summary = (
                frame.groupby("neighbourhood_group", dropna=False)["price"]
                .agg(["median", "mean", "count"])
                .sort_values("median", ascending=False)
                .reset_index()
            )
            area_summary["median"] = area_summary["median"].round(2)
            area_summary["mean"] = area_summary["mean"].round(2)
            st.subheader(t("overview.neighborhood_snapshot"))
            st.dataframe(localize_dataframe_for_display(area_summary), use_container_width=True, hide_index=True)
