from __future__ import annotations

import pandas as pd

from core.formatting import format_currency
from core.i18n import t, translate_room_type


def insight_sentences(frame: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if frame.empty:
        return [t("insight.no_data")]

    if "price" in frame.columns:
        insights.append(
            t(
                "insight.typical_price",
                median=format_currency(frame["price"].median(), fallback=t("common.na")),
                mean=format_currency(frame["price"].mean(), fallback=t("common.na")),
            )
        )

    if {"neighbourhood_group", "price"}.issubset(frame.columns):
        area_prices = frame.groupby("neighbourhood_group", dropna=False)["price"].median().sort_values(ascending=False)
        if not area_prices.empty:
            insights.append(
                t(
                    "insight.top_area",
                    area=area_prices.index[0],
                    price=format_currency(area_prices.iloc[0], fallback=t("common.na")),
                )
            )

    if "room_type" in frame.columns:
        room_mix = frame["room_type"].value_counts(normalize=True)
        if not room_mix.empty:
            insights.append(
                t(
                    "insight.room_mix",
                    room_type=translate_room_type(room_mix.index[0]),
                    share=f"{room_mix.iloc[0] * 100:.1f}",
                )
            )

    if {"room_type", "number_of_reviews"}.issubset(frame.columns):
        review_rank = frame.groupby("room_type", dropna=False)["number_of_reviews"].median().sort_values(ascending=False)
        if not review_rank.empty:
            insights.append(
                t(
                    "insight.review_rank",
                    room_type=translate_room_type(review_rank.index[0]),
                    reviews=f"{review_rank.iloc[0]:.0f}",
                )
            )

    if {"availability_365", "price"}.issubset(frame.columns) and len(frame) > 1:
        correlation = frame[["availability_365", "price"]].corr(numeric_only=True).iloc[0, 1]
        if pd.notna(correlation):
            strength_key = "weak"
            if abs(correlation) >= 0.5:
                strength_key = "strong"
            elif abs(correlation) >= 0.25:
                strength_key = "moderate"
            direction_key = "positive" if correlation >= 0 else "negative"
            insights.append(
                t(
                    "insight.availability_correlation",
                    strength=t(f"insight.strength.{strength_key}"),
                    direction=t(f"insight.direction.{direction_key}"),
                    correlation=f"{correlation:.2f}",
                )
            )

    return insights


def answer_chat_question(question: str, frame: pd.DataFrame) -> str:
    prompt = question.lower()
    insights = insight_sentences(frame)

    if frame.empty:
        return insights[0]

    if any(token in prompt for token in ("neighbourhood", "neighborhood", "area", "borough", "khu", "quận", "vùng")) and {"neighbourhood_group", "price"}.issubset(frame.columns):
        area_prices = frame.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False)
        top_two = ", ".join(
            f"{area}: {format_currency(value, fallback=t('common.na'))}"
            for area, value in area_prices.head(2).items()
        )
        return t("chat.answer.top_areas", areas=top_two)

    if any(token in prompt for token in ("price", "cost", "expensive", "cheap", "giá", "đắt", "rẻ", "chi phí")) and "price" in frame.columns:
        return t(
            "chat.answer.price",
            median=format_currency(frame["price"].median(), fallback=t("common.na")),
            mean=format_currency(frame["price"].mean(), fallback=t("common.na")),
            percentile=format_currency(frame["price"].quantile(0.75), fallback=t("common.na")),
        )

    if any(token in prompt for token in ("room", "type", "phòng", "loại")) and "room_type" in frame.columns:
        room_mix = frame["room_type"].value_counts(normalize=True).mul(100).round(1)
        room_summary = ", ".join(f"{translate_room_type(room)}: {share:.1f}%" for room, share in room_mix.items())
        return t("chat.answer.room_mix", summary=room_summary)

    if any(token in prompt for token in ("review", "rating", "demand", "đánh giá", "nhu cầu")) and "number_of_reviews" in frame.columns:
        return t(
            "chat.answer.reviews",
            median=f"{frame['number_of_reviews'].median():.0f}",
            top_decile=f"{frame['number_of_reviews'].quantile(0.9):.0f}",
        )

    if any(token in prompt for token in ("availability", "book", "open", "trống", "khả dụng", "available")) and "availability_365" in frame.columns:
        return t(
            "chat.answer.availability",
            median=f"{frame['availability_365'].median():.0f}",
            mean=f"{frame['availability_365'].mean():.0f}",
        )

    return " ".join(insights[:2])
