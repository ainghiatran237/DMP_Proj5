from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r".*joblib will operate in serial mode.*",
    category=UserWarning,
)

from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_INPUT_PATH = Path("data/Airbnb_Open_Data.csv")
DEFAULT_OUTPUT_PATH = Path("data/Airbnb_Data_cleaned.csv")

NYC_LAT_RANGE = (40.4774, 40.9176)
NYC_LONG_RANGE = (-74.2591, -73.7004)
SPARSE_COLUMN_THRESHOLD = 0.4

NEIGHBOURHOOD_GROUP_REPLACEMENTS = {
    "manhatten": "manhattan",
    "brookyn": "brooklyn",
    "staten islaand": "staten island",
}
VALID_NEIGHBOURHOOD_GROUPS = {
    "manhattan",
    "brooklyn",
    "queens",
    "bronx",
    "staten island",
}

CURRENCY_COLUMNS = ("price", "service_fee")
NUMERIC_COLUMNS = (
    "lat",
    "long",
    "construction_year",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "review_rate_number",
    "calculated_host_listings_count",
    "availability_365",
)
BOOLEAN_COLUMNS = ("host_identity_verified", "instant_bookable")
DATE_COLUMNS = ("last_review",)
IDENTIFIER_COLUMNS = {"id", "host_id", "name", "host_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the Airbnb Open Data dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the raw Airbnb CSV file. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the cleaned CSV output. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(column).strip().lower())).strip("_")
        for column in normalized.columns
    ]

    alias_map = {
        "neighborhood_group": "neighbourhood_group",
        "neighborhood": "neighbourhood",
        "availability": "availability_365",
        "reviews": "number_of_reviews",
        "review_count": "number_of_reviews",
    }
    rename_map = {column: alias_map[column] for column in normalized.columns if column in alias_map}
    return normalized.rename(columns=rename_map)


def coerce_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def canonicalize_neighbourhood_group(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    normalized = normalized.replace(NEIGHBOURHOOD_GROUP_REPLACEMENTS)
    valid_or_missing = normalized.isna() | normalized.isin(VALID_NEIGHBOURHOOD_GROUPS)
    normalized = normalized.where(valid_or_missing, "unknown")
    return normalized.str.title().replace({"Unknown": "Unknown"})


def coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "t": 1,
            "f": 0,
            "1": 1,
            "0": 0,
        }
    )
    return pd.Series(pd.array(mapped, dtype="Int64"), index=series.index)


def fill_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()

    numeric_columns = filled.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        median_value = filled[column].median()
        if pd.notna(median_value):
            filled[column] = filled[column].fillna(median_value)

    for column in filled.columns.difference(numeric_columns):
        if pd.api.types.is_datetime64_any_dtype(filled[column]):
            continue
        mode = filled[column].mode(dropna=True)
        if not mode.empty:
            filled[column] = filled[column].where(filled[column].notna(), mode.iloc[0])

    return filled


def scale_numeric_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    scaled = frame.copy()
    scale_columns = [
        column
        for column in scaled.select_dtypes(include=["number"]).columns
        if column not in BOOLEAN_COLUMNS and column not in IDENTIFIER_COLUMNS
    ]
    if not scale_columns:
        return scaled, []

    scaler = StandardScaler()
    scaled[scale_columns] = scaler.fit_transform(scaled[scale_columns])
    return scaled, scale_columns


def label_encode_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoded = frame.copy()
    encoded_columns: list[str] = []
    object_columns = encoded.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        if column in IDENTIFIER_COLUMNS:
            continue
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoded_columns.append(column)

    return encoded, encoded_columns


def preprocess_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = normalize_columns(frame)
    rows_before = len(cleaned)

    for column in IDENTIFIER_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string")

    threshold = int(np.ceil(SPARSE_COLUMN_THRESHOLD * len(cleaned))) if len(cleaned) else 0
    sparse_columns = [
        column for column in cleaned.columns if cleaned[column].notna().sum() < threshold
    ]
    if sparse_columns:
        cleaned = cleaned.drop(columns=sparse_columns)

    dropped_long_text_columns: list[str] = []
    if "house_rules" in cleaned.columns:
        cleaned = cleaned.drop(columns=["house_rules"])
        dropped_long_text_columns.append("house_rules")

    for column in CURRENCY_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = coerce_currency(cleaned[column])

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "neighbourhood_group" in cleaned.columns:
        cleaned["neighbourhood_group"] = canonicalize_neighbourhood_group(cleaned["neighbourhood_group"])

    non_positive_replaced: dict[str, int] = {}
    for column in ("price", "minimum_nights"):
        if column not in cleaned.columns:
            continue
        invalid_mask = cleaned[column].notna() & (cleaned[column] <= 0)
        non_positive_replaced[column] = int(invalid_mask.sum())
        if invalid_mask.any():
            cleaned.loc[invalid_mask, column] = np.nan

    cleaned = fill_missing_values(cleaned)

    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    invalid_location_rows = 0
    if {"lat", "long"}.issubset(cleaned.columns):
        lat_valid = cleaned["lat"].between(*NYC_LAT_RANGE)
        long_valid = cleaned["long"].between(*NYC_LONG_RANGE)
        location_mask = lat_valid & long_valid
        invalid_location_rows = int((~location_mask).sum())
        cleaned = cleaned.loc[location_mask].copy()

    price_outliers_removed = 0
    if "price" in cleaned.columns and cleaned["price"].notna().any():
        q1 = cleaned["price"].quantile(0.25)
        q3 = cleaned["price"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = cleaned["price"].between(lower_bound, upper_bound)
        price_outliers_removed = int((~outlier_mask).sum())
        cleaned = cleaned.loc[outlier_mask].copy()

    if "price" in cleaned.columns:
        cleaned["log_price"] = np.log1p(cleaned["price"])

    boolean_fill_defaults: dict[str, int] = {}
    for column in BOOLEAN_COLUMNS:
        if column not in cleaned.columns:
            continue
        cleaned[column] = coerce_boolean(cleaned[column])
        mode = cleaned[column].mode(dropna=True)
        fill_value = int(mode.iloc[0]) if not mode.empty else 0
        boolean_fill_defaults[column] = fill_value
        cleaned[column] = cleaned[column].fillna(fill_value).astype(int)

    for column in DATE_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    cleaned, scaled_columns = scale_numeric_features(cleaned)
    cleaned, encoded_columns = label_encode_columns(cleaned)

    report = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "sparse_columns_dropped": sparse_columns,
        "long_text_columns_dropped": dropped_long_text_columns,
        "non_positive_replaced": non_positive_replaced,
        "duplicates_removed": duplicates_removed,
        "invalid_location_rows_removed": invalid_location_rows,
        "price_outliers_removed": price_outliers_removed,
        "scaled_columns": scaled_columns,
        "encoded_columns": encoded_columns,
        "boolean_fill_defaults": boolean_fill_defaults,
    }
    return cleaned.reset_index(drop=True), report


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_path}. Add Airbnb_Open_Data.csv to the data directory "
            "or pass --input with a valid path."
        )
    return pd.read_csv(input_path)


def save_dataset(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def print_report(report: dict[str, object], output_path: Path) -> None:
    print("Airbnb preprocessing completed.")
    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Rows before: {report['rows_before']}")
    print(f"Rows after: {report['rows_after']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Invalid location rows removed: {report['invalid_location_rows_removed']}")
    print(f"Price outliers removed: {report['price_outliers_removed']}")
    print(f"Non-positive values replaced: {report['non_positive_replaced']}")
    print(f"Sparse columns dropped: {report['sparse_columns_dropped']}")
    print(f"Long-text columns dropped: {report['long_text_columns_dropped']}")
    print(f"Scaled numeric columns: {report['scaled_columns']}")
    print(f"Label-encoded columns: {report['encoded_columns']}")


def main() -> None:
    args = parse_args()
    try:
        raw_frame = load_dataset(args.input)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    cleaned_frame, report = preprocess_dataframe(raw_frame)
    save_dataset(cleaned_frame, args.output)
    print_report(report, args.output)


if __name__ == "__main__":
    main()
