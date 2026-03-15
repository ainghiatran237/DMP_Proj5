from __future__ import annotations

from pathlib import Path

APP_TITLE = "Airbnb Analytics Dashboard"
DATASET_PATH = Path("data/Airbnb_Open_Data.csv")
SAMPLE_SOURCE_LABEL = "Bundled sample data"
NAVIGATION_PAGES = [
    "overview",
    "data_raw",
    "preprocessing",
    "eda",
    "conclusion",
    "chatbot",
]
CHART_COLORS = ["#1f3c5b", "#c95c36", "#d8a65d", "#6d8f71"]
