"""Shared data loading, cleaning and feature engineering for all internship tasks.

Every task script imports from this module so that cleaning rules stay
consistent across the whole project.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")

# Preferred dataset file names, checked in order.
CANDIDATE_FILES = ("Dataset.csv", "netflix_titles.csv", "sample_netflix_titles.csv")

EXPECTED_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]

# Ratings grouped into the audience bands used by Task 3 and Task 6.
AUDIENCE_BANDS = {
    "G": "Kids",
    "TV-Y": "Kids",
    "TV-Y7": "Kids",
    "TV-Y7-FV": "Kids",
    "TV-G": "Kids",
    "PG": "Family",
    "TV-PG": "Family",
    "PG-13": "Teen",
    "TV-14": "Teen",
    "R": "Adult",
    "NC-17": "Adult",
    "TV-MA": "Adult",
}


def resolve_data_path(filename: str | None = None) -> str:
    """Return the path of the dataset to use, raising a clear error if absent."""
    if filename:
        path = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        return path

    for name in CANDIDATE_FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No dataset found. Place the internship Dataset.csv inside the data/ folder. "
        f"Looked for: {', '.join(CANDIDATE_FILES)}"
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and snake_case the column names so variants still load."""
    renamed = {}
    for col in df.columns:
        clean = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower()
        renamed[col] = clean
    df = df.rename(columns=renamed)

    aliases = {
        "listedin": "listed_in",
        "genres": "listed_in",
        "genre": "listed_in",
        "dateadded": "date_added",
        "releaseyear": "release_year",
        "showid": "show_id",
        "content_type": "type",
        "cast_members": "cast",
    }
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})


def parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Split the mixed duration column into numeric minutes and seasons."""
    duration = df.get("duration", pd.Series(index=df.index, dtype=object)).astype(str)
    number = pd.to_numeric(duration.str.extract(r"(\d+)")[0], errors="coerce")
    is_season = duration.str.contains("season", case=False, na=False)

    df["duration_value"] = number
    df["duration_minutes"] = number.where(~is_season)
    df["duration_seasons"] = number.where(is_season)
    df["duration_unit"] = np.where(is_season, "seasons", "minutes")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the date features used by the forecasting and insight tasks."""
    if "date_added" in df.columns:
        added = pd.to_datetime(df["date_added"].astype(str).str.strip(), errors="coerce", format="mixed")
        df["date_added_parsed"] = added
        df["year_added"] = added.dt.year
        df["month_added"] = added.dt.month
        df["quarter_added"] = added.dt.quarter
    else:
        df["date_added_parsed"] = pd.NaT
        df["year_added"] = np.nan
        df["month_added"] = np.nan
        df["quarter_added"] = np.nan

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df["years_to_platform"] = df["year_added"] - df["release_year"]
    return df


def add_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and flags derived from the multi-value text columns."""
    def count_items(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .apply(lambda v: 0 if not v.strip() else len([p for p in v.split(",") if p.strip()]))
        )

    df["genre_count"] = count_items(df.get("listed_in", pd.Series("", index=df.index)))
    df["cast_count"] = count_items(df.get("cast", pd.Series("", index=df.index)))
    df["country_count"] = count_items(df.get("country", pd.Series("", index=df.index)))

    description = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    df["description_length"] = description.str.len()
    df["description_word_count"] = description.str.split().apply(len)

    df["has_director"] = df.get("director", pd.Series(np.nan, index=df.index)).notna().astype(int)
    df["has_cast"] = df.get("cast", pd.Series(np.nan, index=df.index)).notna().astype(int)

    df["primary_genre"] = (
        df.get("listed_in", pd.Series("", index=df.index))
        .fillna("Unknown")
        .astype(str)
        .apply(lambda v: v.split(",")[0].strip() if v.strip() else "Unknown")
    )
    df["primary_country"] = (
        df.get("country", pd.Series("", index=df.index))
        .fillna("Unknown")
        .astype(str)
        .apply(lambda v: v.split(",")[0].strip() if v.strip() else "Unknown")
    )

    if "rating" in df.columns:
        df["audience_band"] = df["rating"].map(AUDIENCE_BANDS).fillna("Unknown")
    else:
        df["audience_band"] = "Unknown"
    return df


def load_netflix(filename: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Load the dataset, clean it and attach every engineered feature."""
    path = resolve_data_path(filename)
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    raw_rows = len(df)

    df = _normalise_columns(df)
    df = df.drop_duplicates()

    for column in ("type", "title", "rating", "duration", "listed_in", "description"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    if "type" in df.columns:
        df = df[df["type"].notna()]

    df = parse_duration(df)
    df = add_time_features(df)
    df = add_content_features(df)
    df = df.reset_index(drop=True)

    if verbose:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        print(f"Loaded {os.path.basename(path)}: {raw_rows} rows -> {len(df)} after cleaning, {df.shape[1]} columns")
        if missing:
            print(f"  Warning: expected columns not present: {', '.join(missing)}")
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values per column, highest first."""
    report = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        }
    )
    return report[report["missing"] > 0].sort_values("missing_pct", ascending=False)


def ensure_output_dirs() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def save_figure(fig, name: str) -> str:
    """Save a matplotlib figure into outputs/figures and return its path."""
    ensure_output_dirs()
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: outputs/figures/{name}")
    return path


if __name__ == "__main__":
    frame = load_netflix()
    print(frame[["title", "type", "primary_genre", "audience_band", "duration_value"]].head())
    print()
    print(missing_value_report(frame))
