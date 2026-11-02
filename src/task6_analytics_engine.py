"""Task 6 (Advanced) - Netflix Content Success Analytics Engine.

An end-to-end pipeline that engineers features, trains and compares several
models, generates automated business insights and writes a visual report.

Workflow
    Step 1: Perform advanced feature engineering.
    Step 2: Build multiple machine learning models.
    Step 3: Compare model performance.
    Step 4: Generate automated insights.
    Step 5: Present findings through visual reports.

Target note
    The dataset carries no viewership or revenue column, so real "success"
    cannot be measured. The modelled target is the audience band a title is
    aimed at (Kids, Family, Teen, Adult), which is the commercial positioning
    decision the catalogue actually records. The insight engine reports
    observed catalogue patterns separately from the model.

Run:
    python src/task6_analytics_engine.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import REPORT_DIR, ensure_output_dirs, load_netflix, save_figure

TARGET = "audience_band"
MIN_CLASS_SIZE = 8


# ---------------------------------------------------------------- Step 1
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the richer features this task needs on top of the shared cleaning."""
    df = df.copy()

    # duration_zscore, which compares length within a content type rather than
    # across types, is provided by common.add_relative_duration.
    df["is_movie"] = (df["type"].str.lower() == "movie").astype(int)
    df["is_international"] = df["listed_in"].fillna("").str.contains("International", case=False).astype(int)
    df["is_documentary"] = df["listed_in"].fillna("").str.contains("Document", case=False).astype(int)
    df["is_stand_up"] = df["listed_in"].fillna("").str.contains("Stand-Up", case=False).astype(int)
    df["is_multi_country"] = (df["country_count"] > 1).astype(int)
    df["is_multi_genre"] = (df["genre_count"] > 1).astype(int)

    # How quickly a title reached the platform after release.
    df["fast_acquisition"] = (df["years_to_platform"] <= 1).astype(int)
    df["is_original_window"] = (df["years_to_platform"] <= 0).astype(int)

    # Catalogue weight of the primary genre and country.
    df["genre_frequency"] = df["primary_genre"].map(df["primary_genre"].value_counts())
    df["country_frequency"] = df["primary_country"].map(df["primary_country"].value_counts())

    description = df["description"].fillna("").astype(str)
    df["description_avg_word_len"] = description.apply(
        lambda t: float(np.mean([len(w) for w in t.split()])) if t.split() else 0.0
    )
    df["title_word_count"] = df["title"].fillna("").astype(str).str.split().apply(len)

    print(f"  Engineered feature set: {df.shape[1]} columns")
    return df


NUMERIC_FEATURES = [
    "release_year",
    "duration_value",
    "duration_zscore",
    "genre_count",
    "cast_count",
    "country_count",
    "description_length",
    "description_word_count",
    "description_avg_word_len",
    "title_word_count",
    "year_added",
    "month_added",
    "years_to_platform",
    "genre_frequency",
    "country_frequency",
    "is_movie",
    "is_international",
    "is_documentary",
    "is_stand_up",
    "is_multi_country",
    "is_multi_genre",
    "fast_acquisition",
    "is_original_window",
    "has_director",
    "has_cast",
]
CATEGORICAL_FEATURES = ["primary_genre", "primary_country", "type"]


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop unknown targets and rare classes, then split features from target."""
    frame = df[df[TARGET] != "Unknown"].copy()
    counts = frame[TARGET].value_counts()
    keep = counts[counts >= MIN_CLASS_SIZE].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"  Dropping classes with fewer than {MIN_CLASS_SIZE} titles: {', '.join(dropped)}")
    frame = frame[frame[TARGET].isin(keep)]

    numeric = [c for c in NUMERIC_FEATURES if c in frame.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    X = frame[numeric + categorical].copy()
    for column in categorical:
        X[column] = X[column].fillna("Unknown").astype(str)

    y = frame[TARGET]
    print(f"  Modelling {len(X)} titles across {y.nunique()} classes: {y.value_counts().to_dict()}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if c in NUMERIC_FEATURES]
    categorical = [c for c in X.columns if c in CATEGORICAL_FEATURES]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


# ---------------------------------------------------------------- Step 2
BASELINE_NAME = "Baseline (most frequent)"


def get_models() -> dict:
    """Candidate models, with a do-nothing baseline for honest comparison.

    Accuracy on its own is misleading when one class dominates: always guessing
    the largest audience band already scores close to half. The baseline is
    trained and scored alongside the real models so the lift is visible rather
    than implied.
    """
    return {
        BASELINE_NAME: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


@dataclass
class ModelRun:
    name: str
    pipeline: Pipeline
    metrics: dict = field(default_factory=dict)


def train_and_compare(X, y) -> tuple[pd.DataFrame, dict, tuple]:
    """Steps 2 and 3: train every candidate and rank them."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    runs, rows = {}, []

    for name, model in get_models().items():
        pipeline = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        folds = min(5, int(y_train.value_counts().min()))
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
        else:
            cv_mean, cv_std = float("nan"), float("nan")

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "f1_macro": round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
            "cv_f1_macro_mean": round(cv_mean, 4),
            "cv_f1_macro_std": round(cv_std, 4),
        }
        rows.append(metrics)
        runs[name] = ModelRun(name, pipeline, metrics)
        print(f"  {name:<20} accuracy={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}")

    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    baseline = table[table["model"] == BASELINE_NAME]
    if not baseline.empty:
        table["accuracy_lift_vs_baseline"] = (
            table["accuracy"] - float(baseline.iloc[0]["accuracy"])
        ).round(4)
    return table, runs, (X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------- Step 4
def generate_insights(df: pd.DataFrame, table: pd.DataFrame, best: ModelRun) -> list[str]:
    """Turn the catalogue and the model into written business insights."""
    insights: list[str] = []

    movie_share = 100 * (df["type"].str.lower() == "movie").mean()
    insights.append(
        f"Catalogue mix: {movie_share:.1f} percent of titles are films and {100 - movie_share:.1f} percent are series."
    )

    genres = df["listed_in"].fillna("").astype(str).str.split(",").explode().str.strip()
    genres = genres[genres != ""]
    if len(genres):
        top = genres.value_counts().head(3)
        insights.append(
            "Largest genres: " + ", ".join(f"{name} ({count} titles)" for name, count in top.items()) + "."
        )

    if df["year_added"].notna().any():
        per_year = df.groupby("year_added").size()
        per_year = per_year[per_year.index.notna()]
        if len(per_year) >= 2:
            peak_year = int(per_year.idxmax())
            insights.append(
                f"Acquisition peaked in {peak_year} with {int(per_year.max())} titles added; "
                f"the most recent year on record added {int(per_year.iloc[-1])}."
            )
        recent = df[df["year_added"] >= df["year_added"].max() - 2]
        if len(recent):
            recent_series = 100 * (recent["type"].str.lower() == "tv show").mean()
            insights.append(
                f"In the last three years of additions, series made up {recent_series:.1f} percent of new titles "
                f"against {100 - movie_share:.1f} percent across the whole catalogue."
            )

    if df["years_to_platform"].notna().any():
        median_gap = float(df["years_to_platform"].median())
        fast_share = 100 * (df["years_to_platform"] <= 1).mean()
        insights.append(
            f"Median gap between release and arrival on the platform is {median_gap:.0f} year(s), "
            f"and {fast_share:.1f} percent of titles arrive within a year of release."
        )

    band_mix = df[df[TARGET] != "Unknown"][TARGET].value_counts(normalize=True) * 100
    if len(band_mix):
        insights.append(
            "Audience positioning: "
            + ", ".join(f"{name} {share:.1f} percent" for name, share in band_mix.items())
            + "."
        )

    countries = df["primary_country"].replace("Unknown", np.nan).dropna()
    if len(countries):
        top_country = countries.value_counts().head(3)
        insights.append(
            "Leading production countries: "
            + ", ".join(f"{name} ({count})" for name, count in top_country.items())
            + "."
        )

    baseline_row = table[table["model"] == BASELINE_NAME]
    if not baseline_row.empty:
        baseline_accuracy = float(baseline_row.iloc[0]["accuracy"])
        insights.append(
            f"Always guessing the largest audience band scores {baseline_accuracy:.1%}, so the "
            f"model's {best.metrics['accuracy']:.1%} is a lift of "
            f"{best.metrics['accuracy'] - baseline_accuracy:.1%}, not a result that stands alone."
        )

    insights.append(
        f"Best model: {best.name} at {best.metrics['accuracy']:.1%} accuracy and "
        f"{best.metrics['f1_macro']:.3f} macro F1."
    )

    model = best.pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = best.pipeline.named_steps["prep"].get_feature_names_out()
        importances = pd.Series(model.feature_importances_, index=names).nlargest(5)
        pretty = [re.sub(r"^(numeric|categorical)__", "", n) for n in importances.index]
        insights.append("Strongest predictors of audience band: " + ", ".join(pretty) + ".")

    return insights


# ---------------------------------------------------------------- Step 5
def build_dashboard(df: pd.DataFrame, table: pd.DataFrame) -> None:
    """One multi-panel figure summarising the catalogue and the models."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Netflix Content Success Analytics Engine", fontsize=15, fontweight="bold")

    ax = axes[0][0]
    counts = df["type"].value_counts()
    ax.bar(counts.index, counts.values, color=["#3b6ea5", "#e08a3c"])
    ax.set_title("Content type mix")
    ax.set_ylabel("Titles")

    ax = axes[0][1]
    genres = df["listed_in"].fillna("").astype(str).str.split(",").explode().str.strip()
    genres = genres[genres != ""].value_counts().head(8)[::-1]
    ax.barh(genres.index, genres.values, color="#3b6ea5")
    ax.set_title("Top genres")
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[0][2]
    per_year = df.groupby("year_added").size()
    per_year = per_year[per_year.index.notna()]
    if len(per_year):
        ax.plot(per_year.index.astype(int), per_year.values, marker="o", color="#5a9e6f")
    ax.set_title("Titles added per year")
    ax.set_xlabel("Year added")

    ax = axes[1][0]
    bands = df[df[TARGET] != "Unknown"][TARGET].value_counts()
    if len(bands):
        ax.bar(bands.index, bands.values, color="#8e6fb0")
    ax.set_title("Audience band distribution")
    ax.tick_params(axis="x", labelsize=9)

    ax = axes[1][1]
    movies = df[df["type"].str.lower() == "movie"]["duration_value"].dropna()
    if len(movies):
        ax.hist(movies, bins=25, color="#3b6ea5", edgecolor="white")
    ax.set_title("Film runtime (minutes)")

    ax = axes[1][2]
    colors = ["#b8b8b8" if m == BASELINE_NAME else "#e08a3c" for m in table["model"]][::-1]
    ax.barh(table["model"][::-1], table["f1_macro"][::-1], color=colors)
    ax.set_xlim(0, 1)
    ax.set_title("Model macro F1")
    ax.tick_params(axis="y", labelsize=8)

    for row in axes:
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "task6_analytics_dashboard.png")
    plt.close(fig)


def write_report(insights: list[str], table: pd.DataFrame, best: ModelRun, report_text: str) -> str:
    ensure_output_dirs()
    path = os.path.join(REPORT_DIR, "task6_analytics_report.md")
    lines = [
        "# Netflix Content Success Analytics Engine",
        "",
        "Task 6 (Advanced) - Auspify Technologies Machine Learning Internship",
        "",
        "## Automated insights",
        "",
    ]
    lines += [f"{i}. {text}" for i, text in enumerate(insights, start=1)]
    lines += [
        "",
        "## Model comparison",
        "",
        table.to_markdown(index=False),
        "",
        f"## Classification report - {best.name}",
        "",
        "```",
        report_text.strip(),
        "```",
        "",
        "## Figures",
        "",
        "- `outputs/figures/task6_analytics_dashboard.png`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    print("Task 6 - Netflix Content Success Analytics Engine")
    print("=" * 60)

    print("\nStep 1: Advanced feature engineering")
    df = engineer_features(load_netflix())
    X, y = build_xy(df)

    print("\nStep 2 and 3: Training and comparing models")
    table, runs, (X_train, X_test, y_train, y_test) = train_and_compare(X, y)
    print()
    print(table.to_string(index=False))

    ranked = table[table["model"] != BASELINE_NAME]
    best = runs[ranked.iloc[0]["model"]]
    report_text = classification_report(y_test, best.pipeline.predict(X_test), zero_division=0)

    print("\nStep 4: Generating automated insights")
    insights = generate_insights(df, table, best)
    for i, text in enumerate(insights, start=1):
        print(f"  {i}. {text}")

    print("\nStep 5: Building the visual report")
    build_dashboard(df, table)
    path = write_report(insights, table, best, report_text)

    ensure_output_dirs()
    table.to_csv(os.path.join(REPORT_DIR, "task6_model_comparison.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "task6_insights.json"), "w", encoding="utf-8") as handle:
        json.dump({"best_model": best.name, "insights": insights, "metrics": best.metrics}, handle, indent=2)

    print(f"\nReport written to outputs/reports/{os.path.basename(path)}")


if __name__ == "__main__":
    main()
