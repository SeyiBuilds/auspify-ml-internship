"""Task 2 (Easy) - Content Type Prediction Model.

Predicts whether a Netflix title is a Movie or a TV Show from its features.

Workflow
    Step 1: Select relevant dataset features.
    Step 2: Encode categorical variables.
    Step 3: Train classification models.
    Step 4: Evaluate prediction performance.
    Step 5: Compare model accuracy.

Run:
    python src/task2_content_type.py
"""

from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import REPORT_DIR, ensure_output_dirs, load_netflix, save_figure

# Step 1: the features available before knowing the content type.
#
# Leakage note: duration and the raw genre labels give the answer away.
# "4 Seasons" only ever describes a TV Show, and listed_in values such as
# "Crime TV Shows" or "Horror Movies" carry the type inside the label. Leaving
# them in produced 100 percent accuracy, which is a leaking model rather than a
# good one. Duration is dropped and the genre labels are stripped of their type
# markers so the model has to learn from real signal.
NUMERIC_FEATURES = [
    "release_year",
    "genre_count",
    "cast_count",
    "country_count",
    "description_length",
    "description_word_count",
    "has_director",
    "has_cast",
    "year_added",
    "month_added",
    "years_to_platform",
]
CATEGORICAL_FEATURES = ["rating", "clean_primary_genre", "primary_country"]


TYPE_MARKERS = re.compile(r"\b(tv\s*shows?|tv|movies?|films?)\b", flags=re.IGNORECASE)


def strip_type_markers(value: object) -> str:
    """Remove Movie/TV wording from a genre label so it cannot leak the target."""
    if pd.isna(value):
        return "Unknown"
    cleaned = TYPE_MARKERS.sub(" ", str(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,-")
    return cleaned or "Unknown"


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the modelling columns and the Movie/TV Show target."""
    df = df.copy()
    df["clean_primary_genre"] = df["primary_genre"].apply(strip_type_markers)
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X = df[numeric + categorical].copy()
    for column in categorical:
        X[column] = X[column].fillna("Unknown").astype(str)

    y = (df["type"].str.strip().str.lower() == "tv show").astype(int)
    print(f"  Features: {len(numeric)} numeric, {len(categorical)} categorical")
    print(f"  Target balance: {int((y == 0).sum())} Movies, {int((y == 1).sum())} TV Shows")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Step 2: impute, scale numeric columns and one-hot encode categoricals."""
    numeric = [c for c in X.columns if c in NUMERIC_FEATURES]
    categorical = [c for c in X.columns if c in CATEGORICAL_FEATURES]

    from sklearn.impute import SimpleImputer

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


def get_models() -> dict:
    """Step 3: the candidate classifiers to train and compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test, cv_folds: int) -> dict:
    """Step 4: fit, predict and score one model."""
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    folds = min(cv_folds, int(y_train.value_counts().min()))
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        cv_mean, cv_std = float("nan"), float("nan")

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "cv_accuracy_mean": round(cv_mean, 4),
        "cv_accuracy_std": round(cv_std, 4),
    }


def plot_model_comparison(results: pd.DataFrame) -> None:
    """Step 5: bar chart comparing the models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["accuracy", "f1", "roc_auc"]
    positions = np.arange(len(results))
    width = 0.26
    colors = ["#3b6ea5", "#e08a3c", "#5a9e6f"]

    for offset, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(positions + offset * width, results[metric], width, label=metric.upper(), color=color)

    ax.set_xticks(positions + width)
    ax.set_xticklabels(results["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Task 2 - Movie vs TV Show model comparison")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "task2_model_comparison.png")
    plt.close(fig)


def plot_confusion(pipeline: Pipeline, X_test, y_test, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["Movie", "TV Show"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {model_name}")
    save_figure(fig, "task2_confusion_matrix.png")
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, model_name: str, top_n: int = 15) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=names).nlargest(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index.str.replace("numeric__|categorical__", "", regex=True), importances.values, color="#3b6ea5")
    ax.set_title(f"Top {top_n} features - {model_name}")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "task2_feature_importance.png")
    plt.close(fig)


def main() -> None:
    print("Task 2 - Content Type Prediction Model")
    print("=" * 60)

    print("\nStep 1: Selecting features")
    df = load_netflix()
    X, y = build_feature_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    print("\nStep 2 and 3: Encoding features and training models")
    results, pipelines = [], {}
    for name, model in get_models().items():
        pipeline = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        scores = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, cv_folds=5)
        results.append(scores)
        pipelines[name] = pipeline
        print(f"  {name:<20} accuracy={scores['accuracy']:.4f}  f1={scores['f1']:.4f}  roc_auc={scores['roc_auc']:.4f}")

    print("\nStep 4 and 5: Comparing performance")
    table = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    print(table.to_string(index=False))

    best_name = table.iloc[0]["model"]
    best_pipeline = pipelines[best_name]
    print(f"\n  Best model: {best_name}")
    print("\n" + classification_report(y_test, best_pipeline.predict(X_test), target_names=["Movie", "TV Show"]))

    plot_model_comparison(table)
    plot_confusion(best_pipeline, X_test, y_test, best_name)
    plot_feature_importance(best_pipeline, best_name)

    ensure_output_dirs()
    table.to_csv(os.path.join(REPORT_DIR, "task2_model_comparison.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "task2_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"best_model": best_name, "results": results}, handle, indent=2)
    print("\nResults written to outputs/reports/task2_model_comparison.csv and task2_summary.json")


if __name__ == "__main__":
    main()
