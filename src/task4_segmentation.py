"""Task 4 (Medium) - Netflix Content Segmentation.

Groups Netflix titles into content clusters using unsupervised learning.

Workflow
    Step 1: Prepare numerical and categorical features.
    Step 2: Apply clustering algorithms.
    Step 3: Identify content groups.
    Step 4: Visualise clusters.
    Step 5: Interpret cluster characteristics.

Run:
    python src/task4_segmentation.py
    python src/task4_segmentation.py --k 6
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import REPORT_DIR, ensure_output_dirs, load_netflix, save_figure

NUMERIC_FEATURES = [
    "release_year",
    "duration_value",
    "genre_count",
    "cast_count",
    "country_count",
    "description_length",
    "description_word_count",
    "years_to_platform",
]
CATEGORICAL_FEATURES = ["type", "audience_band"]
GENRE_COMPONENTS = 8


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Step 1: scale numeric columns, encode categoricals, compress genre text."""
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    numeric = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
    numeric = StandardScaler().fit_transform(numeric)
    names = list(numeric_cols)

    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    blocks = [numeric]
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_cols].fillna("Unknown").astype(str))
        blocks.append(encoded)
        names += list(encoder.get_feature_names_out(categorical_cols))

    # Genres are high dimensional, so they are vectorised then reduced with SVD.
    if "listed_in" in df.columns:
        genre_text = df["listed_in"].fillna("Unknown").astype(str).str.lower()
        tfidf = TfidfVectorizer(token_pattern=r"[^,]+", min_df=1)
        genre_matrix = tfidf.fit_transform(genre_text)
        components = min(GENRE_COMPONENTS, max(2, genre_matrix.shape[1] - 1))
        reduced = TruncatedSVD(n_components=components, random_state=42).fit_transform(genre_matrix)
        reduced = StandardScaler().fit_transform(reduced)
        blocks.append(reduced)
        names += [f"genre_component_{i+1}" for i in range(components)]

    matrix = np.hstack(blocks)
    print(f"  Feature matrix: {matrix.shape[0]} titles x {matrix.shape[1]} features")
    return matrix, names


def select_k(matrix: np.ndarray, k_range: range) -> tuple[pd.DataFrame, int]:
    """Step 2: run the elbow and silhouette checks to choose the cluster count."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(matrix)
        rows.append(
            {
                "k": k,
                "inertia": round(float(model.inertia_), 2),
                "silhouette": round(float(silhouette_score(matrix, labels)), 4),
                "davies_bouldin": round(float(davies_bouldin_score(matrix, labels)), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(matrix, labels)), 2),
            }
        )
        print(f"  k={k}: silhouette={rows[-1]['silhouette']:.4f}  inertia={rows[-1]['inertia']:.1f}")

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return scores, best_k


def plot_selection(scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(scores["k"], scores["inertia"], marker="o", color="#3b6ea5")
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#e08a3c")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Score")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "task4_cluster_selection.png")
    plt.close(fig)


def plot_clusters(matrix: np.ndarray, labels: np.ndarray) -> None:
    """Step 4: project to two dimensions with PCA and colour by cluster."""
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(np.unique(labels)), 1)))

    for cluster, color in zip(sorted(np.unique(labels)), palette):
        mask = labels == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.7, color=color, label=f"Cluster {cluster}")

    ax.set_title("Netflix content clusters (PCA projection)")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "task4_cluster_scatter.png")
    plt.close(fig)


def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: describe what each cluster actually contains."""
    rows = []
    for cluster, group in df.groupby("cluster"):
        genres = (
            group["listed_in"].fillna("").astype(str).str.split(",").explode().str.strip()
        )
        genres = genres[genres != ""]
        top_genres = ", ".join(genres.value_counts().head(3).index) if len(genres) else "Unknown"

        rows.append(
            {
                "cluster": cluster,
                "titles": len(group),
                "share_pct": round(100 * len(group) / len(df), 1),
                "movie_pct": round(100 * (group["type"].str.lower() == "movie").mean(), 1),
                "median_release_year": int(group["release_year"].median()) if group["release_year"].notna().any() else None,
                "median_duration": round(float(group["duration_value"].median()), 1) if group["duration_value"].notna().any() else None,
                "top_audience": group["audience_band"].mode().iat[0] if not group["audience_band"].mode().empty else "Unknown",
                "top_genres": top_genres,
                "sample_titles": ", ".join(group["title"].head(3).astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values("titles", ascending=False).reset_index(drop=True)


def name_cluster(row: pd.Series) -> str:
    """Turn a cluster profile into a readable label."""
    kind = "Movie" if row["movie_pct"] >= 60 else "Series" if row["movie_pct"] <= 40 else "Mixed"
    era = "Recent" if row["median_release_year"] and row["median_release_year"] >= 2017 else "Catalogue"
    genre = str(row["top_genres"]).split(",")[0].strip() or "General"
    return f"{era} {kind} - {genre} ({row['top_audience']})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Netflix content segmentation")
    parser.add_argument("--k", type=int, default=None, help="Force a cluster count")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=9)
    args = parser.parse_args()

    print("Task 4 - Netflix Content Segmentation")
    print("=" * 60)

    print("\nStep 1: Preparing features")
    df = load_netflix()
    matrix, _ = build_feature_matrix(df)

    print("\nStep 2: Choosing the number of clusters")
    k_max = min(args.k_max, max(args.k_min + 1, len(df) - 1))
    scores, suggested_k = select_k(matrix, range(args.k_min, k_max + 1))
    k = args.k or suggested_k
    print(f"  Selected k = {k} (best silhouette at k = {suggested_k})")

    print("\nStep 3: Clustering titles")
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["cluster"] = model.fit_predict(matrix)
    print(f"  Cluster sizes: {df['cluster'].value_counts().sort_index().to_dict()}")

    print("\nStep 4: Visualising clusters")
    plot_selection(scores)
    plot_clusters(matrix, df["cluster"].to_numpy())

    print("\nStep 5: Interpreting clusters")
    profile = profile_clusters(df)
    profile.insert(1, "label", profile.apply(name_cluster, axis=1))
    pd.set_option("display.max_colwidth", 60)
    print(profile[["cluster", "label", "titles", "share_pct", "movie_pct", "top_genres"]].to_string(index=False))

    ensure_output_dirs()
    scores.to_csv(os.path.join(REPORT_DIR, "task4_cluster_selection.csv"), index=False)
    profile.to_csv(os.path.join(REPORT_DIR, "task4_cluster_profiles.csv"), index=False)
    df[["title", "type", "listed_in", "release_year", "cluster"]].to_csv(
        os.path.join(REPORT_DIR, "task4_clustered_titles.csv"), index=False
    )
    print("\nResults written to outputs/reports/task4_*.csv")


if __name__ == "__main__":
    main()
