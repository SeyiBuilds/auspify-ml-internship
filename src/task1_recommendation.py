"""Task 1 (Easy) - Netflix Content Recommendation System.

Builds a content-based recommender that suggests similar Netflix titles from
genres, cast, director, country and description text.

Workflow
    Step 1: Prepare content-related features.
    Step 2: Convert text data into a machine-readable format (TF-IDF).
    Step 3: Calculate content similarity scores (cosine similarity).
    Step 4: Generate recommendations for selected titles.
    Step 5: Evaluate recommendation quality.

Run:
    python src/task1_recommendation.py
    python src/task1_recommendation.py --title "Breaking Bad" --top-n 10
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import REPORT_DIR, ensure_output_dirs, load_netflix, save_figure


class NetflixRecommender:
    """Content-based recommender over a TF-IDF representation of each title."""

    # Genres and cast carry more signal than free text, so they are repeated.
    FIELD_WEIGHTS = {
        "listed_in": 3,
        "cast": 2,
        "director": 2,
        "primary_country": 1,
        "type": 1,
        "rating": 1,
        "description": 1,
    }

    def __init__(self, max_features: int = 20000, min_df: int = 1):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.df: pd.DataFrame | None = None
        self.matrix = None
        self.index_by_title: dict[str, int] = {}

    # Step 1 and 2 -------------------------------------------------------
    def _build_soup(self, df: pd.DataFrame) -> pd.Series:
        """Combine the content columns into one weighted text field per title."""
        def clean(value: object) -> str:
            if pd.isna(value):
                return ""
            return str(value).replace(",", " ").lower().strip()

        soup = pd.Series("", index=df.index, dtype=object)
        for column, weight in self.FIELD_WEIGHTS.items():
            if column not in df.columns:
                continue
            part = df[column].apply(clean)
            soup = soup.str.cat([part] * weight, sep=" ")
        return soup.str.replace(r"\s+", " ", regex=True).str.strip()

    def fit(self, df: pd.DataFrame) -> "NetflixRecommender":
        self.df = df.reset_index(drop=True)
        soup = self._build_soup(self.df)
        self.matrix = self.vectorizer.fit_transform(soup)
        # A title can appear twice in the catalogue (a film and a series of the
        # same name). Keeping the first occurrence makes lookups deterministic
        # instead of depending on row order.
        self.index_by_title = {}
        for idx, title in enumerate(self.df["title"]):
            key = str(title).strip().lower()
            self.index_by_title.setdefault(key, idx)
        duplicates = len(self.df) - len(self.index_by_title)
        if duplicates:
            print(f"  Note: {duplicates} duplicate title(s) in the catalogue, first occurrence used")
        print(f"  TF-IDF matrix: {self.matrix.shape[0]} titles x {self.matrix.shape[1]} terms")
        return self

    # Step 3 and 4 -------------------------------------------------------
    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return the top_n most similar titles to the given title."""
        if self.df is None or self.matrix is None:
            raise RuntimeError("Call fit() before recommend()")

        key = str(title).strip().lower()
        if key not in self.index_by_title:
            match = sorted((t for t in self.index_by_title if key in t), key=len)
            if not match:
                raise KeyError(f"Title not found in the catalogue: {title}")
            key = match[0]  # shortest containing title is the closest match
            print(f"  Exact title not found, using closest match: {self.df.loc[self.index_by_title[key], 'title']}")

        idx = self.index_by_title[key]
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        scores[idx] = -1.0  # never recommend the title back to itself

        top_idx = np.argsort(scores)[::-1][:top_n]
        columns = [c for c in ("title", "type", "listed_in", "release_year", "rating") if c in self.df.columns]
        result = self.df.loc[top_idx, columns].copy()
        result.insert(1, "similarity", scores[top_idx].round(4))
        return result.reset_index(drop=True)

    # Step 5 -------------------------------------------------------------
    def evaluate(self, sample_size: int = 200, top_n: int = 10, seed: int = 42) -> dict:
        """Measure how often recommendations share a genre and content type.

        There are no user ratings in this dataset, so quality is measured with
        content-level proxies: genre overlap, type match and mean similarity.
        """
        rng = np.random.default_rng(seed)
        population = len(self.df)
        sample_size = min(sample_size, population)
        sample_idx = rng.choice(population, size=sample_size, replace=False)

        genre_hits, type_hits, sims = [], [], []
        recommended_items: set[int] = set()
        for idx in sample_idx:
            scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
            scores[idx] = -1.0
            top_idx = np.argsort(scores)[::-1][:top_n]
            recommended_items.update(int(j) for j in top_idx)

            source_genres = self._genre_set(self.df.loc[idx, "listed_in"])
            matched = sum(
                1 for j in top_idx if source_genres & self._genre_set(self.df.loc[j, "listed_in"])
            )
            genre_hits.append(matched / top_n)

            source_type = self.df.loc[idx, "type"]
            type_hits.append(sum(1 for j in top_idx if self.df.loc[j, "type"] == source_type) / top_n)
            sims.append(float(np.mean(scores[top_idx])))

        # A random recommender is the reference point. Genre precision of 0.9
        # only means something next to what picking titles at random scores,
        # because a catalogue dominated by a few large genres will hit often by
        # chance alone.
        random_hits = []
        for idx in sample_idx:
            picks = rng.choice(population, size=top_n, replace=False)
            source_genres = self._genre_set(self.df.loc[idx, "listed_in"])
            matched = sum(
                1 for j in picks if j != idx and source_genres & self._genre_set(self.df.loc[j, "listed_in"])
            )
            random_hits.append(matched / top_n)
        random_precision = float(np.mean(random_hits))

        metrics = {
            "sampled_titles": sample_size,
            "top_n": top_n,
            "genre_precision_at_n": round(float(np.mean(genre_hits)), 4),
            "type_match_rate": round(float(np.mean(type_hits)), 4),
            "mean_similarity": round(float(np.mean(sims)), 4),
            # Share of the catalogue that showed up in at least one list. This
            # is capped by the sample: N queries returning top_n results can
            # surface at most N * top_n distinct titles, so on a large
            # catalogue the raw figure looks low even for a diverse
            # recommender. coverage_vs_ceiling divides by that ceiling and is
            # the number to read: 1.0 means every recommendation slot held a
            # different title, low values mean the same few keep reappearing.
            "catalogue_coverage": round(len(recommended_items) / len(self.df), 4),
            "max_possible_coverage": round(min(sample_size * top_n, len(self.df)) / len(self.df), 4),
            "coverage_vs_ceiling": round(
                len(recommended_items) / min(sample_size * top_n, len(self.df)), 4
            ),
            "random_genre_precision": round(random_precision, 4),
            "lift_over_random": round(float(np.mean(genre_hits)) - random_precision, 4),
        }
        return metrics

    @staticmethod
    def _genre_set(value: object) -> set[str]:
        if pd.isna(value):
            return set()
        return {g.strip().lower() for g in str(value).split(",") if g.strip()}


def plot_similarity_distribution(recommender: NetflixRecommender, sample_size: int = 200, top_n: int = 10) -> None:
    """Compare random pairs against recommended pairs.

    Plotting every pairwise score alone is not informative: almost all pairs of
    titles have nothing in common, so the chart is one bar at zero and a long
    empty tail. Setting the recommended pairs beside the random ones shows what
    the recommender is actually doing, which is the point of the figure.
    """
    rng = np.random.default_rng(7)
    size = min(sample_size, len(recommender.df))
    idx = rng.choice(len(recommender.df), size=size, replace=False)
    sims = cosine_similarity(recommender.matrix[idx], recommender.matrix)

    # Every sampled title scores 1.0 against itself. Those self-matches have to
    # be removed or they pile up a false spike at the right edge of the chart.
    # sims[:, idx] returns a copy, so the removal is written back by position.
    sims[np.arange(size), idx] = np.nan

    all_pairs = sims[~np.isnan(sims)]
    recommended = np.concatenate(
        [np.sort(row[~np.isnan(row)])[::-1][:top_n] for row in sims]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(all_pairs, bins=60, color="#b8b8b8", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("All title pairs (log scale)")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Pair count")

    axes[1].hist(all_pairs, bins=60, color="#b8b8b8", alpha=0.6, density=True, label="Random pair")
    axes[1].hist(recommended, bins=30, color="#3b6ea5", alpha=0.85, density=True, label=f"Top {top_n} recommended")
    axes[1].set_title("Recommended pairs against random pairs")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Task 1 - what the recommender is separating", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "task1_similarity_distribution.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Netflix content-based recommender")
    parser.add_argument("--title", default=None, help="Title to generate recommendations for")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--data", default=None, help="Dataset file name inside data/")
    args = parser.parse_args()

    print("Task 1 - Netflix Content Recommendation System")
    print("=" * 60)

    print("\nStep 1: Preparing content features")
    df = load_netflix(args.data)

    print("\nStep 2 and 3: Vectorising text and computing similarity")
    recommender = NetflixRecommender().fit(df)

    print("\nStep 4: Generating recommendations")
    demo_titles = [args.title] if args.title else list(df["title"].head(3))
    report_lines = ["Task 1 - Recommendation samples", "=" * 60, ""]

    for title in demo_titles:
        recommendations = recommender.recommend(title, top_n=args.top_n)
        header = f"Because you watched: {title}"
        print(f"\n{header}")
        print(recommendations.to_string(index=False))
        report_lines += [header, recommendations.to_string(index=False), ""]

    print("\nStep 5: Evaluating recommendation quality")
    metrics = recommender.evaluate(top_n=args.top_n)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    plot_similarity_distribution(recommender)

    ensure_output_dirs()
    report_lines += ["Evaluation metrics", "-" * 60]
    report_lines += [f"{k}: {v}" for k, v in metrics.items()]
    report_path = os.path.join(REPORT_DIR, "task1_recommendation_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))
    print(f"\nReport written to outputs/reports/{os.path.basename(report_path)}")


if __name__ == "__main__":
    main()
