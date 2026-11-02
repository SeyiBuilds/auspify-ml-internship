# Auspify Technologies - Machine Learning Internship

Machine Learning Intern tasks for Auspify Technologies, October to November 2026.
The programme asks for any 4 of 6 tasks. This repository completes:

| Task | Level | Title | Script |
| --- | --- | --- | --- |
| 1 | Easy | Netflix Content Recommendation System | `src/task1_recommendation.py` |
| 2 | Easy | Content Type Prediction Model | `src/task2_content_type.py` |
| 4 | Medium | Netflix Content Segmentation | `src/task4_segmentation.py` |
| 6 | Advanced | Netflix Content Success Analytics Engine | `src/task6_analytics_engine.py` |

All four tasks run on the same Netflix catalogue dataset and share one cleaning
and feature engineering module, `src/common.py`, so results stay consistent
across tasks. Every task is scored against a baseline model, so each number is
reported next to what doing nothing clever already achieves.

`EXPLAINER.md` walks through every design decision in plain language, including
three defects found during review and how they were fixed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset at `data/Dataset.csv`. The loader also accepts
`data/netflix_titles.csv` and falls back to the small sample committed for
testing.

## Running the tasks

Run everything at once:

```bash
python src/run_all.py
```

Or one at a time:

```bash
python src/task1_recommendation.py
python src/task1_recommendation.py --title "Breaking Bad" --top-n 10

python src/task2_content_type.py

python src/task4_segmentation.py
python src/task4_segmentation.py --k 6

python src/task6_analytics_engine.py
```

Figures are written to `outputs/figures/` and reports to `outputs/reports/`.

## Task 1 - Content Recommendation System

Content-based recommender. Genres, cast, director, country, rating and the
description are combined into one weighted text field per title, vectorised
with TF-IDF over unigrams and bigrams, and compared with cosine similarity.
Genres are weighted three times and cast and director twice, because those
carry more signal than free text.

The dataset has no user ratings, so quality is measured with content-level
proxies instead of click-through: genre precision at N (share of
recommendations sharing at least one genre with the source title), type match
rate, catalogue coverage and mean similarity.

A random recommender is scored alongside it. Genre precision of roughly 0.91
only means something next to the 0.25 that random picks achieve on a catalogue
dominated by a few large genres, so the lift over random is the result worth
quoting.

## Task 2 - Content Type Prediction Model

Binary classification of Movie against TV Show, comparing Logistic Regression,
Decision Tree, Random Forest and Gradient Boosting over a shared preprocessing
pipeline (median imputation and scaling for numeric columns, one-hot encoding
for categoricals).

**Leakage note.** The first version scored 100 percent accuracy, which was a
bug rather than a result. Three features contained the answer: `duration`
states "4 Seasons" or "93 min", genre labels such as "Crime TV Shows" name the
type inside the label, and every rating beginning "TV-" is by definition a
television rating. Duration is now excluded, genre labels are stripped of Movie
and TV wording, and raw ratings are replaced by the audience band. Accuracy
settles around 0.97 against a 0.65 baseline.

**Ablation.** The script retrains with feature groups removed to show what each
one is worth. `has_director` drives roughly 40 percent of the decisions, which
reflects a cataloguing habit rather than content: Netflix records a director for
films and usually leaves it blank for series. Removing it costs about 1.5
points, which is the evidence the rest of the model is doing real work.

## Task 4 - Content Segmentation

K-Means clustering over scaled numeric features, one-hot encoded content type
and audience band, and a TruncatedSVD compression of the TF-IDF genre matrix.
The cluster count is chosen by comparing inertia (elbow) and silhouette scores
across a range of k, taking the simplest k within one percent of the best
score. The script warns if the winning k sits at the edge of the search range,
because a score that is still rising means the search was measured rather than
the data. Clusters are reported with a readable profile: size, film share,
median release year, dominant audience band, top genres and example titles.

## Task 6 - Content Success Analytics Engine

End-to-end pipeline covering advanced feature engineering, four models,
performance comparison, automated written insights and a six-panel dashboard.

**Target note.** The dataset carries no viewership or revenue column, so actual
success cannot be measured and claiming otherwise would be dishonest. The
modelled target is the audience band a title is positioned for (Kids, Family,
Teen, Adult), which is the commercial decision the catalogue does record.
Observed catalogue patterns are reported separately by the insight engine.
Accuracy lands around 0.55 across four classes against a 0.46 baseline, and the
generated report states that nine point lift outright rather than quoting the
raw accuracy alone.

## Project layout

```
auspify-ml/
├── data/                  dataset
├── src/
│   ├── common.py          loading, cleaning, shared feature engineering
│   ├── run_all.py         runs all four tasks and reports pass or fail
│   ├── task1_recommendation.py
│   ├── task2_content_type.py
│   ├── task4_segmentation.py
│   └── task6_analytics_engine.py
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── EXPLAINER.md           plain-language walkthrough of every decision
└── README.md
```

## Author

Daniel Oluwaseyi Bankole
Machine Learning Intern, Auspify Technologies
