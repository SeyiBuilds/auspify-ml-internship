# Auspify Technologies - Machine Learning Internship Submission

**Name:** Daniel Oluwaseyi Bankole
**Role:** Machine Learning Intern
**Email:** banksoluwaseyi@gmail.com
**Internship period:** 25 October 2026 to 25 November 2026
**Tasks completed:** 4 of 6 (Tasks 1, 2, 4 and 6)

## Tasks submitted

| Task | Level | Title | Script |
| --- | --- | --- | --- |
| 1 | Easy | Netflix Content Recommendation System | `src/task1_recommendation.py` |
| 2 | Easy | Content Type Prediction Model | `src/task2_content_type.py` |
| 4 | Medium | Netflix Content Segmentation | `src/task4_segmentation.py` |
| 6 | Advanced | Netflix Content Success Analytics Engine | `src/task6_analytics_engine.py` |

## Results

All results are measured on the supplied `Dataset.csv` (8,790 titles). Every
model is reported against a baseline, so each score can be read against what
doing nothing clever already achieves.

| Task | Result | Baseline | Lift |
| --- | --- | --- | --- |
| 1 Recommender | 0.980 genre precision at 10 | 0.232 random | +0.748 |
| 2 Movie vs TV Show | 0.984 accuracy (Random Forest) | 0.697 majority class | +0.287 |
| 4 Segmentation | 9 clusters, silhouette 0.347 | n/a | elbow and silhouette agree |
| 6 Analytics engine | 0.609 accuracy, 0.590 macro F1 | 0.460 majority class | +0.149 |

## How to run

```bash
pip install -r requirements.txt
python src/run_all.py
```

`src/run_all.py` runs all four tasks in sequence and prints a pass or fail line
for each. Individual tasks also run on their own, for example:

```bash
python src/task1_recommendation.py --title "Ganglands" --top-n 10
python src/task4_segmentation.py --k 6
```

Figures are written to `outputs/figures/` and reports to `outputs/reports/`.

## Folder contents

```
├── SUBMISSION.md       this file
├── README.md           project documentation and method notes
├── EXPLAINER.md        full walkthrough of every design decision
├── requirements.txt
├── data/
│   └── Dataset.csv     the supplied dataset
├── src/
│   ├── common.py       loading, cleaning and shared feature engineering
│   ├── run_all.py      runs all four tasks
│   └── task1/2/4/6 scripts
├── outputs/
│   ├── figures/        7 generated figures
│   └── reports/        generated metrics, cluster profiles and the Task 6 report
└── screenshots/        console output captured from a full run
```

## Notes on method

Three issues were found and corrected during the work, and they are documented
in full in `EXPLAINER.md` section 6:

1. **Target leakage in Task 2.** An initial 100 percent accuracy was traced to
   three features that contained the answer: `duration` states "4 Seasons" or
   "93 min", genre labels such as "Crime TV Shows" name the type inside the
   label, and every rating beginning "TV-" is a television rating. All three
   were removed and the task now scores 0.984. A feature ablation study is
   included showing the model holds at 0.959 without the director metadata
   flag, which reflects a cataloguing convention rather than content.

2. **A search artifact in Task 4.** The cluster count was initially chosen from
   a range that was too narrow, so the silhouette score was still rising at the
   edge of the search. The range was widened until the score genuinely peaks,
   and the code now warns if the chosen k sits at the boundary.

3. **Missing baselines.** Scores were initially reported with no reference
   point. A baseline model now runs inside every task.

Two properties of the supplied dataset are handled explicitly: it has 10
columns rather than the 12 of the public version (no `cast`, no `description`),
and it records missing values as the literal string "Not Given" rather than
leaving cells empty, which affects 2,588 director entries. Features derived
from absent columns are skipped rather than filled with zeros, and placeholder
values are converted to real missing values before any feature is computed.
