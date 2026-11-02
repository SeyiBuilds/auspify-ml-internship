# Explainer

A walkthrough of what this project does and why each decision was made, written
so you can read the code afterwards and defend it if anyone asks.

---

## 1. Running it

```bash
pip install -r requirements.txt
python src/run_all.py
```

That runs all four tasks and prints a pass or fail line for each. Individual
tasks run on their own too:

```bash
python src/task1_recommendation.py --title "Jaguar" --top-n 10
python src/task4_segmentation.py --k 6
```

Figures land in `outputs/figures/`, reports in `outputs/reports/`.

To use the real internship dataset, drop it at `data/Dataset.csv`. The loader
checks for that name first and falls back to the sample committed here.

---

## 2. The dataset

### The important part: their file is not the Kaggle file

The `Dataset.csv` supplied for the internship has **10 columns, not 12**. There
is no `cast` column and no `description` column. It also writes every missing
value as the literal string `"Not Given"` instead of leaving the cell empty,
and formats dates as `9/25/2021` rather than `September 25, 2021`.

All three matter:

- Building a feature from a column that does not exist gives you a column of
  zeros. The model treats it as data, it carries no information, and nothing
  errors to tell you.
- `"Not Given"` is a string, so `isna()` reports nothing missing. The
  `has_director` flag would have come out as 1 for all 700 rows while looking
  completely normal. About a third of directors are `"Not Given"`.
- The loader converts those placeholders to real missing values before any
  feature is computed, which is why ordering inside `load_netflix` matters.

There is a lesson here worth keeping: a schema assumption is a bug waiting to
happen. The pipeline was written against the 12 column public dataset and would
have run on their file without a single error while quietly producing worse
models.

### The columns

One row per Netflix title:

| Column | What it holds |
| --- | --- |
| `show_id` | unique id |
| `type` | Movie or TV Show |
| `title`, `director` | who made it (`cast` is absent from the internship export) |
| `country` | production country, sometimes several |
| `date_added` | when it arrived on the platform |
| `release_year` | when it was originally released |
| `rating` | age certificate, for example TV-MA or PG-13 |
| `duration` | "93 min" for films, "4 Seasons" for series |
| `listed_in` | genres, comma separated |
| `description` | one paragraph synopsis (absent from the internship export) |

The supplied `Dataset.csv` holds 8,790 titles, 6,126 films and 2,664 series.
Every number below comes from that file.

**There is no viewership, revenue or user rating column.** That single fact
shapes most of the design decisions below.

Two samples are committed. `schema_match_sample.csv` matches the internship
schema exactly and is what development runs against. `sample_netflix_titles.csv`
keeps all 12 columns, so the code is exercised against both shapes.

---

## 3. The shared module: `src/common.py`

Every task imports from here, so cleaning happens once and all four tasks agree
with each other. If a reviewer asks "did you clean the data consistently", this
file is the answer.

What it builds on top of the raw columns:

- **`duration_value`, `duration_minutes`, `duration_seasons`** split the mixed
  duration column into numbers.
- **`duration_zscore`** expresses length relative to other titles *of the same
  type*. This matters more than it looks. A film is measured in minutes (about
  90 to 120) and a series in seasons (usually 1 to 5). Putting both in one
  column and handing it to a model compares 93 against 4 as if they were the
  same unit. The z-score asks "is this long *for a film*" or "long *for a
  series*", which is the question you actually mean.
- **`year_added`, `month_added`, `years_to_platform`** are the time features.
  `years_to_platform` is how long after release a title reached Netflix, which
  separates fresh acquisitions from back catalogue.
- **`genre_count`, `cast_count`, `country_count`** count the comma separated
  fields.
- **`primary_genre`, `primary_country`** take the first listed value.
- **`audience_band`** collapses the age certificates into Kids, Family, Teen
  and Adult. TV-MA and R both become Adult.

---

## 4. Concepts you need, in plain terms

**TF-IDF.** A way of turning text into numbers. It scores a word highly if it
appears often in one document but rarely across the whole collection. "The"
appears everywhere so it scores near zero. "Zombie" appears in few descriptions
so it scores high and becomes a useful signal.

**Cosine similarity.** Once every title is a vector of numbers, this measures
the angle between two vectors. 1.0 means identical direction, 0 means nothing
in common. It ignores length, which is what you want: a long description should
not count as more similar just for being long.

**Train/test split.** The model learns from 80 percent of the rows and is
scored on the 20 percent it never saw. Scoring a model on data it memorised
tells you nothing.

**Cross validation.** The same thing repeated five times over different slices,
so a single lucky split cannot flatter the result. In the output tables this is
`cv_accuracy_mean` and its standard deviation. If the standard deviation is
large, the model is unstable.

**Precision and recall.** Precision is "when it said TV Show, how often was it
right". Recall is "of all the real TV Shows, how many did it catch". F1 is the
balance of the two. Macro F1 averages across classes equally, so a small class
cannot be ignored.

**Baseline.** A model that does nothing clever. Here it always answers the most
common class. Every real model is measured against it. This is covered in
detail in section 6 because it is the most important idea in the project.

**Data leakage.** When a feature secretly contains the answer. The model scores
brilliantly and is useless in practice. Also covered in section 6.

**K-Means clustering.** Unsupervised, meaning there are no labels. You tell it
how many groups to find, it places that many centres and shuffles them until
each title sits with its nearest centre.

**Silhouette score.** Measures whether clusters are actually separated.
Runs from -1 to 1. Around 0.27, which is what this dataset gives, means the
groups are real but overlapping, which is honest for a content catalogue where
a title can be a comedy and a drama at once.

---

## 5. The four tasks

### Task 1: Recommendation system

**What it does.** Give it a title, it returns similar titles.

**How.** Genres, director, country, type and rating are pasted into one text
field per title, along with cast and description when the dataset has them. Genres are repeated three times and cast
and director twice, so TF-IDF weights them more heavily than loose description
wording. Then every title is compared to every other with cosine similarity.

**Evaluating it without user data.** There are no click records, so the usual
recommender metrics are unavailable. Three content level proxies are used
instead:

- `genre_precision_at_n`: of the titles recommended, what share share at least
  one genre with the original.
- `type_match_rate`: how often films are recommended for films.
- `catalogue_coverage`: how much of the catalogue ever gets recommended. A
  recommender that only ever suggests the same twenty popular titles would
  score well on precision and badly here.

**The number that matters.** Genre precision is 0.980, but a random
recommender already scores 0.232, because a handful of genres dominate the
catalogue. The lift over random, 0.748, is the real result. Quote that.

**On coverage.** The raw coverage figure is 0.155, which looks alarming until
you notice the ceiling: 200 sampled queries returning 10 results each can
surface at most 2,000 of 8,790 titles, a ceiling of 0.227. Measured against
that ceiling the recommender reaches 0.68, meaning roughly two thirds of its
recommendation slots held a distinct title. A metric without its ceiling is
another version of a score without a baseline.

### Task 2: Movie or TV Show

**What it does.** Predicts the content type from the other columns.

**Models compared.** Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, plus the baseline. Random Forest wins at 0.984 accuracy
against a 0.697 baseline.

**The ablation table.** This is the part worth understanding. It retrains the
model with feature groups removed, to see what each one is really worth:

| Feature set | What it tells you |
| --- | --- |
| All features | the headline number |
| Without metadata flags | drops `has_director` and `has_cast` |
| Without genre | drops the genre label |
| Without both | the honest floor |

`has_director` alone accounts for roughly 40 percent of the model's decisions.
That is not because directors predict content type, but because Netflix records
a director for films and usually leaves the field blank for series. The model
was partly learning a data entry habit. Removing it costs 2.5 points, 0.984
down to 0.959, which is the evidence that the rest of the model is doing real
work. Removing both the flags and genre drops it to 0.788, which is where the
genuinely hard part of the problem sits.

### Task 4: Content segmentation

**What it does.** Groups titles into clusters without being told what the
groups are.

**Features used.** Scaled numeric columns, one hot encoded type and audience
band, and the genre text compressed from hundreds of columns down to eight with
TruncatedSVD. Compression is needed because distance calculations behave badly
in very high dimensions.

**Choosing the number of clusters.** Two checks are run across a range of k:
inertia, which falls as k rises and is read by looking for the elbow, and
silhouette, which rewards genuine separation. The selection rule takes the
*smallest* k within one percent of the best silhouette, because five clusters
you can explain beat thirteen that are marginally tighter.

**The output.** Each cluster gets a readable label built from its film share,
median release year, dominant audience band and top genres, plus example
titles. That turns "cluster 3" into "Recent Movie - Dramas (Adult)".

### Task 6: Analytics engine

**What it does.** The full pipeline: feature engineering, four models plus
baseline, comparison, written insights, and a six panel dashboard.

**What it predicts, and why.** The task brief says "content success", but there
is no success column in the data. Inventing one would be dishonest. The target
is instead the audience band a title is positioned for, Kids through Adult,
which is a real commercial decision the catalogue does record. Catalogue
patterns are reported separately by the insight engine.

**Reading the result.** 0.625 accuracy across four classes against a baseline
of 0.460, a lift of 16 points. Genre and release year carry most of the signal.
Gradient Boosting has the best accuracy while Random Forest has the best macro
F1 (0.590), which is worth understanding: accuracy rewards getting the large
Adult class right, macro F1 weights all four classes equally and so rewards
handling the small Kids class. Which one matters depends on what the model is
for.

**The insight engine** writes sentences from the data: catalogue mix, largest
genres, the peak acquisition year, how quickly titles reach the platform,
audience positioning and leading production countries.

---

## 6. Three things that were wrong, and how they were caught

Worth reading properly. These are the parts a reviewer is most likely to probe,
and they are the most useful things in the project to have learned.

### Leakage in Task 2

The first version scored **100 percent accuracy**. A perfect score on real data
is almost never good news, it means a feature contains the answer. Three did:

1. `duration` says "4 Seasons" or "93 min". The unit alone gives it away.
2. Genre labels read "Crime TV Shows" and "Horror Movies". The label contains
   the target word.
3. Every rating beginning "TV-" is by definition a television rating.

Fixes: duration dropped, Movie and TV wording stripped from genre labels with a
regular expression, and raw ratings replaced by the audience band. Accuracy
fell to about 0.97, which is a real number rather than a bug.

The general lesson: when a score looks too good, go and find the leak before
celebrating.

### A search artifact in Task 4

The cluster count was chosen by taking the highest silhouette score across k
from 2 to 9, and it picked 9 every time. Looking at the curve, the score was
still rising at the edge of the range. Nothing had been discovered, the search
had simply been stopped at 9.

Fixes: the range was widened to 15, where the score genuinely plateaus, and the
code now prints a warning if the winner sits at the edge of the range. The
selection rule also prefers the simplest k within one percent of the best.

The general lesson: if your answer sits at the boundary of what you searched,
you measured your search, not the data.

### A schema assumption, caught by checking the real file

Everything above was built against the public 12 column Kaggle dataset. Opening
the actual internship file showed 10 columns, placeholder strings instead of
empty cells, and a different date format. None of that would have raised an
error. It would simply have produced a model with dead features and a
`has_director` flag stuck at 1.

The general lesson: look at the real file before trusting that your assumptions
about it hold.

### Missing baselines everywhere

Three scores were being reported with nothing to compare them to: 0.97 accuracy
in Task 2, 0.55 in Task 6, 0.91 genre precision in Task 1.

Those numbers mean very different things once a baseline is added. Task 2's
0.984 against a 0.697 baseline is strong. Task 6's 0.625 against a 0.460
baseline is respectable. Task 1's 0.980 against 0.232 random is excellent.
Without the reference point, the smallest achievement looked like the best
one.

A baseline model now runs inside every task and appears in every results table.

---

## 7. Questions to be ready for

- *Why did you pick these four tasks?* Two easy, one medium, one advanced,
  covering recommendation, classification, clustering and an end to end
  pipeline. The brief asks for any four of six.
- *Why is Task 6 accuracy only 55 percent?* Four classes, an imbalanced target,
  and no behavioural data. The baseline is 46 percent, so the model adds nine
  points. The honest framing is in the report.
- *How do you know Task 2 is not leaking?* The ablation table. Remove the
  metadata flags and it still scores about 0.957, so it is not resting on the
  director convention.
- *Why K-Means and not something else?* It is fast, it is interpretable, and
  the brief names it. The selection of k is justified by two independent
  metrics rather than by eye.
- *What would you do next with more data?* Bring in viewership or user ratings
  so "success" can be measured instead of proxied, and try a hybrid
  recommender that combines content similarity with behaviour.
