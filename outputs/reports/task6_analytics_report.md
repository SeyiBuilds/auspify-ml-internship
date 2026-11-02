# Netflix Content Success Analytics Engine

Task 6 (Advanced) - Auspify Technologies Machine Learning Internship

## Automated insights

1. Catalogue mix: 65.3 percent of titles are films and 34.7 percent are series.
2. Largest genres: International Movies (217 titles), Dramas (173 titles), Comedies (132 titles).
3. Acquisition peaked in 2020 with 143 titles added; the most recent year on record added 123.
4. In the last three years of additions, series made up 37.4 percent of new titles against 34.7 percent across the whole catalogue.
5. Median gap between release and arrival on the platform is 1 year(s), and 54.1 percent of titles arrive within a year of release.
6. Audience positioning: Adult 45.9 percent, Teen 28.0 percent, Family 14.7 percent, Kids 11.4 percent.
7. Leading production countries: United States (258), India (80), United Kingdom (44).
8. Always guessing the largest audience band scores 46.0%, so the model's 54.0% is a lift of 7.9%, not a result that stands alone.
9. Best model: Gradient Boosting at 54.0% accuracy and 0.491 macro F1.
10. Strongest predictors of audience band: release_year, primary_genre_Kids' TV, primary_genre_Children & Family Movies, duration_zscore, duration_value.

## Model comparison

| model                    |   accuracy |   f1_macro |   f1_weighted |   cv_f1_macro_mean |   cv_f1_macro_std |   accuracy_lift_vs_baseline |
|:-------------------------|-----------:|-----------:|--------------:|-------------------:|------------------:|----------------------------:|
| Gradient Boosting        |     0.5396 |     0.4913 |        0.5279 |             0.4912 |            0.0818 |                      0.0792 |
| Logistic Regression      |     0.5324 |     0.4843 |        0.5113 |             0.4849 |            0.0626 |                      0.072  |
| Extra Trees              |     0.4748 |     0.4794 |        0.4765 |             0.4909 |            0.0637 |                      0.0144 |
| Random Forest            |     0.4604 |     0.4546 |        0.4665 |             0.5062 |            0.0404 |                      0      |
| Baseline (most frequent) |     0.4604 |     0.1576 |        0.2903 |             0.1572 |            0.0005 |                      0      |

## Classification report - Gradient Boosting

```
precision    recall  f1-score   support

       Adult       0.62      0.72      0.67        64
      Family       0.31      0.20      0.24        20
        Kids       0.71      0.62      0.67        16
        Teen       0.39      0.38      0.39        39

    accuracy                           0.54       139
   macro avg       0.51      0.48      0.49       139
weighted avg       0.52      0.54      0.53       139
```

## Figures

- `outputs/figures/task6_analytics_dashboard.png`
