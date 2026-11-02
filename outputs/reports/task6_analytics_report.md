# Netflix Content Success Analytics Engine

Task 6 (Advanced) - Auspify Technologies Machine Learning Internship

## Automated insights

1. Catalogue mix: 69.7 percent of titles are films and 30.3 percent are series.
2. Largest genres: International Movies (2752 titles), Dramas (2426 titles), Comedies (1674 titles).
3. Acquisition peaked in 2019 with 2016 titles added; the most recent year on record added 1498.
4. In the last three years of additions, series made up 31.4 percent of new titles against 30.3 percent across the whole catalogue.
5. Median gap between release and arrival on the platform is 1 year(s), and 55.0 percent of titles arrive within a year of release.
6. Audience positioning: Adult 46.0 percent, Teen 30.4 percent, Family 13.2 percent, Kids 10.4 percent.
7. Leading production countries: United States (3240), India (1057), United Kingdom (638).
8. Always guessing the largest audience band scores 46.0%, so the model's 60.9% is a lift of 14.9%, not a result that stands alone.
9. Best model: Random Forest at 60.9% accuracy and 0.590 macro F1.
10. Strongest predictors of audience band: genre_frequency, duration_value, duration_zscore, primary_genre_Kids' TV, primary_genre_Children & Family Movies.

## Model comparison

| model                    |   accuracy |   f1_macro |   f1_weighted |   cv_f1_macro_mean |   cv_f1_macro_std |   accuracy_lift_vs_baseline |
|:-------------------------|-----------:|-----------:|--------------:|-------------------:|------------------:|----------------------------:|
| Random Forest            |     0.6091 |     0.5902 |        0.6111 |             0.6035 |            0.0151 |                      0.1487 |
| Extra Trees              |     0.601  |     0.5826 |        0.6025 |             0.5973 |            0.0105 |                      0.1406 |
| Gradient Boosting        |     0.6246 |     0.5787 |        0.6011 |             0.5859 |            0.0151 |                      0.1642 |
| Logistic Regression      |     0.6102 |     0.5689 |        0.5922 |             0.5718 |            0.0154 |                      0.1498 |
| Baseline (most frequent) |     0.4604 |     0.1576 |        0.2903 |             0.1576 |            0      |                      0      |

## Classification report - Random Forest

```
precision    recall  f1-score   support

       Adult       0.71      0.69      0.70       802
      Family       0.38      0.42      0.40       230
        Kids       0.74      0.73      0.74       181
        Teen       0.52      0.53      0.52       529

    accuracy                           0.61      1742
   macro avg       0.59      0.59      0.59      1742
weighted avg       0.61      0.61      0.61      1742
```

## Figures

- `outputs/figures/task6_analytics_dashboard.png`
