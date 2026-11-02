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
8. Always guessing the largest audience band scores 46.0%, so the model's 55.4% is a lift of 9.4%, not a result that stands alone.
9. Best model: Gradient Boosting at 55.4% accuracy and 0.499 macro F1.
10. Strongest predictors of audience band: primary_genre_Kids' TV, release_year, primary_genre_Children & Family Movies, duration_zscore, duration_value.

## Model comparison

| model                    |   accuracy |   f1_macro |   f1_weighted |   cv_f1_macro_mean |   cv_f1_macro_std |   accuracy_lift_vs_baseline |
|:-------------------------|-----------:|-----------:|--------------:|-------------------:|------------------:|----------------------------:|
| Gradient Boosting        |     0.554  |     0.4989 |        0.5321 |             0.4946 |            0.0644 |                      0.0936 |
| Random Forest            |     0.4892 |     0.4827 |        0.4919 |             0.4964 |            0.0504 |                      0.0288 |
| Logistic Regression      |     0.518  |     0.4788 |        0.5009 |             0.4793 |            0.0661 |                      0.0576 |
| Extra Trees              |     0.4532 |     0.4503 |        0.4579 |             0.5014 |            0.0678 |                     -0.0072 |
| Baseline (most frequent) |     0.4604 |     0.1576 |        0.2903 |             0.1572 |            0.0005 |                      0      |

## Classification report - Gradient Boosting

```
precision    recall  f1-score   support

       Adult       0.62      0.78      0.69        64
      Family       0.36      0.20      0.26        20
        Kids       0.73      0.69      0.71        16
        Teen       0.36      0.31      0.33        39

    accuracy                           0.55       139
   macro avg       0.52      0.49      0.50       139
weighted avg       0.53      0.55      0.53       139
```

## Figures

- `outputs/figures/task6_analytics_dashboard.png`
