# Netflix Content Success Analytics Engine

Task 6 (Advanced) - Auspify Technologies Machine Learning Internship

## Automated insights

1. Catalogue mix: 80.2 percent of titles are films and 19.8 percent are series.
2. Largest genres: Dramas (44 titles), International Movies (43 titles), Comedies (33 titles).
3. Acquisition peaked in 2019 with 38 titles added; the most recent year on record added 3.
4. In the last three years of additions, series made up 18.8 percent of new titles against 19.8 percent across the whole catalogue.
5. Median gap between release and arrival on the platform is 1 year(s), and 55.7 percent of titles arrive within a year of release.
6. Audience positioning: Adult 49.6 percent, Teen 24.0 percent, Family 17.8 percent, Kids 8.5 percent.
7. Leading production countries: United States (51), India (10), United Kingdom (9).
8. Best model: Logistic Regression at 42.3% accuracy and 0.401 macro F1, ahead of Gradient Boosting on macro F1 by 0.079.

## Model comparison

| model               |   accuracy |   f1_macro |   f1_weighted |   cv_f1_macro_mean |   cv_f1_macro_std |
|:--------------------|-----------:|-----------:|--------------:|-------------------:|------------------:|
| Logistic Regression |     0.4231 |     0.4011 |        0.4343 |             0.4244 |            0.0731 |
| Gradient Boosting   |     0.3462 |     0.3222 |        0.3385 |             0.4193 |            0.0676 |
| Extra Trees         |     0.3077 |     0.2635 |        0.3035 |             0.4553 |            0.0575 |
| Random Forest       |     0.1923 |     0.1761 |        0.1906 |             0.5662 |            0.0439 |

## Classification report - Logistic Regression

```
precision    recall  f1-score   support

       Adult       0.62      0.38      0.48        13
      Family       0.29      0.40      0.33         5
        Kids       0.25      0.50      0.33         2
        Teen       0.43      0.50      0.46         6

    accuracy                           0.42        26
   macro avg       0.40      0.45      0.40        26
weighted avg       0.49      0.42      0.43        26
```

## Figures

- `outputs/figures/task6_analytics_dashboard.png`
