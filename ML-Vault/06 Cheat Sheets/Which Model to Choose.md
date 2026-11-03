---
tags: [cheatsheet]
---
# Which Model to Choose?

```
Do you have labels (y)?
├── YES (supervised)
│   ├── y is a number → Regression
│   │     1. [[Linear Regression]] (baseline)
│   │     2. Too many features? → [[Ridge]] / [[Lasso]]
│   │     3. Non-linear / best score → [[Random Forest]], [[XGBoost]], [[HistGradientBoosting]]
│   │     4. Small data, curved → [[SVR]]
│   └── y is a class → Classification
│         1. [[Logistic Regression]] (baseline)
│         2. Small data → [[SVM]], [[KNN]]
│         3. Tabular, best score → [[Random Forest]], [[XGBoost]], [[HistGradientBoosting]]
│         4. Need to explain → [[Decision Tree]]
└── NO (unsupervised)
      ├── Round clusters, know k → [[K-Means]]
      └── Any shape / outliers, unknown k → [[DBSCAN]]
```

## Quick rules
- Always start with a **simple baseline**.
- Tabular data → tree ensembles usually win.
- Big data → [[HistGradientBoosting]] or [[XGBoost]].
- Need to explain → [[Linear Regression]], [[Logistic Regression]], [[Decision Tree]].
- Distance-based models → [[Feature Scaling]].
- Check with [[Cross-Validation]], then fix [[Overfitting vs Underfitting]].

Back to [[00 Home]]
