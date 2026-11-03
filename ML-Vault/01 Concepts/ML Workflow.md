---
tags: [concept]
---
# ML Workflow

The steps of every ML project:

1. **Understand the problem** → regression or classification? (see [[Which Model to Choose]])
2. **Clean the data** → missing values, duplicates, outliers.
3. **Split the data** → [[Train-Validation-Test Split]]. Do this *before* anything else that learns from data.
4. **Prepare features** → encode categories, [[Feature Scaling]].
5. **Train a baseline** → simple model first (e.g. [[Linear Regression]] or [[Logistic Regression]]).
6. **Evaluate** → [[Regression Metrics]] or [[Classification Metrics]] with [[Cross-Validation]].
7. **Tune** → [[Hyperparameter Tuning]].
8. **Fix over/underfit** → [[Overfitting vs Underfitting]].
9. **Final test** → score ONCE on the test set.
10. **Predict** on new data.

> ⚠️ Golden rule: **never let the test set influence training or tuning** (data leakage).

Use a `Pipeline` to avoid leakage → see [[Sklearn Basics]].
