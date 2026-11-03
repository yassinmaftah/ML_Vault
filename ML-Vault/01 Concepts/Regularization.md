---
tags: [concept]
---
# Regularization

Add a **penalty** on big weights so the model stays simple → fights [[Overfitting vs Underfitting|overfitting]].

| Type | Penalty | Effect | Note |
|---|---|---|---|
| **L2** | sum of w² | shrinks weights | [[Ridge]] |
| **L1** | sum of \|w\| | sets some weights to 0 (feature selection) | [[Lasso]] |
| **L1 + L2** | both | mix | ElasticNet |

## Strength parameter
- `alpha` (Ridge, Lasso): **bigger = stronger** regularization.
- `C` ([[Logistic Regression]], [[SVM]]): **smaller = stronger** (C is the inverse).
- `reg_alpha`, `reg_lambda` ([[XGBoost]]), `l2_regularization` ([[HistGradientBoosting]]).

⚠️ Always [[Feature Scaling|scale]] features first, otherwise the penalty is unfair.

Related: [[Bias-Variance Tradeoff]]
