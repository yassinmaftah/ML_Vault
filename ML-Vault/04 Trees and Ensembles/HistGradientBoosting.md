---
tags: [model, classification, regression, supervised, trees, ensemble]
---
# HistGradientBoosting

## 💡 Idea
A **fast** [[Gradient Boosting]] from sklearn (like LightGBM). It first puts feature values into **bins (histograms)** (up to 255), so splits are much quicker.

## ✅ When to use
- Big data (> 10k rows), you want speed inside sklearn (no extra install).

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `max_iter` | number of trees | (replaces `n_estimators`) |
| `learning_rate` | step size (default 0.1) | smaller + more iter |
| `max_depth` | tree depth | default None |
| `max_leaf_nodes` | leaves per tree (default 31) | main complexity control |
| `min_samples_leaf` | min samples per leaf (default 20) | bigger = simpler |
| `l2_regularization` | L2 penalty | bigger = simpler |
| `early_stopping` | `"auto"` (on if > 10k rows) | True |
| `categorical_features` | handle categories natively | `"from_dtype"` |

## 🔧 How to tune
1. Turn on `early_stopping=True`, set big `max_iter`.
2. Tune `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`.
3. Add `l2_regularization`.

## 🧪 Syntax
```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

model = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.1, max_leaf_nodes=31,
    early_stopping=True, random_state=42
)
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

## 👍 / 👎
- 👍 very fast, handles NaN natively, no scaling
- 👎 slight overfit on small data, no `feature_importances_` (use permutation importance)

## 🔗 Links
[[Gradient Boosting]] · [[XGBoost]] · [[Random Forest]] · [[Hyperparameter Tuning]]
