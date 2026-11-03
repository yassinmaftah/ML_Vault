---
tags: [model, classification, regression, supervised, trees, ensemble]
---
# Random Forest

## 💡 Idea
Many [[Decision Tree|decision trees]] trained in parallel (**bagging**):
- each tree sees a random sample of rows (bootstrap)
- each split sees a random subset of features
- Final answer: **vote** (classification) or **average** (regression).

Averaging many trees → **lower variance**, less overfitting.

## ✅ When to use
- Strong default for tabular data. Little tuning needed.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `n_estimators` | number of trees (default 100) | more = better but slower. 200-500 |
| `max_depth` | tree depth | limit to reduce overfit |
| `max_features` | features per split | `"sqrt"` (class), 1.0 or 1/3 (reg) |
| `min_samples_leaf` | min samples per leaf | bigger = smoother |
| `bootstrap` | sample with replacement | True |
| `oob_score` | free validation score | True |
| `class_weight` | `"balanced"` | imbalanced |
| `n_jobs` | cores | `-1` |

## 🔧 How to tune
1. `n_estimators` high enough (score stops improving).
2. `max_depth`, `min_samples_leaf`, `max_features`.
3. Overfitting? → lower depth / raise `min_samples_leaf`.

## 🧪 Syntax
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier(n_estimators=300, max_depth=10,
                               n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
model.feature_importances_
```

## 👍 / 👎
- 👍 robust, accurate, no scaling, gives feature importance
- 👎 slower and bigger than one tree, less accurate than boosting often

## 🔗 Links
[[Decision Tree]] · [[Gradient Boosting]] (boosting vs bagging) · [[Bias-Variance Tradeoff]]
