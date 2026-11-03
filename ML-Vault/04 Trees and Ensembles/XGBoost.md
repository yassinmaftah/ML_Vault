---
tags: [model, classification, regression, supervised, trees, ensemble]
---
# XGBoost (Extreme Gradient Boosting)

## 💡 Idea
An optimized version of [[Gradient Boosting]]: faster, built-in **L1/L2 regularization**, handles missing values, uses multiple cores.

## ✅ When to use
- Competitions, large tabular data, when you need top performance.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `n_estimators` | number of trees | use early stopping |
| `learning_rate` (eta) | step size | 0.01-0.3 |
| `max_depth` | tree depth | 3-10 |
| `subsample` | % rows per tree | 0.6-1.0 |
| `colsample_bytree` | % features per tree | 0.6-1.0 |
| `min_child_weight` | min weight in a leaf | bigger = more conservative |
| `gamma` | min gain to split | bigger = more conservative |
| `reg_alpha` | L1 | bigger = simpler |
| `reg_lambda` | L2 (default 1) | bigger = simpler |
| `scale_pos_weight` | imbalanced binary | neg / pos ratio |

## 🔧 How to tune
1. `learning_rate=0.1`, large `n_estimators` + **early stopping**.
2. Tune `max_depth` and `min_child_weight`.
3. Tune `subsample` and `colsample_bytree`.
4. Tune `reg_alpha` / `reg_lambda` / `gamma`.
5. Lower `learning_rate`, raise `n_estimators`.
Tip: use RandomizedSearch or Optuna → [[Hyperparameter Tuning]].

## 🧪 Syntax
```python
from xgboost import XGBClassifier, XGBRegressor

model = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, eval_metric="logloss", random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
model.best_iteration   # best number of trees
```
Install: `pip install xgboost`

## 🎯 Predict
```python
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
model.feature_importances_
```

## 👍 / 👎
- 👍 top accuracy, fast, regularized, handles NaN
- 👎 many params to tune, needs a separate install

## 🔗 Links
[[Gradient Boosting]] · [[HistGradientBoosting]] · [[Random Forest]] · [[Regularization]] · [[Train-Validation-Test Split]]
