---
tags: [model, classification, regression, supervised, trees, ensemble]
---
# Gradient Boosting

## 💡 Idea
Build trees **one after another** (**boosting**). Each new small tree fixes the **errors (residuals)** of the previous ones.
`prediction = tree1 + lr·tree2 + lr·tree3 + ...`
Reduces **bias**. Compare: [[Random Forest]] builds trees in parallel.

## ✅ When to use
- Best accuracy on tabular data. Medium-size data.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `n_estimators` | number of trees | more = better fit, risk overfit |
| `learning_rate` | step size (default 0.1) | **small lr needs more trees** |
| `max_depth` | depth of each tree | **shallow: 3-5** |
| `subsample` | % rows per tree | 0.5-1.0 (<1 = stochastic, less overfit) |
| `min_samples_leaf` | min leaf size | regularizes |
| `n_iter_no_change` | early stopping | stops when no gain |

## 🔧 How to tune
1. Fix `learning_rate` = 0.1, find `n_estimators` (use early stopping).
2. Tune `max_depth`, `min_samples_leaf`.
3. Add `subsample` = 0.8.
4. Lower `learning_rate` (0.05, 0.01) and raise `n_estimators`.

## 🧪 Syntax
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

model = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=3,
    subsample=0.8, n_iter_no_change=10, random_state=42
)
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

## 👍 / 👎
- 👍 very accurate, flexible
- 👎 slow (sequential), many params, overfits if too many trees

## 🔗 Links
[[XGBoost]] (faster, more features) · [[HistGradientBoosting]] (fast, big data) · [[Random Forest]] · [[Overfitting vs Underfitting]]
