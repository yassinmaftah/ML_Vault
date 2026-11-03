---
tags: [model, regression, supervised, regularization]
---
# Ridge Regression

## 💡 Idea
[[Linear Regression]] + **L2 penalty**. It shrinks all weights toward 0 (but never exactly 0). Reduces overfitting and handles correlated features.

## ✅ When to use
- Many features, some correlated.
- Linear regression overfits.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `alpha` | penalty strength (default 1.0) | bigger = simpler model |
| `solver` | algorithm | keep `"auto"` |
| `fit_intercept` | learn b | True |

## 🔧 How to tune `alpha`
- Try on a log scale: `0.001, 0.01, 0.1, 1, 10, 100`.
- alpha too big → [[Overfitting vs Underfitting|underfit]]. Too small → overfit.
- Use `RidgeCV` (built-in cross-validation).
- **Scale features first** → [[Feature Scaling]].

## 🧪 Syntax
```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)

# auto-tune
cv_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
```

## 🎯 Predict
```python
y_pred = model.predict(X_test)
```

## 👍 / 👎
- 👍 stable, fixes overfitting, keeps all features
- 👎 does not remove features (use [[Lasso]] for that)

## 🔗 Links
[[Linear Regression]] · [[Lasso]] · [[Regularization]] · [[Hyperparameter Tuning]]
