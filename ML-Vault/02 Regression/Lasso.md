---
tags: [model, regression, supervised, regularization]
---
# Lasso Regression

## 💡 Idea
[[Linear Regression]] + **L1 penalty**. It can push some weights to **exactly 0** → automatic **feature selection**.

## ✅ When to use
- Many features, only a few matter.
- You want a simpler, sparser model.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `alpha` | penalty strength (default 1.0) | bigger = more weights at 0 |
| `max_iter` | max iterations | raise if "did not converge" |
| `fit_intercept` | learn b | True |

## 🔧 How to tune `alpha`
- Log scale: `0.0001 ... 10`.
- Use `LassoCV`.
- Check `model.coef_`: zeros = removed features.
- **Scale features first** → [[Feature Scaling]].
- Mix of Ridge + Lasso = `ElasticNet(alpha, l1_ratio)`.

## 🧪 Syntax
```python
from sklearn.linear_model import Lasso, LassoCV

model = Lasso(alpha=0.1)
model.fit(X_train_scaled, y_train)
```

## 🎯 Predict
```python
y_pred = model.predict(X_test_scaled)
model.coef_   # see which are 0
```

## 👍 / 👎
- 👍 selects features, simple model
- 👎 unstable if features are correlated (picks one randomly)

## 🔗 Links
[[Ridge]] · [[Linear Regression]] · [[Regularization]] · [[Regression Metrics]]
