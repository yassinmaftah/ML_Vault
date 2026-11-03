---
tags: [model, regression, supervised]
---
# Linear Regression

## 💡 Idea
Find the best straight line (or plane) through the data:
`ŷ = w1·x1 + w2·x2 + ... + b`
It minimizes the **sum of squared errors** (least squares).

## ✅ When to use
- Target is a number.
- Relation between X and y is roughly linear.
- You want a simple, explainable baseline.

## ⚙️ Params
Almost none (that's why it overfits easily with many features).
| Param | Meaning |
|---|---|
| `fit_intercept` | learn `b` (default True) |
| `n_jobs` | CPU cores |

## 🔧 How to improve
- No regularization here → use [[Ridge]] or [[Lasso]].
- Curved relation → `PolynomialFeatures`.
- Check outliers, [[Feature Scaling|scale]] features to compare coefficients.

## 🧪 Syntax
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred = model.predict(X_test)
model.coef_        # one weight per feature
model.intercept_   # b
model.score(X_test, y_test)   # R²
```

## 👍 / 👎
- 👍 fast, simple, easy to explain
- 👎 only linear, sensitive to outliers, overfits with many/correlated features

## 🔗 Links
[[Ridge]] · [[Lasso]] · [[Regression Metrics]] · [[Regularization]] · [[Overfitting vs Underfitting]]
