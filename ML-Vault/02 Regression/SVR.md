---
tags: [model, regression, supervised]
---
# SVR (Support Vector Regression)

## 💡 Idea
The regression version of [[SVM]]. It fits a line/curve inside a tube of width **epsilon**. Errors inside the tube are ignored. Only points outside the tube (support vectors) shape the model.

## ✅ When to use
- Small/medium data, non-linear relation (with RBF kernel).

## ⚙️ Params
| Param     | Meaning                                 | Tip                                     |
| --------- | --------------------------------------- | --------------------------------------- |
| `kernel`  | `"linear"`, `"rbf"` (default), `"poly"` | start with rbf                          |
| `C`       | penalty for errors                      | bigger = fit data harder (overfit risk) |
| `epsilon` | tube width                              | bigger = simpler model                  |
| `gamma`   | reach of one point (rbf)                | bigger = more complex                   |

## 🔧 How to tune
- Grid on `C` [0.1, 1, 10, 100], `gamma` ["scale", 0.01, 0.1, 1], `epsilon` [0.01, 0.1, 1].
- **Scale X (and often y)** → [[Feature Scaling]].
- Slow on big data (> 50k rows).

## 🧪 Syntax
```python
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred = model.predict(X_test)
```

## 👍 / 👎
- 👍 handles non-linear data, robust to outliers
- 👎 slow on big data, needs scaling, hard to interpret

## 🔗 Links
[[SVM]] · [[Regression Metrics]] · [[Hyperparameter Tuning]]
