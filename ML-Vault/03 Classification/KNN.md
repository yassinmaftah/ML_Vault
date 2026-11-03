---
tags: [model, classification, regression, supervised]
---
# KNN (K-Nearest Neighbors)

## 💡 Idea
To predict a new point: find its **k closest** training points, then
- classification → **majority vote**
- regression → **average** of their values.

No real training: it just stores the data ("lazy learner").

## ✅ When to use
- Small data, simple baseline, low dimensions.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `n_neighbors` (k) | number of neighbors (default 5) | small k = overfit, big k = underfit |
| `weights` | `"uniform"` or `"distance"` | distance = closer points count more |
| `metric` / `p` | distance type | p=2 Euclidean, p=1 Manhattan |

## 🔧 How to tune
- Try k from 1 to 30 (odd numbers for binary classification, avoids ties).
- Plot CV score vs k → pick the best.
- **Must scale features** → [[Feature Scaling]].
- Slow at prediction time on big data.

## 🧪 Syntax
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

model = KNeighborsClassifier(n_neighbors=5, weights="distance")
model.fit(X_train_scaled, y_train)
```

## 🎯 Predict
```python
y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)
```

## 👍 / 👎
- 👍 very simple, no training, non-linear
- 👎 slow prediction, needs scaling, bad in high dimensions

## 🔗 Links
[[Bias-Variance Tradeoff]] · [[Classification Metrics]] · [[K-Means]] (do not confuse: KNN is supervised)
