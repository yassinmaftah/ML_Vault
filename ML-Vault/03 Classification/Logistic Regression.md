---
tags: [model, classification, supervised]
---
# Logistic Regression

## 💡 Idea
Despite the name, it is **classification**. It computes a linear score, then passes it through the **sigmoid** to get a probability between 0 and 1.
`p = 1 / (1 + e^-(w·x + b))` → if p > 0.5 → class 1.
Multi-class: softmax / one-vs-rest.

## ✅ When to use
- Binary (or multi-class) classification.
- You want probabilities and an explainable baseline.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `C` | inverse of regularization (default 1.0) | **smaller = stronger** regularization |
| `penalty` | `"l2"` (default), `"l1"`, `"elasticnet"` | l1 needs solver `liblinear`/`saga` |
| `solver` | algorithm | `lbfgs` default, `saga` for big data |
| `max_iter` | iterations (default 100) | raise if no convergence |
| `class_weight` | `"balanced"` | for imbalanced classes |

## 🔧 How to tune
- Grid on `C`: `[0.001, 0.01, 0.1, 1, 10, 100]`.
- Imbalanced data → `class_weight="balanced"`, and use F1/recall.
- [[Feature Scaling|Scale]] features.
- Change the threshold (not always 0.5) → [[Classification Metrics]].

## 🧪 Syntax
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train_scaled, y_train)
```

## 🎯 Predict
```python
y_pred  = model.predict(X_test_scaled)          # classes
y_proba = model.predict_proba(X_test_scaled)    # probabilities
```

## 👍 / 👎
- 👍 fast, gives probabilities, interpretable
- 👎 linear boundary only

## 🔗 Links
[[Classification Metrics]] · [[Regularization]] · [[SVM]] · [[Linear Regression]]
