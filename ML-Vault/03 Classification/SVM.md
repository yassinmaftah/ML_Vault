---
tags: [model, classification, supervised]
---
# SVM (Support Vector Machine)

## 💡 Idea
Find the boundary (hyperplane) with the **largest margin** between classes. Only the closest points (**support vectors**) matter. With a **kernel**, it can draw curved boundaries.

## ✅ When to use
- Small/medium data, many features, clear margin between classes.

## ⚙️ Params
| Param          | Meaning                                 | Tip                                                   |
| -------------- | --------------------------------------- | ----------------------------------------------------- |
| `C`            | error penalty                           | small = wide margin (underfit), big = tight (overfit) |
| `kernel`       | `"linear"`, `"rbf"` (default), `"poly"` | linear for text/high-dim                              |
| `gamma`        | reach of one point (rbf)                | small = smooth, big = overfit                         |
| `degree`       | for poly kernel                         | 2-4                                                   |
| `probability`  | enable `predict_proba`                  | slower                                                |
| `class_weight` | `"balanced"`                            | imbalanced data                                       |

## 🔧 How to tune
- Grid on `C` [0.1, 1, 10, 100] and `gamma` ["scale", 0.001, 0.01, 0.1, 1].
- **Always scale** → [[Feature Scaling]].
- Big data → use `LinearSVC` (much faster).

## 🧪 Syntax
```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred = model.predict(X_test)
# for probabilities: SVC(probability=True) then predict_proba
```

## 👍 / 👎
- 👍 strong on small data, flexible with kernels
- 👎 slow on big data, needs scaling, no direct probabilities

## 🔗 Links
[[SVR]] · [[Logistic Regression]] · [[Classification Metrics]] · [[Hyperparameter Tuning]]
