---
tags: [concept, code]
---
# Feature Scaling

Put features on the same scale (e.g. age 0-100 vs salary 0-100000).

| Scaler           | Result          | Use            |
| ---------------- | --------------- | -------------- |
| `StandardScaler` | mean 0, std 1   | default choice |
| `MinMaxScaler`   | range 0 to 1    | bounded data   |
| `RobustScaler`   | uses median/IQR | many outliers  |

## Who needs scaling?
| Needs it ✅ | Does NOT need it ❌ |
|---|---|
| [[Linear Regression]] (for interpretation) | [[Decision Tree]] |
| [[Ridge]], [[Lasso]] | [[Random Forest]] |
| [[Logistic Regression]] | [[Gradient Boosting]] |
| [[SVM]], [[SVR]] | [[XGBoost]] |
| [[KNN]] | [[HistGradientBoosting]] |
| [[K-Means]], [[DBSCAN]] | |

Rule: models that use **distances or penalties** need scaling. Trees do not.

## Code (avoid leakage!)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit ONLY on train
X_test_s  = scaler.transform(X_test)       # only transform
```
Better: use a `Pipeline` → [[Sklearn Basics]].
