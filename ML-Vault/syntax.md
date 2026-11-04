
---
tags: [cheatsheet, code]
---
# Code Cheat Sheet (all models)

Assumes `X` = features, `y` = target.
Links: [[00 Home]] · [[Model Comparison]]

---
## 1. Split → [[Train-Validation-Test Split]]
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# classification: add stratify=y
```

## 2. Scale (only if needed) → [[Feature Scaling]]
- Needed for: Linear, Ridge, Lasso, Logistic, SVM, SVR, KNN, K-Means, DBSCAN
- NOT needed for: Decision Tree, Random Forest, Gradient Boosting, XGBoost, HistGradientBoosting
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train ONLY
X_test  = scaler.transform(X_test)        # only transform
```

## 3. Train + Predict (same for all models)
```python
model = SomeModel(param=value)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 4. Evaluate

### Regression → [[Regression Metrics]]
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2  :", r2_score(y_test, y_pred))
```

### Classification → [[Classification Metrics]]
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))   # precision, recall, f1

y_proba = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, y_proba))
```

### Overfitting check → [[Overfitting vs Underfitting]]
```python
print("Train:", model.score(X_train, y_train))
print("Test :", model.score(X_test, y_test))   # much lower than train = overfit
```

### Cross-validation → [[Cross-Validation]]
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)   # scoring="r2" / "f1" ...
print(scores.mean(), scores.std())
```

---
# REGRESSION MODELS

## [[Linear Regression]]
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

## [[Ridge]]
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)          # alpha up = simpler
```

## [[Lasso]]
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)          # some coef_ become 0
```

## [[SVR]]
```python
from sklearn.svm import SVR
model = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
```

---
# CLASSIFICATION MODELS

## [[Logistic Regression]]
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000)   # C down = simpler
```

## [[SVM]]
```python
from sklearn.svm import SVC
model = SVC(kernel="rbf", C=1, gamma="scale", probability=True)
```

## [[KNN]]
```python
from sklearn.neighbors import KNeighborsClassifier   # or KNeighborsRegressor
model = KNeighborsClassifier(n_neighbors=5, weights="distance")
```

---
# TREES AND ENSEMBLES (classification AND regression)
For regression, replace `Classifier` with `Regressor`.

## [[Decision Tree]]
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
```

## [[Random Forest]]
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=10,
                               n_jobs=-1, random_state=42)
```

## [[Gradient Boosting]]
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                   max_depth=3, subsample=0.8, random_state=42)
```

## [[XGBoost]]   (`pip install xgboost`)
```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8,
                      early_stopping_rounds=50, eval_metric="logloss",
                      random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
# use a separate validation set (X_val, y_val), not the test set
```

## [[HistGradientBoosting]]
```python
from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1,
                                       max_leaf_nodes=31, early_stopping=True,
                                       random_state=42)
```

Feature importance (trees): `model.feature_importances_`

---
# UNSUPERVISED (no y, no split needed, scale first)

## [[K-Means]]
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(X_scaled)

model.predict(X_new)           # cluster of new points
model.cluster_centers_
model.inertia_                 # elbow method: plot for k = 1..10
silhouette_score(X_scaled, labels)
```

## [[DBSCAN]]
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)     # -1 = noise, NO predict() for new data
```

---
# TUNING → [[Hyperparameter Tuning]]
```python
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 5, 10], "n_estimators": [100, 300]}
grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)   # scoring="f1" etc.
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
best = grid.best_estimator_
y_pred = best.predict(X_test)
```

## Scale + model in one Pipeline (no leakage)
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(StandardScaler(), SVC())
grid = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, 1]}, cv=5)
grid.fit(X_train, y_train)          # scaling is done inside, per fold
grid.predict(X_test)
```
Pipeline param name = `lowercasemodelname__param`.

---
# FULL EXAMPLE (copy and run)
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print("CV:", cross_val_score(model, X_train, y_train, cv=5).mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```