---
tags: [concept, code]
---
# Sklearn Basics

Every sklearn model uses the **same 3 steps**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SomeModel(param=value)   # 1. create
model.fit(X_train, y_train)      # 2. train
y_pred = model.predict(X_test)   # 3. predict
score = model.score(X_test, y_test)
```

- `score()` = **R²** for regression, **accuracy** for classification.
- Classifiers also have `predict_proba(X)` → probabilities.
- Unsupervised models: `fit(X)` then `predict(X)` or `fit_predict(X)`.

## Pipeline (scaling + model together, no leakage)
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), SomeModel())
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

## Useful attributes (after fit)
- `model.coef_`, `model.intercept_` → linear models
- `model.feature_importances_` → tree models
- `model.get_params()` → see all params

Links: [[ML Workflow]], [[Feature Scaling]], [[Hyperparameter Tuning]]
