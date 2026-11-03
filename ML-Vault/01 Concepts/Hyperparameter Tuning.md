---
tags: [concept, code]
---
# Hyperparameter Tuning

- **Parameter** = learned by the model (weights, split points).
- **Hyperparameter** = set by you before training (`max_depth`, `alpha`, `C`...).

## Methods
| Method | How | When |
|---|---|---|
| **GridSearchCV** | tries all combinations | few params |
| **RandomizedSearchCV** | tries random combinations | many params (faster) |
| Bayesian (Optuna) | smart search | big models like [[XGBoost]] |

## Code
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 10],
}
grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

grid.best_params_     # best combination
grid.best_score_      # best CV score
best_model = grid.best_estimator_
best_model.predict(X_test)
```

```python
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, random_state=42)
```

## With a pipeline
Use `step__param` names:
```python
pipe = make_pipeline(StandardScaler(), SVC())
grid = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10]}, cv=5)
```

## Tips
- Tune the **most important params first** (each model note lists them).
- Use a log scale for `C`, `alpha`, `learning_rate`: `[0.001, 0.01, 0.1, 1, 10]`.
- Never tune on the test set → [[Train-Validation-Test Split]].

Uses: [[Cross-Validation]]. Goal: fix [[Overfitting vs Underfitting]].
