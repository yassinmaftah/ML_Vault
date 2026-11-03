---
tags: [concept, metrics]
---
# Regression Metrics

y = true values, ŷ = predictions.

| Metric | Meaning | Good value | Note |
|---|---|---|---|
| **MAE** | mean of \|y − ŷ\| | low | easy to read, same unit as y |
| **MSE** | mean of (y − ŷ)² | low | punishes big errors |
| **RMSE** | √MSE | low | same unit as y |
| **R²** | % of variance explained | close to 1 | 0 = as bad as predicting the mean; can be negative |

## Code
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
```

## Which one?
- Outliers matter → RMSE/MSE.
- Outliers should be ignored → MAE.
- Compare models quickly → R².

Models: [[Linear Regression]], [[Ridge]], [[Lasso]], [[SVR]], [[Random Forest]], [[Gradient Boosting]]
See: [[Classification Metrics]], [[Cross-Validation]]
