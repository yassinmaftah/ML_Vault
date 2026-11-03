---
tags: [concept]
---
# Cross-Validation (CV)

Split the train data in **k folds**. Train on k-1 folds, validate on the remaining one. Repeat k times. Average the scores.

```
Fold1 [VAL][ T ][ T ][ T ][ T ]
Fold2 [ T ][VAL][ T ][ T ][ T ]
...   (k=5 → 5 scores → mean)
```

**Why?** A more stable estimate than one single validation split, and all data is used.

## Code
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

## Variants
| Name | Use |
|---|---|
| `KFold` | default for regression |
| `StratifiedKFold` | classification (keeps class ratio). Default for classifiers |
| `TimeSeriesSplit` | time data |
| `LeaveOneOut` | very small data (slow) |

## Scoring names
- Regression: `"r2"`, `"neg_mean_squared_error"`, `"neg_mean_absolute_error"`
- Classification: `"accuracy"`, `"f1"`, `"roc_auc"`, `"recall"`

(`neg_` = sklearn wants "higher is better", so errors are negative.)

Used inside → [[Hyperparameter Tuning]]
See also: [[Train-Validation-Test Split]], [[Overfitting vs Underfitting]]
