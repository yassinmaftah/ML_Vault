---
tags: [concept, metrics]
---
# Classification Metrics

## Confusion matrix
|  | Predicted + | Predicted − |
|---|---|---|
| **Actual +** | TP | FN |
| **Actual −** | FP | TN |

## Main metrics
| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP+TN) / all | % correct. Bad if classes are imbalanced |
| **Precision** | TP / (TP+FP) | when I say "positive", how often am I right? |
| **Recall** | TP / (TP+FN) | of all real positives, how many did I find? |
| **F1** | 2·P·R / (P+R) | balance of precision and recall |
| **ROC-AUC** | area under ROC curve | quality across all thresholds (0.5 = random, 1 = perfect) |

## When to use
- **Recall** matters: disease detection, fraud (missing a positive is costly).
- **Precision** matters: spam filter (false alarm is costly).
- **F1**: imbalanced data.

## Code
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report)

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))   # all at once

y_proba = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_proba)
```
Multi-class: add `average="macro"` or `"weighted"`.

## Threshold
Default threshold is 0.5. Lower it → recall ↑, precision ↓.

Models: [[Logistic Regression]], [[SVM]], [[KNN]], [[Random Forest]], [[XGBoost]]
See: [[Regression Metrics]], [[Cross-Validation]]
