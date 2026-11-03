---
tags: [cheatsheet]
---
# Model Comparison

| Model | Task | Scaling? | Key params | Overfit control |
|---|---|---|---|---|
| [[Linear Regression]] | Reg | helps | - | add regularization |
| [[Ridge]] | Reg | yes | `alpha` | alpha ↑ |
| [[Lasso]] | Reg | yes | `alpha` | alpha ↑ |
| [[SVR]] | Reg | yes | `C`, `epsilon`, `gamma` | C ↓, gamma ↓ |
| [[Logistic Regression]] | Class | yes | `C`, `penalty` | C ↓ |
| [[SVM]] | Class | yes | `C`, `kernel`, `gamma` | C ↓, gamma ↓ |
| [[KNN]] | Both | yes | `n_neighbors`, `weights` | k ↑ |
| [[Decision Tree]] | Both | no | `max_depth`, `min_samples_leaf` | depth ↓ |
| [[Random Forest]] | Both | no | `n_estimators`, `max_depth`, `max_features` | depth ↓, leaf ↑ |
| [[Gradient Boosting]] | Both | no | `n_estimators`, `learning_rate`, `max_depth` | lr ↓, depth ↓, subsample |
| [[XGBoost]] | Both | no | + `subsample`, `colsample_bytree`, `reg_*` | reg ↑, early stop |
| [[HistGradientBoosting]] | Both | no | `max_iter`, `learning_rate`, `max_leaf_nodes` | l2 ↑, early stop |
| [[K-Means]] | Cluster | yes | `n_clusters` | - |
| [[DBSCAN]] | Cluster | yes | `eps`, `min_samples` | - |

## Same syntax for all
```python
model = Model(params)
model.fit(X_train, y_train)     # clustering: fit(X)
model.predict(X_test)
```

## Bagging vs Boosting
| | Bagging ([[Random Forest]]) | Boosting ([[Gradient Boosting]]) |
|---|---|---|
| Trees | parallel, independent | sequential, fix errors |
| Reduces | variance | bias |
| Tree depth | deep | shallow |
| Risk | less overfit | can overfit |

Related: [[Which Model to Choose]]
