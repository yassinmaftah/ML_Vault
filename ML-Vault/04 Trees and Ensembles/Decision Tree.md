---
tags: [model, classification, regression, supervised, trees]
---
# Decision Tree

## 💡 Idea
A series of **if/else questions** on features. Each split tries to make groups as "pure" as possible.
- Classification: split by **Gini** or **entropy**; leaf = majority class.
- Regression: split by **MSE**; leaf = mean value.

## ✅ When to use
- You want an easy-to-explain model.
- Mixed data, no scaling needed.

## ⚙️ Params (these control complexity!)
| Param | Meaning | Tip |
|---|---|---|
| `max_depth` | max tree depth | **most important**. Smaller = simpler |
| `min_samples_split` | min samples to split a node | bigger = simpler |
| `min_samples_leaf` | min samples in a leaf | bigger = simpler |
| `max_features` | features tried per split | fewer = more random |
| `criterion` | `"gini"`, `"entropy"` (class); `"squared_error"` (reg) | rarely matters |
| `ccp_alpha` | pruning strength | bigger = smaller tree |

## 🔧 How to tune
A full tree **overfits** (memorizes). Limit it:
1. `max_depth` in [3, 5, 7, 10]
2. `min_samples_leaf` in [1, 5, 10, 20]
3. Use [[Cross-Validation]] with [[Hyperparameter Tuning|GridSearchCV]].

## 🧪 Syntax
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
```

## 🎯 Predict
```python
y_pred = model.predict(X_test)
model.feature_importances_
```
Show the tree: `from sklearn.tree import plot_tree; plot_tree(model)`

## 👍 / 👎
- 👍 easy to read, no scaling, handles non-linear data
- 👎 overfits very easily, unstable (small data change → new tree)

## 🔗 Links
[[Random Forest]] · [[Gradient Boosting]] · [[Overfitting vs Underfitting]] · [[Feature Scaling]]
