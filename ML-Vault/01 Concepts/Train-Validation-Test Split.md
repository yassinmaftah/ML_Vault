---
tags: [concept]
---
# Train / Validation / Test Split

| Set            | Used for                    | Typical size |
| -------------- | --------------------------- | ------------ |
| **Train**      | fit the model               | 60-80%       |
| **Validation** | tune params, compare models | 10-20%       |
| **Test**       | final score, used **once**  | 10-20%       |

Why? A model tested on its own training data looks perfect but fails on new data.

## Code
```python
from sklearn.model_selection import train_test_split

# train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# classification with imbalanced classes → keep same ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# add validation: split train again
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)
```

## Tips
- `random_state` → same split every run.
- `stratify=y` for classification.
- Time series → do NOT shuffle, split by date.
- Instead of a fixed validation set, use [[Cross-Validation]].
- Fit scalers on train only → see [[Feature Scaling]].
