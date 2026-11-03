---
tags: [concept]
---
# Overfitting vs Underfitting

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Model is | too simple | balanced | too complex |
| Train score | low | high | very high |
| Test score | low | high | low |
| Problem | high **bias** | ✅ | high **variance** |

**Overfitting** = the model memorizes the training data (even the noise).
**Underfitting** = the model is too simple to learn the pattern.

## How to detect
Compare train score and test (or CV) score.
- Both low → underfitting.
- Train high, test much lower → overfitting.
- Tool: learning curve, [[Cross-Validation]].

## How to fix
**Overfitting**
- Get more data
- Simplify the model (lower `max_depth`, fewer features)
- Add [[Regularization]] (`alpha` ↑, `C` ↓)
- Early stopping (boosting)
- Use bagging → [[Random Forest]]

**Underfitting**
- Use a more complex model
- Add features / polynomial features
- Reduce regularization
- Train longer

Related: [[Bias-Variance Tradeoff]], [[Hyperparameter Tuning]]
