---
tags: [concept]
---
# Bias-Variance Tradeoff

Total error = **Bias² + Variance + Noise**

- **Bias** = error from a model that is too simple (wrong assumptions) → [[Overfitting vs Underfitting|underfitting]].
- **Variance** = error from a model too sensitive to the training data → overfitting.
- **Noise** = error we cannot remove.

## The tradeoff
More complexity → bias ↓ but variance ↑.
We look for the **sweet spot** in the middle.

## Examples
| Model                  | Bias                     | Variance                              |
| ---------------------- | ------------------------ | ------------------------------------- |
| [[Linear Regression]]  | high                     | low                                   |
| Deep [[Decision Tree]] | low                      | high                                  |
| [[Random Forest]]      | low                      | lower (averaging trees cuts variance) |
| [[Gradient Boosting]]  | low (boosting cuts bias) | can be high if too many trees         |
| [[KNN]] with k=1       | low                      | high                                  |
| [[KNN]] with big k     | higher                   | low                                   |

## Tools to control it
- [[Regularization]]
- [[Hyperparameter Tuning]]
- Ensembles: bagging reduces variance, boosting reduces bias.
