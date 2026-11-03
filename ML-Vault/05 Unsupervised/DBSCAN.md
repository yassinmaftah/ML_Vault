---
tags: [model, clustering, unsupervised]
---
# DBSCAN

## 💡 Idea
Clusters are **dense regions** separated by sparse regions.
- **Core point**: has at least `min_samples` neighbors within distance `eps`.
- Clusters grow by linking core points.
- Points in no cluster = **noise** (label `-1`).

## ✅ When to use
- Clusters of any shape, you do not know k, you want outlier detection.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `eps` | neighborhood radius | **most important**. Too small = all noise, too big = one cluster |
| `min_samples` | points needed for a dense area | default 5; rule: ≥ n_features + 1 (or 2·n_features) |
| `metric` | distance | `"euclidean"` |

## 🔧 How to tune
- **k-distance plot**: sort the distance to the k-th neighbor (k = `min_samples`), pick `eps` at the "elbow".
```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
d, _ = NearestNeighbors(n_neighbors=5).fit(X_scaled).kneighbors(X_scaled)
dist = np.sort(d[:, -1])      # plot this, find the elbow
```
- **Scale first** → [[Feature Scaling]].
- Check results with `silhouette_score` (ignore noise points).

## 🧪 Syntax
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)
```

## 🎯 Predict
⚠️ DBSCAN has **no `predict`** for new data. Only `fit_predict` on the same data.
```python
labels            # -1 = noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

## 👍 / 👎
- 👍 no k needed, any shape, finds outliers
- 👎 hard to choose `eps`, bad when density varies, no predict

## 🔗 Links
[[K-Means]] · [[Feature Scaling]]
