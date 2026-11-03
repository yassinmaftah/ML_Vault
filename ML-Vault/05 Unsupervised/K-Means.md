---
tags: [model, clustering, unsupervised]
---
# K-Means

## 💡 Idea
Group data into **k clusters**:
1. Pick k random centers (centroids).
2. Assign each point to the nearest centroid.
3. Move each centroid to the mean of its points.
4. Repeat until nothing changes.

## ✅ When to use
- Round-shaped, similar-size clusters. Customer segmentation.

## ⚙️ Params
| Param | Meaning | Tip |
|---|---|---|
| `n_clusters` | k | **most important** |
| `init` | start method | `"k-means++"` (default, smart) |
| `n_init` | number of restarts | more = more stable |
| `max_iter` | max iterations | 300 |
| `random_state` | reproducibility | any int |

## 🔧 How to choose k
- **Elbow method**: plot `inertia_` for k=1..10, pick the "elbow".
- **Silhouette score**: higher is better (-1 to 1).
```python
from sklearn.metrics import silhouette_score
for k in range(2, 11):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    print(k, silhouette_score(X_scaled, labels))
```
- **Scale first** → [[Feature Scaling]].

## 🧪 Syntax
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, n_init=10, random_state=42)
model.fit(X_scaled)
```

## 🎯 Predict
```python
labels = model.labels_             # cluster of training points
new    = model.predict(X_new)      # cluster of new points
model.cluster_centers_             # centroids
model.inertia_                     # sum of squared distances (lower = tighter)
```

## 👍 / 👎
- 👍 simple, fast, scalable
- 👎 must choose k, only round clusters, sensitive to outliers and scale

## 🔗 Links
[[DBSCAN]] · [[KNN]] (supervised, different!) · [[Feature Scaling]]
