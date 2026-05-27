# K-Means Clustering From Scratch

## Overview

Implements the **K-Means clustering algorithm from scratch** (without sklearn's KMeans) and validates it against a provided 2D dataset. Covers centroid initialisation, point-to-cluster assignment using Manhattan distance, centroid recomputation, and convergence detection.

## Concepts Covered

- Unsupervised learning and clustering
- K-Means algorithm — initialisation, assignment, update, convergence
- Manhattan distance vs. Euclidean distance
- Convergence criterion (centroid stability)
- Cluster visualisation with matplotlib
- Data normalisation using `StandardScaler`

## Files

| File | Description |
|------|-------------|
| `K-Means Clustering.ipynb` | Full K-Means implementation from scratch with visualisation |
| `cluster_validation_data.txt` | 2D dataset used for clustering (600 points, 3 natural clusters) |
| `Report.docx` | Problem statement and report |

## How to Run

```bash
jupyter notebook "K-Means Clustering.ipynb"
```

## Algorithm

```
1. Randomly initialise k centroids
2. Repeat until centroids stop changing:
   a. Assign each point to nearest centroid (Manhattan distance)
   b. Recompute centroids as mean of assigned points
3. Return cluster assignments
```

## Dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

## Results

The algorithm converges and correctly identifies the 3 natural clusters in the dataset:

```
Cluster 1: ~201 points
Cluster 2: ~214 points
Cluster 3: ~185 points
Converged at iteration 2
```