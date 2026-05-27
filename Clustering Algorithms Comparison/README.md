# Clustering Algorithms Comparison — K-Means, K-Medoid, K-Median & Hierarchical

## Overview

This project implements and compares four **unsupervised clustering algorithms** from scratch on a synthetic dataset. The implementations go beyond scikit-learn wrappers — each algorithm is built manually to expose the mathematical mechanics behind cluster formation, centroid updates, and distance metrics.

## Business Goals

- **Customer/Data Segmentation:** Partition unlabeled data into meaningful groups to discover hidden structure — directly applicable to market segmentation, anomaly detection, and data compression.
- **Algorithm Selection Guide:** Benchmark K-Means, K-Medoid, K-Median, and Hierarchical clustering under the same conditions to help practitioners choose the right algorithm based on data characteristics (outliers, distribution shape, scale).
- **Robustness Analysis:** Evaluate each method's sensitivity to initialization, outliers, and non-Euclidean distance measures.

## Dataset

| File | Description |
|------|-------------|
| `Data.csv` | 150 samples × 4 features: x1, x2, x3, y |

150 synthetic data points with 4 continuous features. The `y` column can be used as ground truth for cluster evaluation.

## Notebook

| Notebook | Description |
|----------|-------------|
| `clustering_algorithms.ipynb` | All four clustering algorithms with visual comparison |

## Algorithms Implemented

### 1. K-Means Clustering
- **Distance metric:** Euclidean distance
- **Centroids:** Mean of all points in each cluster
- **Iterations:** 10 with centroid re-computation at each step
- **Objective:** Minimize Sum of Squared Errors (SSE): `J = Σ ||xᵢ - μₖ||²`
- **Complexity:** O(nkt) — fast and scalable
- **Best for:** Spherical, equally-sized clusters with no significant outliers

### 2. K-Medoid Clustering (PAM — Partitioning Around Medoids)
- **Distance metric:** Manhattan distance (L1 norm)
- **Medoids:** Actual data points (not abstract centroids) — more interpretable
- **Initialization:** 100 random trials; best configuration selected by minimum total Manhattan cost
- **Key advantage over K-Means:** Robust to outliers since medoids must be real data points
- **Complexity:** O(n²k) — slower on large datasets
- **Best for:** Data with outliers, mixed-type features, or when cluster centers must be real observations

### 3. K-Median Clustering
- **Distance metric:** Manhattan distance (L1 norm)
- **Centroids:** Component-wise median of cluster members (not the mean)
- **Iterations:** 10 with median re-computation at each step
- **Key advantage:** Outlier-resistant — medians are robust to extreme values
- **Complexity:** O(nk) per iteration with sorting overhead
- **Best for:** Skewed distributions or data with heavy-tailed noise

### 4. Agglomerative Hierarchical Clustering (Single Linkage)
- **Approach:** Bottom-up — starts with 150 singleton clusters and merges greedily
- **Linkage:** Single linkage (minimum pairwise distance between clusters)
- **Dendrogram:** Generated using `scipy.linkage()` and `scipy.dendrogram()` for visualization
- **No need to pre-specify K** — cut the dendrogram at any height to get desired number of clusters
- **Best for:** Discovering hierarchical structure; exploratory analysis without a fixed K

## Algorithm Comparison

| Property | K-Means | K-Medoid | K-Median | Hierarchical |
|----------|---------|----------|----------|-------------|
| Distance | Euclidean | Manhattan | Manhattan | Euclidean |
| Center type | Abstract mean | Real data point | Component median | N/A (merge-based) |
| Outlier sensitivity | High | Low | Low | Depends on linkage |
| Requires K? | Yes | Yes | Yes | No |
| Complexity | O(nkt) | O(n²k) | O(nk) | O(n² log n) |
| Initialization effect | High | Mitigated (100 trials) | Moderate | None |

## Key Findings

- **K-Means** converges quickly and is most efficient, but centroid positions can be distorted by outliers.
- **K-Medoid** with 100 random initializations avoids local minima more reliably; medoids are interpretable as representative samples.
- **K-Median** provides outlier resistance without the O(n²) cost of K-Medoid; effective for skewed feature distributions.
- **Hierarchical clustering** reveals the natural grouping of the 150 data points in a dendrogram; single linkage tends to form elongated "chain" clusters.
- For the given dataset, the optimal K appears to be 3 based on the SSE elbow curve and dendrogram cut height.

## Tech Stack

- Python 3
- NumPy (distance computations, centroid updates)
- Pandas (data loading)
- Matplotlib (cluster visualization)
- SciPy (`linkage`, `dendrogram` for hierarchical clustering)

## How to Run

```bash
jupyter notebook clustering_algorithms.ipynb
```

Ensure `Data.csv` is in the same directory as the notebook.
