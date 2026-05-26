# K-Means and K-Medoids Clustering

## Overview
This notebook compares K-Means and K-Medoids on the same normalized dataset to show how two clustering strategies behave on the same points under different distance rules and update logic.

## What The Notebook Covers
- Min-max normalization before clustering.
- K-Means clustering with Euclidean distance.
- Elbow-method style SSE tracking for K values from 2 to 10.
- K-Medoids clustering with Manhattan distance.
- Side-by-side reasoning about convergence and computational cost.

## What The Output Shows
- Cluster assignments printed for each value of K.
- SSE values that generally decrease as K grows.
- A plotted elbow curve for K-Means.
- A written comparison noting that K-Means converged faster and produced lower error in this notebook.

## Business Value
This type of workflow is useful for customer segmentation, market grouping, product clustering, and any task where unlabeled data needs to be turned into actionable segments.

## Key Takeaways
The notebook shows comfort with unsupervised learning, distance-based algorithms, normalization, and practical model selection.