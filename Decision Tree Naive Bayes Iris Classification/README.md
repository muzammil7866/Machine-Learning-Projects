# Decision Trees, Random Forest & Naive Bayes — Iris Flower Classification

## Overview

This project implements four classical machine learning classification algorithms from scratch and applies them to the Iris flower dataset. The algorithms are built without relying on scikit-learn classifiers, demonstrating a deep understanding of each method's underlying mechanics.

## Business Goals

- **Multi-class Classification:** Automatically identify the species of an iris flower (Setosa, Versicolor, Virginica) from its physical measurements.
- **Algorithm Comparison:** Evaluate and compare the performance and interpretability of tree-based and probabilistic classifiers on the same dataset.
- **Educational Benchmark:** Establish a clear, reproducible baseline for classic ML algorithms on a well-known dataset.

## Dataset

| File | Description |
|------|-------------|
| `Iris.csv` | 150 samples × 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width (cm); 3 class labels |

**Classes:** `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` (50 samples each)

## Notebook

| Notebook | Description |
|----------|-------------|
| `decision_tree_naive_bayes.ipynb` | Full implementation of all four classifiers with analysis |

## Algorithms Implemented

### 1. ID3 Decision Tree (Information Gain)
- **Splitting criterion:** Shannon Entropy and Information Gain
- **Feature discretization:** Continuous features binned into 3 categories (low / medium / high)
- **Max depth:** 7
- **First split:** Petal Width at threshold 1.0 — perfectly isolates Iris-setosa
- Recursive tree construction with leaf nodes at pure classes

### 2. CART Decision Tree (Gini Impurity)
- **Splitting criterion:** Gini Impurity minimization
- Threshold-based binary splits (continuous features)
- Optimal split threshold search per feature at each node

### 3. Random Forest
- **Ensemble of 10 ID3 trees**, each trained on a bootstrap (random with replacement) sample
- **Prediction:** Majority vote across all trees
- **Feature importance** tracked by aggregating split frequencies across the forest
- Reduces overfitting compared to a single decision tree

### 4. Naive Bayes (Gaussian)
- **Assumption:** Features are conditionally independent given the class
- Models `P(feature | class)` as a Gaussian distribution (using per-class mean and std)
- **Posterior:** `P(class | features) ∝ P(features | class) × P(class)` via Bayes' theorem
- 80/20 train-test split with prior probabilities calculated from training set

## Key Findings

- **Petal Width** is the most discriminative feature — used as the first split in both ID3 and CART, as it cleanly separates Setosa from the other two classes.
- ID3 achieves high accuracy with depth-limited trees; deeper trees risk overfitting.
- Random Forest improves generalization over single trees by reducing variance through bagging.
- Gaussian Naive Bayes performs competitively despite the strong independence assumption, validating that the Iris features are weakly correlated.
- All algorithms handle the 3-class problem effectively; Setosa is always correctly classified while Versicolor vs. Virginica is the harder boundary.

## Results Summary

| Algorithm | Approach | Key Metric |
|-----------|----------|-----------|
| ID3 Decision Tree | Information Gain, depth=7 | First split on Petal Width (≤1.0) perfectly separates Setosa |
| CART Decision Tree | Gini Impurity, binary splits | Threshold-based feature splitting |
| Random Forest | 10 trees, bootstrap sampling | Majority vote; feature importance tracked |
| Naive Bayes (Gaussian) | Prior × Likelihood, Bayes theorem | Competitive accuracy with 80/20 split |

## Tech Stack

- Python 3
- NumPy (array operations, probability calculations)
- Pandas (data loading and preprocessing)
- Matplotlib (visualization of decision boundaries and feature distributions)

## How to Run

```bash
jupyter notebook decision_tree_naive_bayes.ipynb
```

Ensure `Iris.csv` is in the same directory as the notebook.
