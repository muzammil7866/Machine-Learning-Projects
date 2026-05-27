# Logistic Regression SGD — Housing Property Classification

## Overview

This project is a focused, standalone implementation of **Logistic Regression** using **Stochastic Gradient Descent (SGD)** for binary classification. Applied to the same housing dataset used in the main regression assignment, it classifies properties as "costly" or "non-costly" based on three physical features.

> **Related project:** [Linear And Logistic Regression Housing Prices](../Linear%20And%20Logistic%20Regression%20Housing%20Prices/) — covers both linear regression (gradient descent + normal equation) and logistic regression in a single full assignment notebook.

## Business Goal

**Binary Property Classification:** Given a house's living area, number of bedrooms, and number of floors, determine whether the property falls into a "costly" or "non-costly" price tier — useful for automated investment screening and budget-based property filtering.

## Dataset

| File | Description |
|------|-------------|
| `DataX.dat` | Feature matrix — 50 housing samples × 3 features: Living Area (sq ft), Bedrooms, Floors |
| `DataY.dat` | Continuous house prices (used for context/normalization reference) |
| `ClassY.dat` | Binary classification target — Costly (1) vs. Non-Costly (0) |

## Notebook

| Notebook | Description |
|----------|-------------|
| `logistic_regression_sgd.ipynb` | Dedicated logistic regression classifier with SGD, sigmoid activation, and decision boundary analysis |

## Algorithm

### Logistic Regression with Stochastic Gradient Descent

**Model:**
- Hypothesis: `h(x) = σ(θᵀx)` where `σ(z) = 1 / (1 + e⁻ᶻ)`
- Output: Probability that a property is "costly" (`P(y=1 | x; θ)`)

**Training:**
- Optimizer: Stochastic Gradient Descent (SGD) — one sample per weight update
- Loss: Binary Cross-Entropy (Log Loss): `J = -[y log(h) + (1-y) log(1-h)]`
- Iterations: 100
- Learning rate: `α = 0.02`
- Data normalization applied before training (critical for convergence)

**Prediction:**
- Threshold: `P(y=1) ≥ 0.5` → classified as "Costly"

### Weight Update Rule
```
θⱼ := θⱼ + α * (yᵢ - h(xᵢ)) * xᵢⱼ    (for each sample i)
```

## Key Findings

- **Data normalization is essential** for SGD convergence in logistic regression — unnormalized features with very different scales cause the optimizer to oscillate or diverge.
- SGD converges faster per iteration than batch gradient descent but introduces more noise in the weight update trajectory.
- The sigmoid function maps any real-valued score to [0, 1], making it a natural probability estimator for binary classification.
- With 100 SGD iterations and `α = 0.02`, the classifier reaches a stable decision boundary that correctly separates the two property classes on the housing dataset.

## Comparison with Full Assignment

| Aspect | This Notebook | Main Assignment |
|--------|--------------|-----------------|
| Focus | Logistic regression only | Linear + Logistic regression |
| Optimizer | SGD (stochastic, 1 sample at a time) | Batch gradient descent |
| Scope | Classifier study | Full regression + classification pipeline |

## Tech Stack

- Python 3
- NumPy (sigmoid function, weight updates, normalization)
- Matplotlib (decision boundary, loss curve visualization)

## How to Run

```bash
jupyter notebook logistic_regression_sgd.ipynb
```

Ensure `DataX.dat`, `DataY.dat`, and `ClassY.dat` are in the same directory as the notebook.
