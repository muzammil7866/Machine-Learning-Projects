# Linear And Logistic Regression — Housing Price Prediction

## Overview

This project implements **Linear Regression** and **Logistic Regression** from scratch using gradient-based optimization and closed-form solutions, applied to a real estate dataset to predict house prices and classify properties by cost.

> **Related project:** [Logistic Regression SGD](../Logistic%20Regression%20SGD/) — a dedicated, standalone implementation of logistic regression using Stochastic Gradient Descent on the same housing dataset.

## Business Goals

- **Regression:** Predict the sale price of a house given its physical attributes (living area, bedrooms, floors), enabling data-driven property valuation.
- **Classification:** Classify whether a property is "costly" or "non-costly" using logistic regression, useful for investment screening and budget-based filtering.

## Dataset

| File | Description |
|------|-------------|
| `DataX.dat` | Feature matrix — 50 housing samples × 3 features: Living Area (sq ft), Bedrooms, Floors |
| `DataY.dat` | Continuous target — House prices (regression labels) |
| `ClassY.dat` | Binary target — Costly (1) vs. Non-Costly (0) (classification labels) |

50 synthetic housing samples with 3 input features.

## Notebook

| Notebook | Description |
|----------|-------------|
| `linear_logistic_regression.ipynb` | Full assignment: linear regression (gradient descent + normal equation) and logistic regression |

## Algorithms Implemented

### Part A — Linear Regression (Gradient Descent)
- Custom gradient descent with **Mean Squared Error (MSE)** loss
- Manual weight updates: `θ = θ - α * ∇J(θ)`
- Feature normalization applied before training
- Convergence tracked across iterations

### Part B — Linear Regression (Closed-Form / Normal Equation)
- Solved analytically using the matrix normal equation: `θ = (XᵀX)⁻¹ Xᵀy`
- No iterative optimization required
- Final parameters: `θ = [48828.94, 127.62, 53037.97, -63000.09]`

### Part C — Logistic Regression (SGD)
- Binary classification with sigmoid activation: `σ(z) = 1 / (1 + e⁻ᶻ)`
- Stochastic Gradient Descent with 100 iterations
- Learning rate: `α = 0.02`
- Decision threshold: `P(y=1) ≥ 0.5`

## Key Findings

- **Living Area** is the most significant predictor of house price — highest absolute weight in the linear model.
- Gradient descent converges successfully after normalization; without normalization, convergence is unstable.
- Closed-form solution produces identical parameters to converged gradient descent, validating both implementations.
- Logistic regression successfully separates costly vs. non-costly properties after data normalization.

## Results Summary

| Model | Approach | Outcome |
|-------|----------|---------|
| Linear Regression | Gradient Descent | MSE decreases across iterations; model converges |
| Linear Regression | Normal Equation | `θ = [48828.94, 127.62, 53037.97, -63000.09]` |
| Logistic Regression | SGD + Sigmoid | Binary classification with 0.5 decision threshold |

## Tech Stack

- Python 3
- NumPy (matrix operations, gradient computation)
- Matplotlib (training curves, decision boundary visualization)

## How to Run

```bash
jupyter notebook linear_logistic_regression.ipynb
```

Ensure `DataX.dat`, `DataY.dat`, and `ClassY.dat` are in the same directory as the notebook.
