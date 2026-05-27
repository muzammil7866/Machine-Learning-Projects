# Perceptron — Supervised Learning

## Overview

Implements a **single-layer Perceptron** trained using the Perceptron Learning Rule to learn a Boolean OR function. The perceptron is the simplest artificial neural network — a binary linear classifier that updates its weights based on prediction errors.

## Concepts Covered

- Supervised learning paradigm
- Perceptron model (inputs, weights, threshold, activation)
- Perceptron Learning Rule (weight update on error)
- Training epochs and convergence
- Boolean function learning (OR gate)
- Linear separability

## Files

| File | Description |
|------|-------------|
| `Perceptron Learning Rule.ipynb` | Perceptron implementation and training trace |

## How to Run

```bash
jupyter notebook "Perceptron Learning Rule.ipynb"
```

## Perceptron Model

```
output = 1  if  (w · x) >= threshold
         0  otherwise

Weight update:
  w = w + learning_rate × (label − prediction) × input
```

## Training Data — OR Function

| Input x1 | Input x2 | Label (OR) |
|-----------|-----------|------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

## Result

After training for 100 epochs:

```
Trained Weights: [0.9, 0.9]

Input: [0, 0] → Predicted: 0  ✓
Input: [0, 1] → Predicted: 1  ✓
Input: [1, 0] → Predicted: 1  ✓
Input: [1, 1] → Predicted: 1  ✓
```