# Perceptron Vs Delta Rule — Iris Species Classifier

## Overview

Implements and compares two fundamental supervised learning algorithms **from scratch** on the classic Iris flower dataset (150 samples, 4 features, 3 species). Evaluates both algorithms across multiple activation functions and learning rates using an 80/20 train-test split.

## File

| File | Description |
|------|-------------|
| `Perceptron Vs Delta Rule Classifier.py` | Perceptron + Delta Rule from scratch, experiments, plots, reflection answers |

## How to Run

```bash
pip install numpy matplotlib scikit-learn
python "Perceptron Vs Delta Rule Classifier.py"
```

> The Iris dataset is loaded via `sklearn.datasets.load_iris()` — the exact same data as the UCI repository (https://archive.ics.uci.edu/dataset/53/iris).

## Dataset

| Property | Value |
|----------|-------|
| Samples | 150 (50 per class) |
| Features | Sepal length, sepal width, petal length, petal width |
| Classes | Iris-setosa (0), Iris-versicolor (1), Iris-virginica (2) |
| Split | 80% train (120) / 20% test (30), stratified |
| Preprocessing | StandardScaler (zero mean, unit variance) |

## Algorithm 1 — Perceptron Learning Rule

```
For each epoch:
    For each sample (xi, yi):
        output  = activation(xi · W + b)
        error   = one_hot(yi) − (output ≥ 0.5)
        W      += lr × xi ⊗ error
        b      += lr × error
```

- Updates weights only on misclassification
- One-vs-Rest for 3-class Iris
- Guaranteed to converge on linearly separable data
- Best activation: `step` or `sigmoid`

## Algorithm 2 — Gradient Descent Delta Rule

```
For each epoch (batch gradient):
    z       = X_train · W + b
    output  = activation(z)
    error   = Y_one_hot − output
    delta   = error × activation'(z)       ← chain rule
    W      += lr × X_train.T · delta / n
    b      += lr × mean(delta)
    loss    = MSE(error)
```

- Minimises Mean Squared Error via gradient descent
- Smooth updates every epoch (not just on misclassification)
- Works with any differentiable activation function
- Best activation: `sigmoid` or `tanh`

## Activation Functions Tested

| Function | Formula | Derivative | Notes |
|----------|---------|------------|-------|
| Step | `1 if x ≥ 0.5 else 0` | — | Perceptron standard |
| Sigmoid | `1/(1+e⁻ˣ)` | `s(1−s)` | Smooth, probabilistic |
| Tanh | `tanh(x)` | `1−tanh²(x)` | Zero-centred, −1 to +1 |
| ReLU | `max(0,x)` | `1 if x>0` | Sparse activation |

## Experiment Configuration

| Setting | Values tested |
|---------|--------------|
| Activations | step, sigmoid, tanh, relu |
| Learning rates | 0.1, 0.01, 0.001 |
| Epochs | 200 |
| Multi-class | One-vs-Rest (3 binary classifiers) |

## Key Differences

| Property | Perceptron Learning Rule | Gradient Descent Delta Rule |
|----------|-------------------------|----------------------------|
| Update trigger | Misclassification only | Every sample (gradient) |
| Loss function | None (threshold rule) | Mean Squared Error |
| Activation | Typically step | Any differentiable fn |
| Convergence guarantee | Yes (linearly separable) | Yes (convex loss) |
| Multi-class | One-vs-Rest | One-vs-Rest |
| Sensitivity | Non-separable data | Learning rate choice |

## Visualisations Produced

1. **Accuracy curves (8 subplots)** — train vs test accuracy per epoch for each configuration
2. **Test accuracy bar chart** — side-by-side comparison of all configurations
3. **MSE loss curves** — Delta Rule convergence for each activation function

## Dependencies

```
numpy, matplotlib, scikit-learn
```