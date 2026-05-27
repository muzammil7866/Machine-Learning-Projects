# Neural Network Activation Functions — Backpropagation Study

## Overview

This project implements a **3-layer neural network from scratch** and rigorously compares four activation functions — **Sigmoid**, **Tanh**, **ReLU**, and **Softmax** — by observing their effect on convergence behavior, error dynamics, and final classification accuracy during manual backpropagation training.

> **Related project:** [SVM And Neural Networks MNIST](../SVM%20And%20Neural%20Networks%20MNIST/) — applies SVM, ANN, and CNN to the MNIST handwritten digit recognition task using Keras/scikit-learn.

## Business Goal

**Inform Architecture Decisions:** Understanding which activation function works best under given conditions (small data, binary output, deep networks) is a foundational decision in neural network design. This study provides empirical evidence for choosing the right activation function — knowledge that directly impacts model convergence, training stability, and final accuracy in production systems.

## Dataset

No external data files — all data is embedded in the notebook.

- **Samples:** 15 synthetic binary instances
- **Features:** 5 binary input features per sample
- **Output:** 1 binary label per sample
- **Purpose:** Controlled environment to isolate and compare activation function behavior independent of dataset complexity

## Notebook

| Notebook | Description |
|----------|-------------|
| `activation_functions_comparison.ipynb` | Manual backpropagation with 4 activation functions; error tracking, convergence analysis, and accuracy comparison |

## Neural Network Architecture

```
Input Layer:  5 neurons  (binary features x1–x5)
Hidden Layer: 3 neurons  (activation: varies per experiment)
Output Layer: 1 neuron   (activation: varies per experiment)

Total weights: 15 (input → hidden) + 3 (hidden → output) = 18
Total biases:  2 (1 per layer)
```

**Training hyperparameters:**
- Learning rate: `α = 0.3`
- Weight update iterations: 1000
- Optimizer: Full-batch gradient descent with manual backpropagation

## Activation Functions Compared

### 1. Sigmoid (Hidden + Output)
- Formula: `σ(z) = 1 / (1 + e⁻ᶻ)` — output range (0, 1)
- Initial error range: 0.028 – 0.278
- Final error: ~0 (full convergence)
- **Final Accuracy: 100%** ✅

### 2. Tanh (Hidden + Output)
- Formula: `tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)` — output range (-1, 1)
- Initial error range: 0.009 – 0.340
- Final error: 0 or 0.5 (inconsistent convergence)
- **Final Accuracy: 66.67%**

### 3. ReLU (Hidden + Output)
- Formula: `ReLU(z) = max(0, z)` — output range [0, ∞)
- Initial error range: 0.0001 – 1.48
- Final error: Exploded to ~9.1×10⁷ (gradient explosion)
- **Final Accuracy: 40%** ❌

### 4. ReLU (Hidden) + Softmax (Output)
- Hidden: `ReLU(z) = max(0, z)` | Output: `softmax(z) = eᶻⱼ / Σeᶻₖ`
- Similar instability to pure ReLU
- **Final Accuracy: 46.67%** ❌

## Results Summary

| Activation | Error Behavior | Final Accuracy | Recommendation |
|-----------|----------------|----------------|----------------|
| Sigmoid | Smooth convergence to ~0 | **100%** | Best for small binary networks |
| Tanh | Partially converges | 66.67% | Better gradient flow than Sigmoid for deeper networks |
| ReLU | Gradient explosion | 40% | Needs weight init + clipping; unsuitable here |
| ReLU + Softmax | Gradient explosion | 46.67% | Same ReLU instability issue |

## Key Findings

- **Sigmoid dominates for this architecture:** On a small binary dataset with a shallow network, Sigmoid's bounded output (0, 1) provides natural gradient stability and achieves perfect classification.
- **ReLU gradient explosion is a known risk:** Without proper weight initialization (e.g., He initialization), batch normalization, or gradient clipping, ReLU's unbounded output causes catastrophic error explosion — visible in this experiment where error reached 9.1×10⁷.
- **Tanh's zero-centering helps partially:** Tanh converges better than ReLU but less consistently than Sigmoid, occasionally getting stuck near 0.5 error (random guessing boundary).
- **Context matters for activation choice:** ReLU is preferred in deep networks with proper initialization (ResNets, transformers), while Sigmoid/Tanh remain appropriate for shallow networks with small datasets.
- **Practical takeaway:** For hidden layers in deep networks → ReLU with He init; for output layers in binary classification → Sigmoid; for multi-class → Softmax.

## Activation Function Cheat Sheet

| Activation | Range | Gradient Issue | Best Used For |
|-----------|-------|----------------|---------------|
| Sigmoid | (0, 1) | Vanishing (deep) | Output (binary classification) |
| Tanh | (-1, 1) | Vanishing (less severe) | Hidden layers, RNNs |
| ReLU | [0, ∞) | Dying/Exploding (without init) | Hidden layers in deep networks |
| Softmax | (0, 1), sums to 1 | — | Output (multi-class classification) |

## Tech Stack

- Python 3
- NumPy (manual forward pass, backpropagation, weight updates)
- Matplotlib (error curves per activation function, accuracy comparison)

## How to Run

```bash
jupyter notebook activation_functions_comparison.ipynb
```

No external data files required — the dataset is defined directly within the notebook.
