# SVM, ANN & CNN — MNIST Handwritten Digit Recognition

## Overview

This project explores **Support Vector Machines (SVM)** and **Deep Neural Networks (ANNs & CNNs)** for classifying handwritten digits from the MNIST dataset. It benchmarks classical kernel-based SVMs against modern deep learning architectures on the same image recognition task.

> **Related project:** [Neural Network Activation Functions](../Neural%20Network%20Activation%20Functions/) — a dedicated study of how Sigmoid, Tanh, ReLU, and Softmax activations affect training dynamics using custom backpropagation on synthetic data.

## Business Goals

- **Digit Recognition:** Build models that accurately classify handwritten digits (0–9) — core technology behind postal code reading, bank cheque processing, and form digitization.
- **SVM vs. Deep Learning:** Compare the performance of classical kernel-based SVMs against modern deep neural networks on the same image recognition task.

## Dataset

MNIST — the standard benchmark for handwritten digit recognition.

| File | Description |
|------|-------------|
| `train-images.idx3-ubyte` | 60,000 training images (28×28 grayscale pixels) |
| `train-labels.idx1-ubyte` | 60,000 training labels (digits 0–9) |
| `t10k-images.idx3-ubyte` | 10,000 test images |
| `t10k-labels.idx1-ubyte` | 10,000 test labels |

> **Note:** MNIST binary files can also be downloaded from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) if not present locally.

## Notebook

| Notebook | Description |
|----------|-------------|
| `svm_neural_networks_mnist.ipynb` | SVM classifiers (Linear + RBF), ANN, and CNN on MNIST |

## Algorithms & Models

### Question 1 — Support Vector Machines

| Model | Preprocessing | Test Accuracy |
|-------|--------------|---------------|
| Linear SVM (unscaled) | Raw pixel values | **86.94%** |
| Linear SVM (scaled) | StandardScaler normalization | **88.33%** |
| RBF SVM | Raw pixel values | 86.91% |

- **Best SVM:** Scaled Linear SVM (88.33%) — feature scaling significantly improves linear SVM performance
- **Insight:** RBF kernel did not outperform scaled linear SVM, suggesting the MNIST feature space is approximately linearly separable after normalization

### Question 2 — Deep Neural Networks

#### Simple ANN (Feedforward)
- **Architecture:** Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Output(10, Softmax)
- **Training:** 10 epochs, batch size 32
- **Test Accuracy: 89.12%**

#### CNN (Convolutional Neural Network)
- **Architecture:** Conv2D(32, 3×3) → Conv2D(64, 3×3) → MaxPool(2×2) → Dropout(0.25) → Dense(128) → Dropout(0.5) → Output(10, Softmax)
- **Training:** 10 epochs, batch size 32
- **Test Accuracy: 87.98%**

## Key Findings

- **Feature scaling is critical for SVMs:** StandardScaler normalization improved Linear SVM accuracy by ~1.4 percentage points on MNIST.
- **ANN outperforms CNN on this configuration:** The simple ANN (89.12%) slightly outperformed the CNN (87.98%) — likely due to insufficient CNN training epochs and the relatively small architecture chosen. CNNs generally dominate on image tasks with more training.
- **All models comfortably surpass chance (10%)** on the 10-class MNIST problem.

## Model Ranking

| Rank | Model | Accuracy |
|------|-------|---------|
| 1 | ANN (Dense) | 89.12% |
| 2 | Linear SVM (scaled) | 88.33% |
| 3 | CNN | 87.98% |
| 4 | RBF SVM | 86.91% |
| 5 | Linear SVM (unscaled) | 86.94% |

## Tech Stack

- Python 3
- NumPy (array operations, preprocessing)
- scikit-learn (`LinearSVC`, `StandardScaler`, `SVC`)
- TensorFlow / Keras (ANN and CNN)
- Matplotlib (loss curves, confusion matrices)

## How to Run

```bash
jupyter notebook svm_neural_networks_mnist.ipynb
```

Ensure all four MNIST `.idx3-ubyte` / `.idx1-ubyte` files are in the same directory as the notebook.
