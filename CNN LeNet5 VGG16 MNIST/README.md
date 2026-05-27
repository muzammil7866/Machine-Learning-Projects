# CNN Architectures: LeNet-5 & VGG-16 — MNIST Digit Recognition

## Overview

This project implements two landmark **Convolutional Neural Network (CNN)** architectures — **LeNet-5** (from scratch) and **VGG-16** (via transfer learning) — and benchmarks them on the MNIST handwritten digit recognition task. It demonstrates the dramatic accuracy improvement achieved by deep CNNs over simpler models and shows how pre-trained ImageNet weights can be leveraged for a completely different domain.

## Business Goals

- **High-Accuracy Digit Recognition:** Push MNIST accuracy to ~99% using deep CNN architectures — production-quality performance for OCR, cheque processing, and form digitization pipelines.
- **Architecture Comparison:** Evaluate a lightweight custom CNN (LeNet-5) against a heavyweight pre-trained model (VGG-16) to understand the accuracy/complexity tradeoff.
- **Transfer Learning Demonstration:** Show that weights pre-trained on ImageNet can be effectively transferred to grayscale digit recognition tasks, even with minimal fine-tuning.

## Dataset

MNIST — 70,000 handwritten digit images.

| File | Description |
|------|-------------|
| `train-images.idx3-ubyte` | 60,000 training images (28×28 grayscale) |
| `train-labels.idx1-ubyte` | 60,000 training labels (digits 0–9) |
| `t10k-images.idx3-ubyte` | 10,000 test images |
| `t10k-labels.idx1-ubyte` | 10,000 test labels |

**Preprocessing:**
- LeNet-5: Images normalized to [0, 1], reshaped to (28, 28, 1)
- VGG-16: Images resized to (32, 32), grayscale channel triplicated to (32, 32, 3) to match ImageNet input format

> **Note:** MNIST binary files can be downloaded from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) if not present locally.

## Notebook

| Notebook | Description |
|----------|-------------|
| `cnn_lenet5_vgg16_mnist.ipynb` | LeNet-5 implementation and VGG-16 transfer learning with full comparison |

## Model Architectures

### LeNet-5 (Custom Implementation)

Inspired by Yann LeCun's 1998 architecture — the original CNN for digit recognition.

```
Input (28×28×1)
  → Conv2D(6 filters, 5×5, ReLU)
  → MaxPool(2×2)
  → Conv2D(16 filters, 5×5, ReLU)
  → MaxPool(2×2)
  → Flatten
  → Dense(120, ReLU)
  → Dense(84, ReLU)
  → Output Dense(10, Softmax)
```

**Training:** 5 epochs, batch size 128

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|-------------|
| 1 | 90.80% | — |
| 2 | 97.56% | — |
| 3 | 98.18% | — |
| 4 | 98.59% | — |
| 5 | 98.80% | — |

**Final Test Accuracy: 98.40%**

### VGG-16 (Transfer Learning from ImageNet)

Using pre-trained VGG-16 weights from ImageNet; only the classification head is trained on MNIST.

```
VGG-16 Base (pre-trained, frozen/fine-tuned)
  → Flatten
  → Dense(256, ReLU)
  → Output Dense(10, Softmax)
```

**Preprocessing:** 28×28 grayscale → 32×32 RGB (channel duplication)  
**Training:** 5 epochs, batch size 128

**Final Test Accuracy: ~99%**

## Results Comparison

| Model | Parameters | Test Accuracy | Training Epochs |
|-------|-----------|---------------|-----------------|
| LeNet-5 | ~60K | **98.40%** | 5 |
| VGG-16 (Transfer) | ~138M | **~99%** | 5 |

## Key Findings

- **LeNet-5 achieves 98.4% in just 5 epochs** — a remarkable result for such a compact architecture (~60K parameters), confirming that spatial feature extraction via convolutions is far superior to fully-connected approaches for image data.
- **VGG-16 with transfer learning pushes to ~99%** — even though VGG-16 was trained on color ImageNet images, its learned edge and texture detectors transfer effectively to grayscale digit images.
- **Training progression is steep:** LeNet-5 jumps from 90.8% (epoch 1) to 98.8% (epoch 5), showing rapid feature learning in early epochs.
- **Transfer learning is dramatically efficient:** VGG-16 achieves ~99% accuracy in 5 epochs by leveraging pre-trained convolutional features, despite the domain mismatch (natural images vs. handwritten digits).
- Both architectures significantly outperform the ANN (89.12%) and SVM (88.33%) from Assignment 5, demonstrating the superiority of convolutional inductive biases for 2D spatial data.

## Architecture Insight

| Aspect | LeNet-5 | VGG-16 |
|--------|---------|--------|
| Depth | 5 layers | 16 layers |
| Parameters | ~60K | ~138M |
| Training data | MNIST only | ImageNet + MNIST fine-tune |
| Input size | 28×28 | 32×32 |
| Accuracy | 98.4% | ~99% |
| Practical use | Edge/embedded devices | Server-side, high-accuracy systems |

## Tech Stack

- Python 3
- TensorFlow / Keras (model construction and training)
- NumPy (data preprocessing)
- Matplotlib (training curves, sample predictions)

## How to Run

```bash
jupyter notebook cnn_lenet5_vgg16_mnist.ipynb
```

Ensure all four MNIST `.idx3-ubyte` / `.idx1-ubyte` files are in the same directory as the notebook.

> VGG-16 weights are downloaded automatically from Keras on first run (requires internet connection).
