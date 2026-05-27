"""
Iris Species Classification
Compares Perceptron Learning Rule vs Gradient Descent Delta Rule from scratch.
Uses the Iris dataset with 80/20 train-test split across multiple activation functions.
Dataset: https://archive.ics.uci.edu/dataset/53/iris (same data as sklearn's built-in)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Activation Functions ──────────────────────────────────────────────────────

def step(x):        return (x >= 0.5).astype(float)
def step_d(x):      return np.ones_like(x)              # gradient not used in Perceptron

def sigmoid(x):     return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_d(x):   s = sigmoid(x); return s * (1 - s)

def tanh_fn(x):     return np.tanh(x)
def tanh_d(x):      return 1 - np.tanh(x) ** 2

def relu(x):        return np.maximum(0, x)
def relu_d(x):      return (x > 0).astype(float)

ACTIVATIONS = {
    "step":    (step,    step_d),
    "sigmoid": (sigmoid, sigmoid_d),
    "tanh":    (tanh_fn, tanh_d),
    "relu":    (relu,    relu_d),
}


# ── Perceptron (Learning Rule) ────────────────────────────────────────────────

class Perceptron:
    """
    Single-layer Perceptron using the Perceptron Learning Rule.
    Trained one sample at a time; updates only on misclassification.
    One-vs-Rest for multi-class (3 classifiers for 3 Iris species).
    """

    def __init__(self, n_classes=3, lr=0.1, epochs=100, activation="step"):
        self.n_classes  = n_classes
        self.lr         = lr
        self.epochs     = epochs
        self.activation = activation
        self.act_fn     = ACTIVATIONS[activation][0]
        self.W          = None    # (n_features, n_classes)
        self.b          = None    # (n_classes,)

        self.train_acc_history = []
        self.test_acc_history  = []

    def _forward(self, X):
        z = X @ self.W + self.b          # (n_samples, n_classes)
        return self.act_fn(z)

    def fit(self, X_train, y_train, X_test, y_test):
        n_features = X_train.shape[1]
        self.W = np.zeros((n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)

        # One-hot encode targets
        Y_train = np.eye(self.n_classes)[y_train]

        for _ in range(self.epochs):
            for xi, yi in zip(X_train, Y_train):
                output = self.act_fn(xi @ self.W + self.b)
                error  = yi - (output >= 0.5).astype(float)
                self.W += self.lr * np.outer(xi, error)
                self.b += self.lr * error

            self.train_acc_history.append(self.score(X_train, y_train))
            self.test_acc_history.append(self.score(X_test, y_test))

        return self

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Gradient Descent Delta Rule ───────────────────────────────────────────────

class DeltaRule:
    """
    Gradient Descent Delta Rule (Widrow-Hoff / Adaline variant).
    Updates weights using the gradient of the mean-squared error loss.
    Supports pluggable activation functions.
    One-vs-Rest for multi-class.
    """

    def __init__(self, n_classes=3, lr=0.01, epochs=100, activation="sigmoid"):
        self.n_classes  = n_classes
        self.lr         = lr
        self.epochs     = epochs
        self.activation = activation
        self.act_fn, self.act_d = ACTIVATIONS[activation]
        self.W = None
        self.b = None

        self.loss_history      = []
        self.train_acc_history = []
        self.test_acc_history  = []

    def _forward(self, X):
        z      = X @ self.W + self.b
        output = self.act_fn(z)
        return z, output

    def fit(self, X_train, y_train, X_test, y_test):
        n_features = X_train.shape[1]
        self.W = np.random.randn(n_features, self.n_classes) * 0.01
        self.b = np.zeros(self.n_classes)

        Y_train = np.eye(self.n_classes)[y_train]   # one-hot

        for _ in range(self.epochs):
            z, output = self._forward(X_train)
            error     = Y_train - output             # (n_samples, n_classes)
            delta     = error * self.act_d(z)        # chain rule

            self.W += self.lr * X_train.T @ delta / len(X_train)
            self.b += self.lr * delta.mean(axis=0)

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            self.train_acc_history.append(self.score(X_train, y_train))
            self.test_acc_history.append(self.score(X_test, y_test))

        return self

    def predict(self, X):
        _, output = self._forward(X)
        return np.argmax(output, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Experiment Runner ─────────────────────────────────────────────────────────

def run_experiment(X_train, y_train, X_test, y_test,
                   Model, activations, lrs, epochs=200):
    """
    Train the model for each (activation, lr) combination and return results.
    """
    results = []
    for act in activations:
        for lr in lrs:
            model = Model(n_classes=3, lr=lr, epochs=epochs, activation=act)
            t0 = time.perf_counter()
            model.fit(X_train, y_train, X_test, y_test)
            elapsed = (time.perf_counter() - t0) * 1000

            results.append({
                "model":      model.__class__.__name__,
                "activation": act,
                "lr":         lr,
                "train_acc":  model.score(X_train, y_train),
                "test_acc":   model.score(X_test, y_test),
                "time_ms":    elapsed,
                "history":    model,
            })
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_accuracy_curves(perceptron_results, delta_results):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Training Accuracy Curves — Perceptron vs Delta Rule",
                 fontsize=13, fontweight="bold")

    for row, (results, title) in enumerate([(perceptron_results, "Perceptron"),
                                             (delta_results, "Delta Rule")]):
        for col, res in enumerate(results[:4]):
            ax = axes[row][col]
            ax.plot(res["history"].train_acc_history, label="Train", color="#3498DB")
            ax.plot(res["history"].test_acc_history,  label="Test",  color="#E74C3C")
            ax.set_title(f"{title}\nact={res['activation']}, lr={res['lr']}", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=8)
            ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    plt.show()


def plot_comparison_bars(p_results, d_results):
    labels_p = [f"P-{r['activation']}\nlr={r['lr']}" for r in p_results]
    labels_d = [f"D-{r['activation']}\nlr={r['lr']}" for r in d_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test Accuracy Comparison — Perceptron (P) vs Delta Rule (D)",
                 fontsize=12, fontweight="bold")

    for ax, results, labels, color, title in [
        (axes[0], p_results, labels_p, "#3498DB", "Perceptron"),
        (axes[1], d_results, labels_d, "#E74C3C", "Delta Rule"),
    ]:
        accs = [r["test_acc"] * 100 for r in results]
        bars = ax.bar(range(len(accs)), accs, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(title)
        ax.axhline(y=100, color="green", linestyle="--", linewidth=0.8, label="100%")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 1,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_loss_curve(delta_results):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Delta Rule — MSE Loss Curves", fontsize=12, fontweight="bold")

    for ax, res in zip(axes, delta_results[:4]):
        ax.plot(res["history"].loss_history, color="#E74C3C", linewidth=1.5)
        ax.set_title(f"act={res['activation']}, lr={res['lr']}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    plt.show()


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary(p_results, d_results):
    print("\n" + "=" * 75)
    print("  RESULTS SUMMARY")
    print("=" * 75)
    print(f"  {'Model':<15} {'Activation':<10} {'LR':>6}  {'Train Acc':>10}  {'Test Acc':>10}  {'Time(ms)':>10}")
    print(f"  {'─'*73}")

    all_results = p_results + d_results
    for r in all_results:
        print(f"  {r['model']:<15} {r['activation']:<10} {r['lr']:>6.4f}  "
              f"{r['train_acc']*100:>9.1f}%  {r['test_acc']*100:>9.1f}%  {r['time_ms']:>10.2f}")

    best_p = max(p_results, key=lambda x: x["test_acc"])
    best_d = max(d_results, key=lambda x: x["test_acc"])

    print(f"\n  Best Perceptron : act={best_p['activation']}, lr={best_p['lr']} "
          f"→ test acc {best_p['test_acc']*100:.1f}%")
    print(f"  Best Delta Rule : act={best_d['activation']}, lr={best_d['lr']} "
          f"→ test acc {best_d['test_acc']*100:.1f}%")

    print(f"\n  KEY DIFFERENCES — Perceptron vs Delta Rule:")
    print(f"  {'Perceptron Learning Rule':<40} {'Gradient Descent Delta Rule'}")
    print(f"  {'─'*75}")
    diffs = [
        ("Updates on misclassification only",   "Updates every sample (gradient descent)"),
        ("Uses step / threshold activation",     "Works with any differentiable activation"),
        ("Guaranteed convergence (lin. sep.)",   "Minimises MSE; converges to global min"),
        ("No gradient computation needed",       "Requires activation derivative"),
        ("Binary output (0 or 1)",               "Continuous output (probability-like)"),
        ("Sensitive to non-separable data",      "Handles non-separable data better"),
    ]
    for p, d in diffs:
        print(f"  {p:<40} {d}")

    print(f"\n  ANSWERS TO REFLECTION QUESTIONS:")
    qs = [
        ("Q1 Key differences?",
         "Perceptron updates on error only; Delta Rule minimises MSE continuously."),
        ("Q2 Activation function impact?",
         "Step works for Perceptron; sigmoid/tanh smooth gradients → better Delta Rule."),
        ("Q3 Learning rate strategy?",
         "Start large (0.1), decay if loss oscillates; use 0.001-0.01 for Delta Rule."),
        ("Q4 80/20 split implications?",
         "More training data → better generalisation; 80/20 is the standard trade-off."),
        ("Q5 Challenges?",
         "Vanishing gradients with tanh; Perceptron fails on non-separable classes."),
        ("Q6 Strengths / limitations?",
         "Perceptron: fast, simple; limited to linear boundaries. "
         "Delta Rule: flexible; slower, needs tuning."),
    ]
    for q, a in qs:
        print(f"\n  {q}")
        print(f"    → {a}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  IRIS SPECIES CLASSIFICATION")
    print("  Perceptron Learning Rule vs Gradient Descent Delta Rule")
    print("=" * 65)

    # Load and split dataset (80 / 20)
    iris     = load_iris()
    X, y     = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n  Dataset  : Iris ({len(X)} samples, {X.shape[1]} features, 3 classes)")
    print(f"  Train    : {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print(f"  Classes  : {iris.target_names.tolist()}")

    # ── Perceptron — vary activations and learning rates ──────────────────────
    print("\n  Training Perceptron models …")
    p_results = run_experiment(
        X_train, y_train, X_test, y_test,
        Perceptron,
        activations=["step", "sigmoid", "tanh", "relu"],
        lrs=[0.1, 0.01, 0.001, 0.1],
        epochs=200,
    )

    # ── Delta Rule — vary activations and learning rates ──────────────────────
    print("  Training Delta Rule models …")
    d_results = run_experiment(
        X_train, y_train, X_test, y_test,
        DeltaRule,
        activations=["sigmoid", "tanh", "relu", "sigmoid"],
        lrs=[0.01, 0.001, 0.01, 0.001],
        epochs=200,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(p_results, d_results)

    # ── Visualisations ────────────────────────────────────────────────────────
    plot_accuracy_curves(p_results, d_results)
    plot_comparison_bars(p_results, d_results)
    plot_loss_curve(d_results)


if __name__ == "__main__":
    main()