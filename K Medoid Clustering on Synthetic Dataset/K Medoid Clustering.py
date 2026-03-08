import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def manhattan_pairwise(a, b):
    """Return pairwise Manhattan distances between rows of a and b."""
    return np.sum(np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :]), axis=2)


def kmedoids(x, k, max_iters=300, random_state=42):
    """Basic K-Medoids clustering using Manhattan distance."""
    m = x.shape[0]
    if k <= 0 or k > m:
        raise ValueError("k must be between 1 and number of samples")

    rng = np.random.default_rng(random_state)
    medoid_indices = rng.choice(m, size=k, replace=False)
    labels = np.zeros(m, dtype=int)

    for iteration in range(1, max_iters + 1):
        distances_to_medoids = manhattan_pairwise(x, x[medoid_indices])
        labels = np.argmin(distances_to_medoids, axis=1)

        new_medoid_indices = medoid_indices.copy()
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]

            # Empty clusters are reassigned to a random non-medoid point.
            if cluster_indices.size == 0:
                candidate_pool = np.setdiff1d(np.arange(m), new_medoid_indices)
                if candidate_pool.size > 0:
                    new_medoid_indices[cluster_id] = rng.choice(candidate_pool)
                continue

            cluster_points = x[cluster_indices]
            cluster_distances = manhattan_pairwise(cluster_points, cluster_points)
            costs = np.sum(cluster_distances, axis=1)
            best_local_idx = np.argmin(costs)
            new_medoid_indices[cluster_id] = cluster_indices[best_local_idx]

        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        medoid_indices = new_medoid_indices

    final_distances = manhattan_pairwise(x, x[medoid_indices])
    total_cost = np.sum(np.min(final_distances, axis=1))
    return medoid_indices, x[medoid_indices], labels, iteration, total_cost


def plot_clusters(x, labels, medoids, k, save_dir):
    """Save a two-view cluster plot for quick interpretation."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10", k)

    feature_pairs = [
        (0, 1, "Age", "Annual Income (k$)"),
        (1, 2, "Annual Income (k$)", "Spending Score (1-100)"),
    ]

    for ax, (i, j, xlab, ylab) in zip(axes, feature_pairs):
        for cluster_id in range(k):
            cluster_points = x[labels == cluster_id]
            if cluster_points.size == 0:
                continue
            ax.scatter(
                cluster_points[:, i],
                cluster_points[:, j],
                s=35,
                alpha=0.8,
                color=cmap(cluster_id),
                label=f"Cluster {cluster_id + 1}",
            )

        ax.scatter(
            medoids[:, i],
            medoids[:, j],
            marker="X",
            s=220,
            c="black",
            edgecolors="white",
            linewidths=1,
            label="Medoids",
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"K-Medoids (K={k})")
        ax.grid(alpha=0.2)

    handles, labels_txt = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels_txt, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=min(k + 1, 5))
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = save_dir / f"kmedoids_k{k}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_summary(results, save_dir):
    """Plot overall clustering quality and cluster-size distribution across K."""
    save_dir.mkdir(parents=True, exist_ok=True)

    ks = [r["k"] for r in results]
    costs = [r["cost"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: objective trend across K values.
    axes[0].plot(ks, costs, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_xticks(ks)
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Total Manhattan Cost")
    axes[0].set_title("K-Medoids Cost vs K")
    axes[0].grid(alpha=0.25)

    # Right: stacked cluster-size comparison for each K.
    max_clusters = max(len(r["counts"]) for r in results)
    bottoms = np.zeros(len(ks))
    for cluster_idx in range(max_clusters):
        heights = [
            r["counts"][cluster_idx] if cluster_idx < len(r["counts"]) else 0
            for r in results
        ]
        axes[1].bar(ks, heights, bottom=bottoms, label=f"Cluster {cluster_idx + 1}")
        bottoms += np.array(heights)

    axes[1].set_xticks(ks)
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Points per K")
    axes[1].set_title("Cluster Size Composition")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    out_path = save_dir / "kmedoids_summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run K-Medoids clustering and plots.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show generated plots after saving them.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "customer.txt"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Add customer.txt in this folder."
        )

    data = pd.read_csv(dataset_path, delimiter="\t")
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    x = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values

    k_values = [2, 3, 4]
    output_dir = base_dir / "plots"
    results = []

    print("K-Medoids results")
    print("=" * 60)

    for k in k_values:
        medoid_indices, medoids, labels, iterations, cost = kmedoids(
            x, k, max_iters=300, random_state=42
        )

        counts = np.bincount(labels, minlength=k)
        plot_path = plot_clusters(x, labels, medoids, k, output_dir)

        print(f"K = {k}")
        print(f"Iterations: {iterations}")
        print(f"Total Manhattan Cost: {cost:.2f}")
        print(f"Cluster Sizes: {counts.tolist()}")
        print(f"Medoid indices: {medoid_indices.tolist()}")
        print("Medoid points [Age, Income, Spending]:")
        print(medoids)
        print(f"Plot saved: {plot_path}")
        print("-" * 60)

        results.append(
            {
                "k": k,
                "cost": float(cost),
                "counts": counts.tolist(),
            }
        )

    summary_path = plot_summary(results, output_dir)
    print(f"Summary plot saved: {summary_path}")

    if args.show:
        for k in k_values:
            img = plt.imread(output_dir / f"kmedoids_k{k}.png")
            plt.figure(figsize=(10, 4.5))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Saved visualization for K={k}")

        summary_img = plt.imread(summary_path)
        plt.figure(figsize=(10, 4.5))
        plt.imshow(summary_img)
        plt.axis("off")
        plt.title("Summary visualization")
        plt.show()


if __name__ == "__main__":
    main()

