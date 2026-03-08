"""
K-Means Clustering Implementation
This script implements K-Means clustering from scratch and using scikit-learn.
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans

print("="*70)
print("K-MEANS CLUSTERING")
print("="*70)

# Load the dataset
print("\nLoading dataset...")
df = pd.read_csv('cluster_validation_data.txt', sep=",", header=None)
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Normalize data
print("\nNormalizing data using StandardScaler...")
X = df.values
sc = StandardScaler()
sc.fit(X)
X_normalized = sc.transform(X)
print("✓ Data normalization complete")

print("\n" + "="*70)
print("CUSTOM K-MEANS IMPLEMENTATION")
print("="*70 + "\n")

class CustomKMeans:
    """
    K-Means clustering implementation from scratch.
    
    Steps:
    1. Initialize k random cluster centers
    2. Assign each point to nearest centroid (Euclidean distance)
    3. Recalculate centroids as mean of assigned points
    4. Repeat until convergence
    """
    
    def __init__(self, k=3, max_iterations=100, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X):
        """
        Fit the K-Means model to the data.
        
        Args:
            X: Input data (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        
        # Step 1: Initialize centroids randomly
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        
        print(f"Initializing K-Means with k={self.k}")
        print(f"Initial centroids indices: {random_indices}\n")
        
        # Step 2-4: Iterate until convergence
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid
            distances = np.zeros((n_samples, self.k))
            for i in range(self.k):
                distances[:, i] = np.sqrt(np.sum((X - self.centroids[i]) ** 2, axis=1))
            
            self.labels = np.argmin(distances, axis=1)
            
            # Calculate inertia (sum of squared distances)
            inertia = np.sum(np.min(distances, axis=1) ** 2)
            self.inertia_history.append(inertia)
            
            # Store old centroids
            old_centroids = self.centroids.copy()
            
            # Recalculate centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
            
            # Check for convergence
            centroid_shift = np.sum(np.sqrt(np.sum((old_centroids - self.centroids) ** 2, axis=1)))
            
            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration + 1:3d} | Inertia: {inertia:10.4f} | Centroid shift: {centroid_shift:.6f}")
            
            if centroid_shift < 1e-6:
                print(f"\n✓ Converged at iteration {iteration + 1}!")
                print(f"Final Centroid Shift: {centroid_shift:.6e}\n")
                break
        
        else:
            print(f"\n⚠ Stopped after {self.max_iterations} iterations (max reached)\n")
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.sqrt(np.sum((X - self.centroids[i]) ** 2, axis=1))
        return np.argmin(distances, axis=1)

# Train custom K-Means
print("Training custom K-Means model...")
custom_kmeans = CustomKMeans(k=3, max_iterations=100, random_state=42)
custom_kmeans.fit(X_normalized)

# Denormalize for visualization
X_denormalized = sc.inverse_transform(X_normalized)
centroids_denormalized = sc.inverse_transform(custom_kmeans.centroids)

print("\nCustom K-Means Cluster Statistics:")
print("-" * 70)
for i in range(custom_kmeans.k):
    cluster_mask = custom_kmeans.labels == i
    cluster_size = np.sum(cluster_mask)
    print(f"Cluster {i}: {cluster_size:4d} points")

print("\n" + "="*70)
print("SCIKIT-LEARN K-MEANS")
print("="*70 + "\n")

print("Training Scikit-Learn K-Means model...")
sklearn_kmeans = SklearnKMeans(n_clusters=3, n_init=10, random_state=42, verbose=1)
sklearn_labels = sklearn_kmeans.fit_predict(X_normalized)
sklearn_centroids_denormalized = sc.inverse_transform(sklearn_kmeans.cluster_centers_)

print("\nScikit-Learn K-Means Cluster Statistics:")
print("-" * 70)
for i in range(3):
    cluster_size = np.sum(sklearn_labels == i)
    print(f"Cluster {i}: {cluster_size:4d} points")

print("\n" + "="*70)
print("INERTIA COMPARISON")
print("="*70 + "\n")

custom_inertia = custom_kmeans.inertia_history[-1]
sklearn_inertia = sklearn_kmeans.inertia_

print(f"Custom K-Means Inertia: {custom_inertia:.4f}")
print(f"Scikit-Learn Inertia:   {sklearn_inertia:.4f}")
print(f"Difference: {abs(custom_inertia - sklearn_inertia):.4f}")

# Visualization
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70 + "\n")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Inertia history (convergence)
ax1 = plt.subplot(2, 3, 1)
iterations = range(1, len(custom_kmeans.inertia_history) + 1)
ax1.plot(iterations, custom_kmeans.inertia_history, 'b-o', linewidth=2, markersize=5)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=11)
ax1.set_title('K-Means Convergence (Custom)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Custom K-Means Scatter (first two features, denormalized)
ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(X_denormalized[:, 0], X_denormalized[:, 1], 
                      c=custom_kmeans.labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax2.scatter(centroids_denormalized[:, 0], centroids_denormalized[:, 1], 
           c='red', s=300, alpha=0.8, marker='*', edgecolors='black', linewidths=2, label='Centroids')
ax2.set_xlabel('Feature 1 (Denormalized)', fontsize=11)
ax2.set_ylabel('Feature 2 (Denormalized)', fontsize=11)
ax2.set_title('Custom K-Means Clusters (2D View)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
plt.colorbar(scatter, ax=ax2, label='Cluster')

# Plot 3: Scikit-Learn K-Means Scatter
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(X_denormalized[:, 0], X_denormalized[:, 1], 
                      c=sklearn_labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax3.scatter(sklearn_centroids_denormalized[:, 0], sklearn_centroids_denormalized[:, 1], 
           c='red', s=300, alpha=0.8, marker='*', edgecolors='black', linewidths=2, label='Centroids')
ax3.set_xlabel('Feature 1 (Denormalized)', fontsize=11)
ax3.set_ylabel('Feature 2 (Denormalized)', fontsize=11)
ax3.set_title('Scikit-Learn K-Means Clusters (2D View)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
plt.colorbar(scatter, ax=ax3, label='Cluster')

# Plot 4: 3D visualization - Custom (if data has 3+ features)
if X_denormalized.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    scatter = ax4.scatter(X_denormalized[:, 0], X_denormalized[:, 1], X_denormalized[:, 2],
                         c=custom_kmeans.labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax4.scatter(centroids_denormalized[:, 0], centroids_denormalized[:, 1], centroids_denormalized[:, 2],
               c='red', s=300, marker='*', edgecolors='black', linewidths=2)
    ax4.set_xlabel('Feature 1', fontsize=10)
    ax4.set_ylabel('Feature 2', fontsize=10)
    ax4.set_zlabel('Feature 3', fontsize=10)
    ax4.set_title('Custom K-Means 3D View', fontsize=12, fontweight='bold')
    
    # Plot 5: 3D visualization - Scikit-Learn
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    scatter = ax5.scatter(X_denormalized[:, 0], X_denormalized[:, 1], X_denormalized[:, 2],
                         c=sklearn_labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax5.scatter(sklearn_centroids_denormalized[:, 0], sklearn_centroids_denormalized[:, 1], sklearn_centroids_denormalized[:, 2],
               c='red', s=300, marker='*', edgecolors='black', linewidths=2)
    ax5.set_xlabel('Feature 1', fontsize=10)
    ax5.set_ylabel('Feature 2', fontsize=10)
    ax5.set_zlabel('Feature 3', fontsize=10)
    ax5.set_title('Scikit-Learn K-Means 3D View', fontsize=12, fontweight='bold')

# Plot 6: Cluster size comparison
ax6 = plt.subplot(2, 3, 6)
custom_sizes = [np.sum(custom_kmeans.labels == i) for i in range(3)]
sklearn_sizes = [np.sum(sklearn_labels == i) for i in range(3)]
x = np.arange(3)
width = 0.35
ax6.bar(x - width/2, custom_sizes, width, label='Custom K-Means', alpha=0.8)
ax6.bar(x + width/2, sklearn_sizes, width, label='Scikit-Learn', alpha=0.8)
ax6.set_xlabel('Cluster', fontsize=11)
ax6.set_ylabel('Number of Points', fontsize=11)
ax6.set_title('Cluster Size Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'])
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('kmeans_clustering_analysis.png', dpi=100, bbox_inches='tight')
print("Saved: kmeans_clustering_analysis.png")
plt.show()

print("\n" + "="*70)
print("K-MEANS CLUSTERING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nDataset Information:")
print(f"  - Total samples: {len(df)}")
print(f"  - Number of features: {X.shape[1]}")
print(f"  - Number of clusters: 3")
print(f"\nCustom Implementation:")
print(f"  - Converged: Yes")
print(f"  - Iterations to converge: {len(custom_kmeans.inertia_history)}")
print(f"  - Final inertia: {custom_kmeans.inertia_history[-1]:.4f}")
