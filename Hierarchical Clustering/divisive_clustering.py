"""
Divisive Hierarchical Clustering Implementation
This script demonstrates divisive clustering (top-down approach).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("="*60)
print("DIVISIVE HIERARCHICAL CLUSTERING")
print("="*60)

# Sample dataset
data = {
    "Feature1": [2.5, 7.1, 3.0, 5.5, 8.2, 4.2, 6.5, 2.8],
    "Feature2": [3.6, 4.0, 6.5, 4.8, 5.2, 3.8, 4.5, 5.5],
    "Feature3": [1.8, 2.2, 5.8, 3.1, 2.9, 4.5, 3.2, 5.0]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("\nDataset:")
print(df)
print("\n" + "="*60 + "\n")

# Normalize the data
X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data normalized using StandardScaler")
print(f"Shape: {X_scaled.shape}")
print("\n" + "="*60 + "\n")

class DivisiveClustering:
    """
    Implement Divisive Hierarchical Clustering using recursive K-means splitting.
    
    This is a top-down approach that starts with all points in one cluster
    and recursively splits them into two clusters until each point is its own cluster.
    """
    
    def __init__(self, data, max_iterations=100):
        self.data = data
        self.max_iterations = max_iterations
        self.n_samples = len(data)
        self.linkage_matrix = []
        self.cluster_id = self.n_samples
        self.clusters = {0: list(range(self.n_samples))}
        self.distances = []
        
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def split_cluster(self, cluster_points):
        """
        Split a cluster into two sub-clusters using K-means with k=2.
        
        Args:
            cluster_points: Indices of points in the current cluster
            
        Returns:
            Tuple of (indices_cluster1, indices_cluster2, distance_between_clusters)
        """
        if len(cluster_points) <= 1:
            return None, None, float('inf')
        
        # Extract data for the current cluster
        cluster_data = self.data[cluster_points]
        
        # Use KMeans to split into 2 clusters
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(cluster_data)
            
            # Get the two clusters
            cluster1_idx = np.where(labels == 0)[0]
            cluster2_idx = np.where(labels == 1)[0]
            
            # Map back to original indices
            cluster1 = [cluster_points[i] for i in cluster1_idx]
            cluster2 = [cluster_points[i] for i in cluster2_idx]
            
            # Calculate distance between cluster centers
            center1 = kmeans.cluster_centers_[0]
            center2 = kmeans.cluster_centers_[1]
            distance = self.euclidean_distance(center1, center2)
            
            return cluster1, cluster2, distance
            
        except Exception as e:
            print(f"Error during clustering: {e}")
            return None, None, float('inf')
    
    def fit(self, num_clusters=3):
        """
        Perform divisive clustering until reaching desired number of clusters.
        
        Args:
            num_clusters: Target number of clusters
        """
        print(f"Starting Divisive Clustering to create {num_clusters} clusters...")
        print("="*60)
        
        step = 0
        active_clusters = {0: list(range(self.n_samples))}
        
        while len(active_clusters) < num_clusters:
            step += 1
            print(f"\nStep {step}: Current number of clusters: {len(active_clusters)}")
            
            # Find the cluster with maximum variance to split
            max_variance = -1
            cluster_to_split = None
            
            for cluster_id, indices in active_clusters.items():
                if len(indices) > 1:
                    cluster_data = self.data[indices]
                    variance = np.var(cluster_data)
                    if variance > max_variance:
                        max_variance = variance
                        cluster_to_split = cluster_id
            
            if cluster_to_split is None:
                print("No more clusters can be split!")
                break
            
            # Split the selected cluster
            cluster_points = active_clusters[cluster_to_split]
            cluster1, cluster2, distance = self.split_cluster(cluster_points)
            
            if cluster1 is None:
                print(f"Could not split cluster {cluster_to_split}")
                break
            
            print(f"  Splitting cluster {cluster_to_split}")
            print(f"  - Cluster size: {len(cluster1)} + {len(cluster2)}")
            print(f"  - Distance between centers: {distance:.4f}")
            
            # Store linkage information
            self.linkage_matrix.append([
                cluster_to_split if cluster_to_split < self.n_samples else cluster_to_split,
                self.cluster_id,
                distance,
                len(cluster1) + len(cluster2)
            ])
            
            # Remove the merged cluster and add the new ones
            del active_clusters[cluster_to_split]
            active_clusters[self.cluster_id] = cluster1
            self.cluster_id += 1
            active_clusters[self.cluster_id] = cluster2
            self.cluster_id += 1
        
        self.active_clusters = active_clusters
        
        print("\n" + "="*60)
        print(f"Divisive clustering completed!")
        print(f"Final number of clusters: {len(active_clusters)}")
        print("="*60)
        
        return active_clusters
    
    def get_cluster_labels(self):
        """Get cluster labels for all points."""
        labels = np.zeros(self.n_samples, dtype=int)
        for cluster_id, indices in self.active_clusters.items():
            for idx in indices:
                labels[idx] = cluster_id
        return labels

# Initialize and fit the divisive clustering
div_clustering = DivisiveClustering(X_scaled)
clusters = div_clustering.fit(num_clusters=3)

# Get cluster assignments
labels = div_clustering.get_cluster_labels()

print(f"\n" + "="*60)
print("CLUSTER ASSIGNMENTS")
print("="*60)

cluster_df = pd.DataFrame({
    'Index': range(len(labels)),
    'Cluster': labels
})
print(cluster_df)

# Visualize the clusters in 2D space (first two features)
plt.figure(figsize=(12, 5))

# Plot 1: Scatter plot of clusters
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.title('Divisive Hierarchical Clustering (k=3)', fontsize=12, fontweight='bold')
plt.xlabel('Feature 1 (normalized)', fontsize=11)
plt.ylabel('Feature 2 (normalized)', fontsize=11)
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, alpha=0.3)

# Plot 2: 3D visualization (first three features)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(1, 2, 2, projection='3d')
scatter_3d = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                        c=labels, cmap='viridis', s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.set_xlabel('Feature 1', fontsize=10)
ax.set_ylabel('Feature 2', fontsize=10)
ax.set_zlabel('Feature 3', fontsize=10)
ax.set_title('3D View of Divisive Clustering', fontsize=12, fontweight='bold')
plt.colorbar(scatter_3d, ax=ax, label='Cluster ID', shrink=0.5)

plt.tight_layout()
plt.savefig('divisive_clustering_visualization.png', dpi=100, bbox_inches='tight')
print("\nSaved: divisive_clustering_visualization.png")
plt.show()

# Statistics
print("\n" + "="*60)
print("CLUSTER STATISTICS")
print("="*60)

for cluster_id in sorted(labels):
    if cluster_id not in [None, np.nan]:
        cluster_indices = np.where(labels == cluster_id)[0]
        print(f"\nCluster {cluster_id}:")
        print(f"  - Number of points: {len(cluster_indices)}")
        print(f"  - Indices: {list(cluster_indices)}")
        cluster_data = X_scaled[cluster_indices]
        print(f"  - Mean: {np.mean(cluster_data, axis=0)}")
        print(f"  - Std: {np.std(cluster_data, axis=0)}")

print("\n" + "="*60)
print("DIVISIVE CLUSTERING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"Number of features: {df.shape[1]}")
print(f"Final clusters: {len(np.unique(labels))}")
