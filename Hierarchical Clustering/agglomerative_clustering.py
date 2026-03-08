"""
Agglomerative Hierarchical Clustering Implementation
This script demonstrates hierarchical clustering using different linkage methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# Sample dataset
print("="*60)
print("AGGLOMERATIVE HIERARCHICAL CLUSTERING")
print("="*60)

data = {
    "Feature1": [2.5, 7.1, 3.0, 5.5, 8.2, 4.2],
    "Feature2": [3.6, 4.0, 6.5, 4.8, 5.2, 3.8],
    "Feature3": [1.8, 2.2, 5.8, 3.1, 2.9, 4.5]
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

# Compute linkage matrix using Ward's method (default)
print("Computing linkage matrix using Ward's method...")
Z_ward = linkage(X_scaled, method='ward')
print("Linkage matrix shape:", Z_ward.shape)
print("\n" + "="*60 + "\n")

# Create dendrogram with Ward's method
plt.figure(figsize=(14, 7))
dendrogram(Z_ward, labels=df.index.astype(str), leaf_font_size=10)
plt.title('Agglomerative Clustering Dendrogram (Ward\'s Method)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig('agglomerative_dendrogram_ward.png', dpi=100, bbox_inches='tight')
print("Saved: agglomerative_dendrogram_ward.png")
plt.show()

# Compare different linkage methods
print("\n" + "="*60)
print("COMPARING DIFFERENT LINKAGE METHODS")
print("="*60 + "\n")

linkage_methods = ['ward', 'complete', 'average', 'single']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    Z_method = linkage(X_scaled, method=method)
    dendrogram(Z_method, ax=axes[idx], labels=df.index.astype(str), leaf_font_size=10)
    axes[idx].set_title(f'Dendrogram ({method.capitalize()} Linkage)', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Sample Index', fontsize=10)
    axes[idx].set_ylabel('Distance', fontsize=10)

plt.tight_layout()
plt.savefig('agglomerative_dendrograms_comparison.png', dpi=100, bbox_inches='tight')
print("Saved: agglomerative_dendrograms_comparison.png")
plt.show()

# Extract clusters at different height thresholds
print("\n" + "="*60)
print("CLUSTER ASSIGNMENT AT DIFFERENT THRESHOLDS")
print("="*60 + "\n")

# Using Ward's linkage
clusters_2 = fcluster(Z_ward, t=2, criterion='maxclust')
clusters_3 = fcluster(Z_ward, t=3, criterion='maxclust')

print(f"Clusters with k=2:\n{clusters_2}")
print(f"\nClusters with k=3:\n{clusters_3}")

# Visualize clusters in 2D (using first two features)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot with 2 clusters
scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_2, cmap='viridis', s=200, alpha=0.7)
ax1.set_title('Agglomerative Clustering (k=2)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Feature 1 (normalized)', fontsize=10)
ax1.set_ylabel('Feature 2 (normalized)', fontsize=10)
plt.colorbar(scatter1, ax=ax1, label='Cluster')

# Plot with 3 clusters
scatter2 = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_3, cmap='viridis', s=200, alpha=0.7)
ax2.set_title('Agglomerative Clustering (k=3)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Feature 1 (normalized)', fontsize=10)
ax2.set_ylabel('Feature 2 (normalized)', fontsize=10)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

plt.tight_layout()
plt.savefig('agglomerative_clusters_2d.png', dpi=100, bbox_inches='tight')
print("\nSaved: agglomerative_clusters_2d.png")
plt.show()

print("\n" + "="*60)
print("AGGLOMERATIVE CLUSTERING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Number of samples: {len(df)}")
print(f"Number of features: {df.shape[1]}")
print(f"Linkage matrix dimensions: {Z_ward.shape}")
