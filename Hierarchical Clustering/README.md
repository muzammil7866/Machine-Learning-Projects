# Hierarchical Clustering Analysis

## Project Overview
This project demonstrates **hierarchical clustering** techniques using both **agglomerative (bottom-up)** and **divisive (top-down)** approaches on multidimensional datasets.

## Business Goals

### Primary Objectives
1. **Customer Segmentation**: Group customers based on behavioral and demographic features for targeted marketing campaigns
2. **Pattern Discovery**: Identify natural clusters in data without pre-specifying the number of clusters
3. **Data Organization**: Establish hierarchical relationships between data points for better insights

### Expected Business Outcomes
- **Improved Targeting**: Enable personalized marketing strategies for each customer segment
- **Cost Optimization**: Identify homogeneous groups to optimize resource allocation
- **Risk Assessment**: Segment customers by risk profiles for better credit/fraud management
- **Actionable Insights**: Create dendrograms to visualize hierarchical relationships at different granularities

---

## Files

### Main Scripts
- **`agglomerative_clustering.py`** - Bottom-up hierarchical clustering implementation
  - Uses Ward's linkage method for cluster merging
  - Supports multiple linkage methods (Ward, Complete, Average, Single)
  - Generates dendrograms for visualization
  
- **`divisive_clustering.py`** - Top-down hierarchical clustering implementation
  - Recursively splits clusters using K-Means
  - Starts with all points in one cluster
  - Splits clusters with maximum variance iteratively

---

## Algorithm Details

### Agglomerative Clustering (Bottom-Up)
**Process:**
1. Start with each point as its own cluster
2. Calculate distances between all clusters
3. Merge the two closest clusters
4. Repeat until only one cluster remains

**Linkage Methods:**
- **Ward's**: Minimizes within-cluster variance
- **Complete**: Maximum distance between clusters
- **Average**: Mean distance between clusters
- **Single**: Minimum distance between clusters

### Divisive Clustering (Top-Down)
**Process:**
1. Start with all points in one cluster
2. Find cluster with maximum variance
3. Split it into two sub-clusters using K-Means
4. Repeat until reaching desired number of clusters

**Advantages:**
- More interpretable than agglomerative
- Better for hierarchical decomposition
- Can be more efficient for large datasets

---

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Number of Samples | 6-8 samples |
| Features | 3 (Feature1, Feature2, Feature3) |
| Data Type | Continuous |
| Normalization | StandardScaler |

---

## How to Run

### Requirements
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Agglomerative Clustering
```bash
python agglomerative_clustering.py
```

**Output Files Generated:**
- `agglomerative_dendrogram_ward.png` - Dendrogram with Ward's linkage
- `agglomerative_dendrograms_comparison.png` - Comparison of all linkage methods
- `agglomerative_clusters_2d.png` - 2D scatter plots of clusters

### Divisive Clustering
```bash
python divisive_clustering.py
```

**Output Files Generated:**
- `divisive_clustering_visualization.png` - 2D and 3D visualizations of clusters

---

## Key Insights & Interpretation

### Dendrogram Reading
- **Height of merge**: Greater height = clusters are more dissimilar
- **Vertical distance**: Indicates how different clusters are when merged
- **Horizontal line placement**: Shows at what threshold clusters would merge

### Optimal Cluster Number
- Cut the dendrogram at an appropriate height
- Use **Elbow Method** or **Silhouette Score** to determine optimal k
- Business logic should guide cluster interpretation

### Feature Importance
- Analyze which features drove cluster formation
- Use domain knowledge to validate clusters

---

## Business Applications

### 1. **Customer Segmentation**
- Group customers by purchasing patterns
- Different marketing strategies per segment
- Personalized recommendations

### 2. **Product Categorization**
- Cluster similar products for cross-selling
- Identify product affinities
- Optimize product mix

### 3. **Network Analysis**
- Identify communities in social networks
- Detect organizational silos
- Improve collaboration structure

### 4. **Genomic Research**
- Classify genes with similar expression patterns
- Identify disease subtypes
- Discover biological relationships

---

## Performance Metrics

### Clustering Quality
- **Dendrogram Height**: Larger heights indicate better separation
- **Cophenetic Correlation**: How well dendrogram preserves distances
- **Silhouette Score**: Measure of cluster cohesion and separation

### Computational Complexity
- **Agglomerative**: O(n² log n) to O(n³)
- **Divisive**: O(n × k) depending on K-Means iterations

---

## Advantages & Limitations

### Advantages
✓ No need to pre-specify cluster count  
✓ Hierarchical structure provides rich information  
✓ Visual representation via dendrograms  
✓ Interpretable clustering process  

### Limitations
✗ Computationally expensive for large datasets  
✗ Cannot undo previous merges (agglomerative)  
✗ Sensitive to outliers  
✗ Dendrogram can be difficult to interpret with many points  

---

## Future Enhancements

1. **Implement Additional Linkage Methods**
   - Centroid linkage
   - Median linkage
   - Ward's variance reduction

2. **Add Dendrogram Cutting Strategies**
   - Automatic optimal k detection
   - Consistency-based cutting
   - Distance threshold-based cutting

3. **Performance Optimization**
   - Implement fast clustering with approximate algorithms
   - Use GPU acceleration for large datasets
   - Parallel processing of distance calculations

4. **Advanced Visualization**
   - Interactive dendrograms
   - 3D cluster visualization
   - Heatmaps of distance matrices

---

## References

- **Literature**: Hastie, Tibshirani, Friedman - "The Elements of Statistical Learning"
- **SciPy Documentation**: [Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- **Scikit-learn**: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)

---

## Author Notes

This project provides a solid foundation for understanding hierarchical clustering methods. The implementation combines educational value with practical applicability for real-world data analysis tasks.

**Last Updated:** 2024  
**Status:** ✓ Complete and Production-Ready
