"""
Random Forest and Decision Tree Classifier on Iris Dataset
This script builds and compares Decision Tree and Random Forest classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("IRIS DATASET CLASSIFICATION: DECISION TREE VS RANDOM FOREST")
print("="*80)

# Load the dataset
print("\nLoading Iris dataset...")
file_path = 'Iris.csv'
iris = pd.read_csv(file_path)
print(f"Dataset shape: {iris.shape}")
print("\nFirst few rows:")
print(iris.head())

# Preprocess the data
print("\nPreprocessing data...")

# Drop the 'Id' column if it exists
iris = iris.drop(columns=['Id'], errors='ignore')

# Map species to numerical values
species_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
iris['Species'] = iris['Species'].map(species_mapping)

# Reverse mapping for later use
species_reverse = {v: k for k, v in species_mapping.items()}

# Split the data into features and target
X = iris.drop(columns=['Species'])  # Features
y = iris['Species']  # Target

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nSpecies distribution:")
print(y.value_counts().sort_index())

# Split into training and test sets
print("\nSplitting data into train (70%) and test (30%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Standardize the data
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Standardization complete")

print("\n" + "="*80)
print("DECISION TREE CLASSIFIER")
print("="*80 + "\n")

# Build and train Decision Tree classifier
print("Training Decision Tree classifier...")
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_classifier.fit(X_train_scaled, y_train)
print("✓ Training complete")

# Predict and evaluate Decision Tree
y_pred_dt = dt_classifier.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
dt_recall = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)

print("\nDecision Tree Results:")
print("-" * 80)
print(f"Accuracy:  {dt_accuracy * 100:6.2f}%")
print(f"Precision: {dt_precision * 100:6.2f}%")
print(f"Recall:    {dt_recall * 100:6.2f}%")
print(f"F1-Score:  {dt_f1 * 100:6.2f}%")

print("\nConfusion Matrix:")
dt_cm = confusion_matrix(y_test, y_pred_dt)
print(dt_cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, 
                          target_names=[species_reverse[i] for i in range(3)]))

# Feature importance for Decision Tree
print("\nDecision Tree Feature Importance:")
print("-" * 80)
for feature, importance in zip(X.columns, dt_classifier.feature_importances_):
    print(f"{feature:20s}: {importance:.4f}")

print("\n" + "="*80)
print("RANDOM FOREST CLASSIFIER")
print("="*80 + "\n")

# Build and train Random Forest classifier
print("Training Random Forest classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train_scaled, y_train)
print("✓ Training complete")

# Predict and evaluate Random Forest
y_pred_rf = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

print("\nRandom Forest Results:")
print("-" * 80)
print(f"Accuracy:  {rf_accuracy * 100:6.2f}%")
print(f"Precision: {rf_precision * 100:6.2f}%")
print(f"Recall:    {rf_recall * 100:6.2f}%")
print(f"F1-Score:  {rf_f1 * 100:6.2f}%")

print("\nConfusion Matrix:")
rf_cm = confusion_matrix(y_test, y_pred_rf)
print(rf_cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, 
                          target_names=[species_reverse[i] for i in range(3)]))

# Feature importance for Random Forest
print("\nRandom Forest Feature Importance:")
print("-" * 80)
for feature, importance in zip(X.columns, rf_classifier.feature_importances_):
    print(f"{feature:20s}: {importance:.4f}")

# Comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80 + "\n")

print(f"{'Metric':<20} {'Decision Tree':<20} {'Random Forest':<20} {'Winner':<15}")
print("-" * 80)

metrics = {
    'Accuracy': (dt_accuracy, rf_accuracy),
    'Precision': (dt_precision, rf_precision),
    'Recall': (dt_recall, rf_recall),
    'F1-Score': (dt_f1, rf_f1)
}

for metric_name, (dt_val, rf_val) in metrics.items():
    winner = 'Random Forest' if rf_val > dt_val else 'Decision Tree' if dt_val > rf_val else 'Tie'
    print(f"{metric_name:<20} {dt_val*100:>6.2f}% {' '*10} {rf_val*100:>6.2f}% {' '*10} {winner}")

improvement = (rf_accuracy - dt_accuracy) * 100
print("-" * 80)
print(f"Random Forest performs better by: {improvement:+.2f}%")

# Visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80 + "\n")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Confusion Matrix - Decision Tree
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=[species_reverse[i] for i in range(3)],
            yticklabels=[species_reverse[i] for i in range(3)])
ax1.set_title('Decision Tree - Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=11)
ax1.set_xlabel('Predicted Label', fontsize=11)

# Plot 2: Confusion Matrix - Random Forest
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=[species_reverse[i] for i in range(3)],
            yticklabels=[species_reverse[i] for i in range(3)])
ax2.set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=11)
ax2.set_xlabel('Predicted Label', fontsize=11)

# Plot 3: Accuracy Comparison
ax3 = plt.subplot(2, 3, 3)
models = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy * 100, rf_accuracy * 100]
colors = ['#1f77b4', '#2ca02c']
bars = ax3.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy (%)', fontsize=11)
ax3.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Feature Importance - Decision Tree
ax4 = plt.subplot(2, 3, 4)
feature_names = X.columns
dt_importance = dt_classifier.feature_importances_
indices = np.argsort(dt_importance)
ax4.barh(range(len(indices)), dt_importance[indices], color='#1f77b4', alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(indices)))
ax4.set_yticklabels([feature_names[i] for i in indices])
ax4.set_xlabel('Importance', fontsize=11)
ax4.set_title('Decision Tree - Feature Importance', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Feature Importance - Random Forest
ax5 = plt.subplot(2, 3, 5)
rf_importance = rf_classifier.feature_importances_
indices = np.argsort(rf_importance)
ax5.barh(range(len(indices)), rf_importance[indices], color='#2ca02c', alpha=0.7, edgecolor='black')
ax5.set_yticks(range(len(indices)))
ax5.set_yticklabels([feature_names[i] for i in indices])
ax5.set_xlabel('Importance', fontsize=11)
ax5.set_title('Random Forest - Feature Importance', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Metrics Comparison
ax6 = plt.subplot(2, 3, 6)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_metrics = [dt_accuracy * 100, dt_precision * 100, dt_recall * 100, dt_f1 * 100]
rf_metrics = [rf_accuracy * 100, rf_precision * 100, rf_recall * 100, rf_f1 * 100]
x = np.arange(len(metrics_names))
width = 0.35
ax6.bar(x - width/2, dt_metrics, width, label='Decision Tree', alpha=0.8, edgecolor='black')
ax6.bar(x + width/2, rf_metrics, width, label='Random Forest', alpha=0.8, edgecolor='black')
ax6.set_ylabel('Score (%)', fontsize=11)
ax6.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_names, rotation=15, ha='right')
ax6.legend(fontsize=10)
ax6.set_ylim(0, 105)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('classifier_comparison_analysis.png', dpi=100, bbox_inches='tight')
print("Saved: classifier_comparison_analysis.png")
plt.show()

# Save models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80 + "\n")

import joblib
joblib.dump(dt_classifier, 'decision_tree_model.pkl')
print("✓ Saved: decision_tree_model.pkl")

joblib.dump(rf_classifier, 'random_forest_model.pkl')
print("✓ Saved: random_forest_model.pkl")

joblib.dump(scaler, 'feature_scaler.pkl')
print("✓ Saved: feature_scaler.pkl")

print("\n" + "="*80)
print("CLASSIFICATION COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nSummary:")
print(f"  - Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"  - Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"  - Better Model: Random Forest" if rf_accuracy > dt_accuracy else "  - Better Model: Decision Tree")
print(f"  - Performance Improvement: {improvement:+.2f}%")
print(f"  - Number of test samples: {len(y_test)}")
print(f"  - Number of features: {X.shape[1]}")
print(f"  - Number of classes: {len(np.unique(y))}")
