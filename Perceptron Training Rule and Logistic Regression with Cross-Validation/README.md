# Perceptron Training Rule and Logistic Regression with Cross-Validation

## Overview
This notebook combines the perceptron training rule with cross-validation on a breast-cancer dataset, then compares that setup against a logistic regression workflow using the same validation strategy.

## What The Notebook Covers
- Loading and preparing the cancer dataset.
- Train-test splitting with stratification.
- 5-fold KFold cross-validation.
- Fold-wise metrics for malignant and benign classes.
- Final test-set evaluation for both Perceptron and Logistic Regression.

## What The Output Shows
- Training, validation, and test metrics printed for accuracy, precision, recall, and F1 score.
- Separate metrics for malignant and benign classes.
- A structured comparison of how the classifier behaves across folds.

## Business Value
This approach matters when a business needs a reliable classifier and cannot trust performance from only one train-test split. It is especially useful for risk, screening, and decision-support problems.

## Key Takeaways
The notebook shows cross-validation discipline, class-specific evaluation, and a good understanding of how to compare linear classifiers in practice.