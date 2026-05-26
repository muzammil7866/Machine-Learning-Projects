# Naive Bayes Classification

## Overview
This notebook compares a scratch implementation of categorical Naive Bayes against scikit-learn's CategoricalNB on the classic Play Tennis dataset.

## What The Notebook Covers
- Frequency-based probability estimation from categorical features.
- Prior probability calculation for the target class.
- Manual posterior probability comparison.
- A built-in Naive Bayes baseline after label encoding.

## What The Output Shows
- Printed probability tables for feature/value combinations.
- Predicted labels versus true labels.
- A comparison note showing that the scratch implementation achieved 75 percent accuracy while the built-in version achieved 50 percent on this tiny dataset.

## Business Value
Naive Bayes is useful when a team wants a fast, lightweight classifier for categorical decision support, especially when the training data is small and the deployment logic needs to stay simple.

## Key Takeaways
The notebook shows probability modeling, categorical feature handling, and practical comparison between a manual approach and a library implementation.