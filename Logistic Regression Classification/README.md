# Logistic Regression Classification

## Overview
This notebook builds logistic regression classification from both scratch and with scikit-learn. It starts with a hand-coded sigmoid-based implementation and then validates the same style of binary classification using the built-in LogisticRegression model.

## What The Notebook Covers
- Manual prediction scoring with weighted features.
- Sigmoid probability conversion.
- Binary thresholding at 0.5.
- Cost calculation for logistic loss.
- A scikit-learn logistic regression baseline for comparison.

## What The Output Shows
- Training and test tables with predictions, probabilities, and residuals.
- Cost values for the custom implementation.
- Predicted labels and class probabilities from the built-in model.
- A final score call on held-out data.

## Business Value
This workflow is a good fit for binary decisions such as churn risk, approval, lead scoring, or any use case where the business wants a probability rather than only a yes/no answer.

## Key Takeaways
The notebook shows understanding of logistic regression math, probability calibration, and how to validate a custom implementation against a library model.