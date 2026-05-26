# Hypothesis Testing

## Overview
This notebook is focused on building and evaluating a prediction hypothesis function from scratch. It works through manual prediction generation, error computation, and gradient-based updates for both a two-feature and a three-feature case.

## What The Notebook Covers
- Manual construction of a linear hypothesis function.
- Calculation of prediction error using MAE and MSE.
- Gradient-descent style parameter updates.
- Plotting predicted values against the target values.
- A second example that extends the same logic to three input features.

## What The Output Shows
- Tabular comparison of predictions and residuals.
- Parameter updates across iterations.
- Scatter and line plots showing how the predicted curve fits the data.

## Business Value
This project is useful wherever a team needs to understand how a numeric model is behaving before moving to production. It shows how to evaluate model fit, compare error metrics, and iterate toward a better approximation of real business data.

## Key Takeaways
The notebook demonstrates manual machine-learning reasoning, error analysis, and optimization basics without hiding behind a black-box library.