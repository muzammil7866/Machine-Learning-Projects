# Bayesian Networks & Probabilistic Inference

## Overview

This project applies **Bayes' Theorem**, **Bayesian Networks**, and **Naive Bayes classifiers** to solve real-world probabilistic inference problems. It covers three distinct problem domains: medical diagnosis, email spam classification, and car value prediction — demonstrating how probabilistic reasoning can be formalized and computed from conditional probability tables (CPTs).

## Business Goals

- **Medical Diagnosis:** Compute the posterior probability of disease given a positive diagnostic test, accounting for disease prevalence and test accuracy — directly applicable to clinical decision-support systems.
- **Spam Detection:** Classify emails as Spam, Low Priority, or High Priority using keyword features and Bayesian classification, foundational to modern spam filters.
- **Car Valuation:** Build a Bayesian Network that models dependencies between car attributes (mileage, engine condition, air conditioning) and estimate the probability distribution over car values.
- **General Classifier:** Implement a multi-hypothesis Naive Bayes classifier for arbitrary binary feature vectors.

## Notebook

| Notebook | Description |
|----------|-------------|
| `bayesian_networks.ipynb` | All four problems with mathematical derivations and Python implementations |

## Problems Solved

### Question 1A — Medical Diagnosis (Bayes' Theorem)
- **Problem:** Given disease prevalence, test sensitivity, and specificity — what is the probability a patient actually has the disease given a positive test?
- **Approach:** Applied Bayes' Theorem:  
  `P(Disease | Test+) = P(Test+ | Disease) × P(Disease) / P(Test+)`
- **Result:** `P(Disease | Positive) ≈ 0.000917` — highlights the base-rate fallacy: even with a sensitive test, a rare disease yields mostly false positives.

### Question 1B — Email Spam Classification
- **Problem:** Classify an email as Spam, Low Priority, or High Priority based on the presence of the word "free".
- **Prior probabilities:** P(Spam) = 0.7, P(Low) = 0.2, P(High) = 0.1
- **Approach:** Naive Bayes with given likelihoods for word "free" in each class
- **Result:** `P(Spam | "free") ≈ 0.8889` — the email is classified as Spam with 88.89% confidence.

### Question 2 — Bayesian Network: Car Value Prediction
- **Network variables:** Mileage → Car Value, Engine Condition → Car Value, Air Conditioner → Car Value
- **Approach:** Constructed Conditional Probability Tables (CPTs) for each variable; computed joint probability distributions
- Full probability inference over all states of Car Value given evidence on parent nodes

### Question 3 — Naive Bayes Multi-Hypothesis Classifier
- **Dataset:** 15 samples with 5 binary input features; 4 competing hypotheses (h1, h2, h3, h4)
- **Test instance:** `(1, 1, 0, ?, ?)` 
- **Approach:** Computed posteriors `P(hᵢ | evidence) ∝ P(evidence | hᵢ) × P(hᵢ)` for all hypotheses
- **Result:** h1 and h2 achieve highest posterior probabilities and are selected as most likely hypotheses

## Key Findings

- **Base-Rate Fallacy:** Even a highly accurate medical test can produce misleading results when the disease is rare — posterior probability matters more than raw test accuracy.
- **Bayesian Networks** provide a compact, interpretable representation of probabilistic dependencies; CPTs allow efficient inference without exhaustive joint probability enumeration.
- **Naive Bayes** assumes feature independence but remains surprisingly effective for text classification (spam detection) where features are only weakly correlated.
- Keyword-based Bayesian spam filters can achieve high confidence with just one or two discriminating features when priors are well-calibrated.

## Probabilistic Concepts Covered

| Concept | Application |
|---------|-------------|
| Bayes' Theorem | Medical diagnosis, spam classification |
| Prior Probability | Class frequency in training data |
| Likelihood | Feature probability given class |
| Posterior Probability | Final classification decision |
| Conditional Probability Tables | Bayesian Network structure |
| Independence Assumption | Naive Bayes classifier |
| Joint Probability | Bayesian Network inference |

## Tech Stack

- Python 3
- NumPy (probability calculations, matrix operations)

## How to Run

```bash
jupyter notebook bayesian_networks.ipynb
```

No external data files required — all data is embedded within the notebook.
