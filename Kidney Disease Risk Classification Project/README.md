# Kidney Disease Risk Classification Project

This project predicts Kidney Disease risk from health indicators using a cleaned version of the Multiple Disease dataset.

## Project Structure

- Data/
  - Multiple Disease Raw Data.csv
  - Multiple Disease Cleaned Data.csv
- Notebooks/
  - Kidney Disease Modeling.ipynb
- Source/
  - PrepareDataset.py
  - TrainKidneyDiseaseModels.py
- Reports/
  - SavedModels/ (generated after training)
  - ModelMetrics.json (generated after training)
- requirements.txt

## Setup

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Pipeline

### 1) Prepare cleaned dataset from raw data

```bash
python Source/PrepareDataset.py
```

### 2) Train and evaluate models

```bash
python Source/TrainKidneyDiseaseModels.py
```

This trains Logistic Regression, Decision Tree, and Random Forest using a consistent preprocessing pipeline:
- Missing value imputation
- One-hot encoding for categorical features
- Feature scaling for numeric features
- Stratified train-test split

## Outputs

After running training, the project generates:
- Reports/ModelMetrics.json: complete evaluation summary (accuracy, precision, recall, F1, confusion matrix, ROC-AUC where available)
- Reports/SavedModels/: best-performing model pipeline serialized as a .joblib file

## Target Variable

- Target column: KidneyDisease
- Class mapping for training: No -> 0, Yes -> 1

## Notes

- The notebook provides exploratory and comparative modeling workflow.
- The Python scripts provide a reproducible, git-friendly execution path for model training and reporting.
