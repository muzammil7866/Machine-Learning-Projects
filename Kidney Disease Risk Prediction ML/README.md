# Kidney Disease Risk Prediction — Multi-Algorithm Classification

A supervised machine learning project that predicts **Kidney Disease risk** from a large-scale multiple-disease health survey dataset. The project covers the full ML pipeline: data exploration, cleaning, feature engineering, model training, hyperparameter tuning, and comparative evaluation of multiple classification algorithms.

---

## Dataset

| Property | Detail |
|---|---|
| File | `data/Multiple_Disease_Data.csv` |
| Records | 59,068 patients |
| Features | 18 (demographic + health indicators) |
| Target Variable | `KidneyDisease` (binary: Yes / No) |
| Source | Multi-disease health survey (CDC BRFSS-style) |

### Features

| Feature | Type | Description |
|---|---|---|
| BMI | Float | Body Mass Index |
| Smoking | Categorical | Smoking history (Yes/No) |
| AlcoholDrinking | Categorical | Heavy alcohol use (Yes/No) |
| Stroke | Categorical | History of stroke (Yes/No) |
| PhysicalHealth | Integer | Days of poor physical health (0–30) |
| MentalHealth | Float | Days of poor mental health (0–30) |
| DiffWalking | Categorical | Difficulty walking (Yes/No) |
| Sex | Categorical | Biological sex |
| AgeCategory | Categorical | Age group (18–24 … 80+) |
| Race | Categorical | Race/ethnicity |
| Diabetic | Categorical | Diabetes status |
| PhysicalActivity | Categorical | Regular physical activity (Yes/No) |
| GenHealth | Categorical | Self-reported general health |
| SleepTime | Integer | Average sleep hours per night |
| HeartDisease | Categorical | History of heart disease |
| Asthma | Categorical | Asthma diagnosis |
| SkinCancer | Categorical | Skin cancer history |
| **KidneyDisease** | **Categorical** | **Target — Kidney disease diagnosis** |

---

## Project Structure

```
kidney-disease-risk-prediction-ml/
├── README.md
├── kidney_disease_prediction.ipynb   # Main notebook (full pipeline)
├── data/
│   ├── Multiple_Disease_Data.csv     # Raw dataset (59,068 records)
│   └── Multiple_Disease_Cleaned.csv  # Preprocessed dataset
└── docs/
    └── ML_Project_Proposal.pdf       # Project proposal document
```

---

## ML Pipeline

### 1. Exploratory Data Analysis
- Dataset shape, dtypes, and summary statistics
- Null value heatmap and percentage analysis
- Feature correlation matrix (coolwarm heatmap)
- Class distribution visualization

### 2. Data Preprocessing
- **Null handling:** Missing values in `MentalHealth` (28 rows) and `GenHealth` (9 rows) imputed using mode
- **Feature selection:** Dropped `HeartDisease`, `Race`, `Sex`, `SleepTime` based on low correlation with `KidneyDisease`
- **Label encoding:** `LabelEncoder` applied to all categorical features
- **One-hot encoding:** `pd.get_dummies` applied to remaining categoricals
- **Standard scaling:** `StandardScaler` applied to `BMI`, `PhysicalHealth`, `MentalHealth`, `SleepTime`

### 3. Train/Test Split
- 80/20 split with `stratify=label_encoded` to preserve class balance

---

## Algorithms Implemented

| Algorithm | Library | Key Configuration |
|---|---|---|
| Support Vector Machine (SVM) | `sklearn.svm.SVC` | RBF kernel, C=1, gamma='scale' |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | n_estimators=150 |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | max_iter=1000, C=1 |
| Linear Regression (baseline) | `sklearn.linear_model.LinearRegression` | Thresholded at 0.5 for binary output |
| Ridge Regression | `sklearn.linear_model.Ridge` | L2 regularization, alpha sweep |
| Lasso Regression | `sklearn.linear_model.Lasso` | L1 regularization, alpha sweep |

### Hyperparameter Experiments
- **SVM:** Regularization sweep over C ∈ {0.1, 1, 10, 100, 1000}
- **Random Forest:** Estimator sweep over n_estimators ∈ {50, 100, 150, 200, 250}
- **Logistic Regression:** Regularization sweep over C ∈ {0.1, 1, 10, 100, 1000}
- **Ridge/Lasso:** Alpha sweep over α ∈ {0.1, 1, 10, 100, 1000}

---

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score per class)
- **Confusion Matrix** (heatmap visualization)
- **ROC Curve** with AUC score (binary classification)
- Hyperparameter effect plots (accuracy vs. regularization strength)

---

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## How to Run

1. Clone the repository and navigate to this project folder.
2. Ensure the dataset is present at `data/Multiple_Disease_Data.csv`.
3. Launch Jupyter:
   ```bash
   jupyter notebook kidney_disease_prediction.ipynb
   ```
4. Run all cells in order (`Kernel > Restart & Run All`).

---

## Course Context

This project was developed as the final project for a **Machine Learning course**, covering supervised classification techniques including linear models, tree-based ensembles, kernel methods, and regularization strategies.
