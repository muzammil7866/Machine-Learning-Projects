from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Data" / "Multiple Disease Cleaned Data.csv"
REPORTS_DIR = ROOT_DIR / "Reports"
MODEL_DIR = ROOT_DIR / "Reports" / "SavedModels"
TARGET_COLUMN = "KidneyDisease"


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feature_df = df.drop(columns=[TARGET_COLUMN])

    categorical_features = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = feature_df.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(x_test)

    metrics = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precisionWeighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "recallWeighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "f1Weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "confusionMatrix": confusion_matrix(y_test, predictions).tolist(),
        "classificationReport": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        try:
            metrics["rocAuc"] = round(float(roc_auc_score(y_test, probabilities)), 4)
        except ValueError:
            metrics["rocAuc"] = None

    return metrics


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from dataset.")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1}).fillna(df[TARGET_COLUMN]).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }

    all_results = []
    best_model_name = None
    best_score = -1.0

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)

        results = evaluate_model(model_name, pipeline, x_test, y_test)
        all_results.append(results)

        score = results.get("rocAuc", results["f1Weighted"])
        if score is not None and score > best_score:
            best_score = score
            best_model_name = model_name
            joblib.dump(pipeline, MODEL_DIR / f"{model_name}.joblib")

    all_results.sort(key=lambda item: item.get("rocAuc", item["f1Weighted"]), reverse=True)

    summary = {
        "datasetPath": str(DATA_PATH),
        "trainRows": int(x_train.shape[0]),
        "testRows": int(x_test.shape[0]),
        "targetDistributionTrain": y_train.value_counts(normalize=True).to_dict(),
        "targetDistributionTest": y_test.value_counts(normalize=True).to_dict(),
        "bestModel": best_model_name,
        "selectionMetric": "rocAuc if available, otherwise f1Weighted",
        "results": all_results,
    }

    output_path = REPORTS_DIR / "ModelMetrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation report to: {output_path}")
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
