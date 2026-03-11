from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "Data" / "Multiple Disease Raw Data.csv"
CLEANED_DATA_PATH = ROOT_DIR / "Data" / "Multiple Disease Cleaned Data.csv"


# Columns with many missing values are dropped to keep the baseline cleaning simple and reproducible.
MISSING_VALUE_THRESHOLD = 20


def prepare_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)

    missing_counts = df.isna().sum()
    columns_to_drop = missing_counts[missing_counts > MISSING_VALUE_THRESHOLD].index.tolist()

    cleaned_df = df.drop(columns=columns_to_drop).copy()

    for column in cleaned_df.columns:
        if cleaned_df[column].isna().any():
            mode_series = cleaned_df[column].mode(dropna=True)
            if not mode_series.empty:
                cleaned_df[column] = cleaned_df[column].fillna(mode_series.iloc[0])

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    return cleaned_df


if __name__ == "__main__":
    cleaned = prepare_dataset()
    print(f"Saved cleaned dataset to: {CLEANED_DATA_PATH}")
    print(f"Rows: {cleaned.shape[0]}, Columns: {cleaned.shape[1]}")
