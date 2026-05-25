"""Download the MovieLens latest-small dataset into the local workspace."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens_latest_small(target_dir: Path | None = None) -> Path:
    target_dir = target_dir or Path(__file__).resolve().parent / "ml-latest-small"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if (target_dir / "ratings.csv").exists() and (target_dir / "movies.csv").exists():
        print(f"MovieLens data already exists in {target_dir}")
        return target_dir

    print(f"Downloading MovieLens latest-small dataset to {target_dir}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "ml-latest-small.zip"
        with urllib.request.urlopen(DATASET_URL) as response, zip_path.open("wb") as zip_file:
            shutil.copyfileobj(response, zip_file)

        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir.parent)

    print("Download complete.")
    return target_dir


def main() -> None:
    download_movielens_latest_small()


if __name__ == "__main__":
    main()