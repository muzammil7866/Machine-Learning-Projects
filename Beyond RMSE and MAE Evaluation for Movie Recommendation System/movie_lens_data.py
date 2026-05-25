"""Helpers for loading MovieLens data and supporting recommender evaluation."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from surprise import Dataset, Reader


class MovieLensData:
    def __init__(self, dataset_dir: Path | None = None) -> None:
        self.project_dir = Path(__file__).resolve().parent
        self.dataset_dir = dataset_dir or self._resolve_default_dataset_dir()
        self.movie_id_to_name: dict[int, str] = {}
        self.name_to_movie_id: dict[str, int] = {}
        self.ratings_path = self.dataset_dir / "ratings.csv"
        self.movies_path = self.dataset_dir / "movies.csv"

    def _resolve_default_dataset_dir(self) -> Path:
        real_dataset_dir = self.project_dir / "ml-latest-small"
        synthetic_dataset_dir = self.project_dir / "sample_data" / "ml-latest-small"

        for candidate in (real_dataset_dir, synthetic_dataset_dir):
            if (candidate / "ratings.csv").exists() and (candidate / "movies.csv").exists():
                return candidate

        return synthetic_dataset_dir

    def load_movie_lens_latest_small(self) -> Dataset:
        if not self.ratings_path.exists() or not self.movies_path.exists():
            raise FileNotFoundError(
                "No dataset found. Use the bundled synthetic files under "
                f"{self.project_dir / 'sample_data' / 'ml-latest-small'} or place real MovieLens files under "
                f"{self.project_dir / 'ml-latest-small'}."
            )

        self.movie_id_to_name.clear()
        self.name_to_movie_id.clear()

        reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
        ratings_dataset = Dataset.load_from_file(str(self.ratings_path), reader=reader)

        with self.movies_path.open(newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader, None)
            for row in movie_reader:
                movie_id = int(row[0])
                movie_name = row[1]
                self.movie_id_to_name[movie_id] = movie_name
                self.name_to_movie_id[movie_name] = movie_id

        return ratings_dataset

    def get_user_ratings(self, user_id: int) -> list[tuple[int, float]]:
        user_ratings: list[tuple[int, float]] = []
        hit_user = False
        with self.ratings_path.open(newline="") as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader, None)
            for row in rating_reader:
                current_user = int(row[0])
                if user_id == current_user:
                    user_ratings.append((int(row[1]), float(row[2])))
                    hit_user = True
                elif hit_user:
                    break
        return user_ratings

    def get_popularity_ranks(self) -> dict[int, int]:
        ratings = defaultdict(int)
        rankings: dict[int, int] = {}

        with self.ratings_path.open(newline="") as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader, None)
            for row in rating_reader:
                ratings[int(row[1])] += 1

        for rank, (movie_id, _) in enumerate(sorted(ratings.items(), key=lambda item: item[1], reverse=True), start=1):
            rankings[movie_id] = rank

        return rankings

    def get_genres(self) -> dict[int, list[int]]:
        genres = defaultdict(list)
        genre_ids: dict[str, int] = {}
        next_genre_id = 0

        with self.movies_path.open(newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader, None)
            for row in movie_reader:
                movie_id = int(row[0])
                genre_id_list: list[int] = []
                for genre in row[2].split("|"):
                    if genre not in genre_ids:
                        genre_ids[genre] = next_genre_id
                        next_genre_id += 1
                    genre_id_list.append(genre_ids[genre])
                genres[movie_id] = genre_id_list

        for movie_id, genre_id_list in list(genres.items()):
            bitfield = [0] * next_genre_id
            for genre_id in genre_id_list:
                bitfield[genre_id] = 1
            genres[movie_id] = bitfield

        return genres

    def get_years(self) -> dict[int, int]:
        pattern = re.compile(r"(?:\((\d{4})\))?\s*$")
        years: dict[int, int] = {}
        with self.movies_path.open(newline="", encoding="ISO-8859-1") as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader, None)
            for row in movie_reader:
                movie_id = int(row[0])
                match = pattern.search(row[1])
                if match and match.group(1):
                    years[movie_id] = int(match.group(1))
        return years

    def get_mise_en_scene(self) -> dict[int, list[float]]:
        feature_file = self.dataset_dir / "LLVisualFeatures13K_Log.csv"
        if not feature_file.exists():
            return {}

        mise_en_scene = defaultdict(list)
        with feature_file.open(newline="") as csvfile:
            feature_reader = csv.reader(csvfile)
            next(feature_reader, None)
            for row in feature_reader:
                movie_id = int(row[0])
                mise_en_scene[movie_id] = [float(value) for value in row[1:8]]
        return mise_en_scene

    def get_movie_name(self, movie_id: int) -> str:
        return self.movie_id_to_name.get(movie_id, "")

    def get_movie_id(self, movie_name: str) -> int:
        return self.name_to_movie_id.get(movie_name, 0)
