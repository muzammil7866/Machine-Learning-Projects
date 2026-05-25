"""End-to-end MovieLens evaluation script."""

from __future__ import annotations

from pathlib import Path

from surprise import SVD
from surprise.model_selection import LeaveOneOut, train_test_split

from movie_lens_data import MovieLensData
from recommender_metrics import RecommenderMetrics


def evaluate(dataset_dir: Path | None = None) -> None:
    movie_lens = MovieLensData(dataset_dir=dataset_dir)

    print("Loading movie ratings...")
    data = movie_lens.load_movie_lens_latest_small()

    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = movie_lens.get_popularity_ranks()
    full_train_set = data.build_full_trainset()

    print("\nBuilding recommendation model...")
    train_set, test_set = train_test_split(data, test_size=0.25, random_state=1)

    algo = SVD(random_state=10)
    algo.fit(train_set)

    print("\nComputing recommendations...")
    predictions = algo.test(test_set)

    print("\nEvaluating accuracy of model...")
    print("RMSE:", RecommenderMetrics.rmse(predictions))
    print("MAE:", RecommenderMetrics.mae(predictions))

    print("\nEvaluating top-10 recommendations...")
    leave_one_out = LeaveOneOut(n_splits=1, random_state=1)

    for leave_train_set, leave_test_set in leave_one_out.split(data):
        print("Computing recommendations with leave-one-out...")
        algo.fit(leave_train_set)

        print("Predict ratings for left-out set...")
        left_out_predictions = algo.test(leave_test_set)

        print("Predict all missing ratings...")
        big_test_set = leave_train_set.build_anti_testset()
        all_predictions = algo.test(big_test_set)

        print("Compute top 10 recs per user...")
        top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n=10)

        print("\nHit Rate:", RecommenderMetrics.hit_rate(top_n_predicted, left_out_predictions))
        print("\nrHR (Hit Rate by Rating value):")
        RecommenderMetrics.rating_hit_rate(top_n_predicted, left_out_predictions)
        print("\ncHR (Cumulative Hit Rate, rating >= 4):", RecommenderMetrics.cumulative_hit_rate(top_n_predicted, left_out_predictions, 4.0))
        print("\nARHR (Average Reciprocal Hit Rank):", RecommenderMetrics.average_reciprocal_hit_rank(top_n_predicted, left_out_predictions))

    print("\nComputing complete recommendations, no hold outs...")
    algo.fit(full_train_set)
    big_test_set = full_train_set.build_anti_testset()
    all_predictions = algo.test(big_test_set)
    top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n=10)

    print("\nUser coverage:", RecommenderMetrics.user_coverage(top_n_predicted, full_train_set.n_users, rating_threshold=4.0))
    print("\nDiversity:", RecommenderMetrics.diversity(top_n_predicted, full_train_set))
    print("\nNovelty (average popularity rank):", RecommenderMetrics.novelty(top_n_predicted, rankings))


def main() -> None:
    try:
        evaluate()
    except FileNotFoundError as exc:
        print(exc)
        print("Place ratings.csv and movies.csv in ml-latest-small/, then run the script again.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
