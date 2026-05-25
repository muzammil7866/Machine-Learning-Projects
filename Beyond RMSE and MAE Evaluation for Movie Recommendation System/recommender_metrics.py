"""Metric helpers for MovieLens recommender evaluation."""

from __future__ import annotations

import itertools
from collections import defaultdict

from surprise import accuracy


class RecommenderMetrics:
    @staticmethod
    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)

    @staticmethod
    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)

    @staticmethod
    def get_top_n(predictions, n: int = 10, minimum_rating: float = 4.0):
        top_n = defaultdict(list)
        for user_id, movie_id, _, estimated_rating, _ in predictions:
            if estimated_rating >= minimum_rating:
                top_n[int(user_id)].append((int(movie_id), estimated_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda item: item[1], reverse=True)
            top_n[user_id] = ratings[:n]

        return top_n

    @staticmethod
    def hit_rate(top_n_predicted, left_out_predictions):
        hits = 0
        total = 0
        for left_out in left_out_predictions:
            user_id = int(left_out[0])
            left_out_movie_id = int(left_out[1])
            hit = any(int(left_out_movie_id) == int(movie_id) for movie_id, _ in top_n_predicted[user_id])
            hits += int(hit)
            total += 1
        return hits / total if total else 0.0

    @staticmethod
    def cumulative_hit_rate(top_n_predicted, left_out_predictions, rating_cutoff: float = 0):
        hits = 0
        total = 0
        for user_id, left_out_movie_id, actual_rating, _, _ in left_out_predictions:
            if actual_rating >= rating_cutoff:
                hit = any(int(left_out_movie_id) == int(movie_id) for movie_id, _ in top_n_predicted[int(user_id)])
                hits += int(hit)
                total += 1
        return hits / total if total else 0.0

    @staticmethod
    def rating_hit_rate(top_n_predicted, left_out_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)
        for user_id, left_out_movie_id, actual_rating, _, _ in left_out_predictions:
            hit = any(int(left_out_movie_id) == int(movie_id) for movie_id, _ in top_n_predicted[int(user_id)])
            if hit:
                hits[actual_rating] += 1
            total[actual_rating] += 1

        for rating in sorted(hits):
            print(rating, hits[rating] / total[rating])

    @staticmethod
    def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions):
        summation = 0.0
        total = 0
        for user_id, left_out_movie_id, _, _, _ in left_out_predictions:
            hit_rank = 0
            for rank, (movie_id, _) in enumerate(top_n_predicted[int(user_id)], start=1):
                if int(left_out_movie_id) == int(movie_id):
                    hit_rank = rank
                    break
            if hit_rank > 0:
                summation += 1.0 / hit_rank
            total += 1
        return summation / total if total else 0.0

    @staticmethod
    def user_coverage(top_n_predicted, num_users: int, rating_threshold: float = 0):
        hits = 0
        for user_id, ratings in top_n_predicted.items():
            if any(predicted_rating >= rating_threshold for _, predicted_rating in ratings):
                hits += 1
        return hits / num_users if num_users else 0.0

    @staticmethod
    def diversity(top_n_predicted, trainset):
        item_users: dict[int, set[int]] = defaultdict(set)
        for inner_user_id, ratings in trainset.ur.items():
            for inner_item_id, _ in ratings:
                item_users[inner_item_id].add(inner_user_id)

        n = 0
        total = 0.0
        for user_id in top_n_predicted:
            for pair in itertools.combinations(top_n_predicted[user_id], 2):
                movie_1 = pair[0][0]
                movie_2 = pair[1][0]
                try:
                    inner_id_1 = trainset.to_inner_iid(str(movie_1))
                    inner_id_2 = trainset.to_inner_iid(str(movie_2))
                except ValueError:
                    continue

                users_1 = item_users.get(inner_id_1, set())
                users_2 = item_users.get(inner_id_2, set())
                union_size = len(users_1 | users_2)
                similarity = len(users_1 & users_2) / union_size if union_size else 0.0
                total += similarity
                n += 1
        return 1 - (total / n) if n else 0.0

    @staticmethod
    def novelty(top_n_predicted, rankings):
        total = 0
        n = 0
        for user_id in top_n_predicted:
            for movie_id, _ in top_n_predicted[user_id]:
                total += rankings[movie_id]
                n += 1
        return total / n if n else 0.0
