"""Фиксированные наборы кандидатов и оценка рекомендателей."""

import pickle
import numpy as np
import pandas as pd

from .config import CAND_VAL_PATH, CAND_TEST_PATH
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, ild_at_k


class RecommenderEvaluator:
    def __init__(
        self, train_df, test_df, all_items, rating_threshold=4.0,
        n_neg=200, seed=42, use_seen_from="train", min_pos=1,
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.all_items = np.array(sorted(set(all_items)))
        self.rating_threshold = rating_threshold
        self.n_neg = int(n_neg)
        self.rng = np.random.default_rng(seed)
        self.min_pos = min_pos

        if use_seen_from == "train":
            base = train_df[["user_id", "movie_id"]]
        else:
            base = pd.concat(
                [train_df[["user_id", "movie_id"]],
                 test_df[["user_id", "movie_id"]]],
                ignore_index=True,
            )
        self.seen = base.groupby("user_id")["movie_id"].apply(set).to_dict()
        self.test_pos = (
            test_df[test_df["rating"] >= rating_threshold]
            .groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        self.candidates_by_user = {}
        self.pos_by_user = {}
        self.users_ = []

    def prepare_candidates(self, users=None, max_users=None):
        users = list(self.test_pos.keys()) if users is None else list(users)
        self.rng.shuffle(users)
        if max_users is not None:
            users = users[:max_users]

        all_items_set = set(self.all_items.tolist())
        for user_id in users:
            positives = self.test_pos.get(user_id, set())
            if len(positives) < self.min_pos:
                continue
            seen = self.seen.get(user_id, set())
            pool = np.array(list(all_items_set - seen - positives), dtype=np.int64)
            if len(pool) < self.n_neg:
                continue
            negatives = self.rng.choice(pool, size=self.n_neg, replace=False)
            candidates = np.array(list(positives) + negatives.tolist(), dtype=np.int64)
            self.candidates_by_user[user_id] = candidates
            self.pos_by_user[user_id] = positives

        self.users_ = list(self.candidates_by_user)
        return self.users_

    def evaluate_user(self, model, user_id, k=10, similarity_func=None):
        positives = self.pos_by_user[user_id]
        candidates = self.candidates_by_user[user_id]
        scores = model.score(user_id, candidates)
        top_idx = np.argsort(scores)[::-1][:k]
        recommendations = candidates[top_idx].tolist()

        result = {
            f"precision@{k}": precision_at_k(recommendations, positives, k),
            f"recall@{k}": recall_at_k(recommendations, positives, k),
            f"ndcg@{k}": ndcg_at_k(recommendations, positives, k),
        }
        if similarity_func is not None:
            result[f"ild@{k}"] = ild_at_k(recommendations, similarity_func)
        return result

    def evaluate(self, model, k=10, similarity_func=None):
        rows = [
            self.evaluate_user(model, user_id, k, similarity_func)
            for user_id in self.users_
        ]
        if not rows:
            return {}
        return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def prepare_or_load_candidates(evaluator, cache_path, max_users):
    if cache_path.exists():
        with open(cache_path, "rb") as stream:
            cache = pickle.load(stream)
        evaluator.users_ = cache["users_eval"]
        evaluator.candidates_by_user = cache["candidates_by_user"]
        evaluator.pos_by_user = cache["pos_by_user"]
        print(f"Загружены кандидаты из {cache_path.name}: {len(evaluator.users_)}")
        return evaluator.users_

    users = evaluator.prepare_candidates(max_users=max_users)
    cache = {
        "users_eval": users,
        "candidates_by_user": evaluator.candidates_by_user,
        "pos_by_user": evaluator.pos_by_user,
    }
    with open(cache_path, "wb") as stream:
        pickle.dump(cache, stream)
    return users


def make_validation_and_test_evaluators(train_fit, val_data, train_data, test_data,
                                        all_items, rating_threshold, n_neg, seed,
                                        max_users):
    val_evaluator = RecommenderEvaluator(
        train_fit, val_data, all_items, rating_threshold, n_neg, seed,
        use_seen_from="train",
    )
    test_evaluator = RecommenderEvaluator(
        train_data, test_data, all_items, rating_threshold, n_neg, seed,
        use_seen_from="train",
    )
    val_users = prepare_or_load_candidates(val_evaluator, CAND_VAL_PATH, max_users)
    test_users = prepare_or_load_candidates(test_evaluator, CAND_TEST_PATH, max_users)
    print("Пользователей для валидации:", len(val_users), "для теста:", len(test_users))
    return val_evaluator, test_evaluator
