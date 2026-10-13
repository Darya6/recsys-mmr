"""Implicit ALS и MMR-переранжирование."""

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSImplicitRecommender:
    def __init__(self, rating_threshold=4.0, factors=64, regularization=0.05,
                 iterations=30, seed=42, alpha=15):
        self.rating_threshold = rating_threshold
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.alpha = alpha
        self.model = None
        self.user2idx = {}
        self.item2idx = {}
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_df):
        positive = train_df[train_df["rating"] >= self.rating_threshold].copy()
        users = sorted(positive["user_id"].unique().tolist())
        items = sorted(positive["movie_id"].unique().tolist())
        self.user2idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.item2idx = {item_id: idx for idx, item_id in enumerate(items)}

        user_idx = positive["user_id"].map(self.user2idx).to_numpy()
        item_idx = positive["movie_id"].map(self.item2idx).to_numpy()
        values = (positive["rating"].values * self.alpha).astype(np.float32)
        user_item = sp.csr_matrix(
            (values, (user_idx, item_idx)), shape=(len(users), len(items))
        )

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
        )
        self.model.fit(user_item)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        return self

    def score(self, user_id, item_ids):
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            return np.zeros(len(item_ids), dtype=np.float32)

        user_vector = self.user_factors[user_idx]
        scores = np.zeros(len(item_ids), dtype=np.float32)
        for idx, item_id in enumerate(item_ids):
            item_idx = self.item2idx.get(int(item_id))
            if item_idx is not None:
                scores[idx] = float(user_vector @ self.item_factors[item_idx])
        return scores


def mmr_rerank(movie_ids, relevance, k, lambda_, sim_func):
    if len(movie_ids) == 0:
        return []
    relevance = relevance.astype(np.float32)
    relevance = (
        (relevance - relevance.min())
        / (relevance.max() - relevance.min() + 1e-9)
    )
    relevance_map = dict(zip(movie_ids, relevance))
    selected = []
    remaining = list(movie_ids)

    while len(selected) < k and remaining:
        best_item = None
        best_score = -1e18
        for candidate in remaining:
            rel = relevance_map[candidate]
            max_sim = (
                0.0 if not selected
                else max(sim_func(candidate, chosen) for chosen in selected)
            )
            score = lambda_ * rel - (1 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_item = candidate
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


class ALSPlusMMRRecommender:
    def __init__(self, als_base, lambda_, top_k=10, top_n=200, sim_func=None):
        self.als = als_base
        self.lambda_ = float(lambda_)
        self.top_k = int(top_k)
        self.top_n = int(top_n)
        self.sim_func = sim_func

    def fit(self, train_df):
        return self

    def score(self, user_id, item_ids):
        relevance = self.als.score(user_id, item_ids)
        n = min(self.top_n, len(item_ids))
        top_idx = np.argsort(relevance)[::-1][:n]
        shortlist_items = item_ids[top_idx].astype(int)
        shortlist_relevance = relevance[top_idx]

        selected = mmr_rerank(
            shortlist_items.tolist(), shortlist_relevance, self.top_k,
            self.lambda_, self.sim_func,
        )
        scores = np.full(len(item_ids), -1e9, dtype=np.float32)
        positions = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
        for rank, item_id in enumerate(selected):
            idx = positions.get(int(item_id))
            if idx is not None:
                scores[idx] = float(self.top_k - rank)
        return scores
