"""Метрики ранжирования и разнообразия."""

import numpy as np


def precision_at_k(recommendations, relevant_items, k):
    return len(set(recommendations[:k]) & relevant_items) / k


def recall_at_k(recommendations, relevant_items, k):
    if not relevant_items:
        return 0.0
    return len(set(recommendations[:k]) & relevant_items) / len(relevant_items)


def ndcg_at_k(recommendations, relevant_items, k):
    rec_k = recommendations[:k]
    gains = np.array(
        [1.0 if item in relevant_items else 0.0 for item in rec_k],
        dtype=np.float32,
    )
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal_len = min(len(relevant_items), k)
    if ideal_len == 0:
        return 0.0
    idcg = float(discounts[:ideal_len].sum())
    return dcg / idcg if idcg > 0 else 0.0


def ild_at_k(recommendations, similarity_func):
    if len(recommendations) < 2:
        return 0.0
    similarities = [
        float(similarity_func(recommendations[i], recommendations[j]))
        for i in range(len(recommendations))
        for j in range(i + 1, len(recommendations))
    ]
    return 1.0 - float(np.mean(similarities)) if similarities else 0.0


def f_tradeoff(ndcg, ild, eps=1e-12):
    return (2.0 * ndcg * ild) / (ndcg + ild + eps)
