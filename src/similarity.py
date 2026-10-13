"""Жанровая матрица сходства объектов."""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import SIM_PATH, MOV2IDX_PATH


def build_similarity_and_index(movies):
    genres_onehot = movies["genres"].str.get_dummies(sep="|")
    genre_matrix = genres_onehot.to_numpy(dtype=np.float32)
    similarity = cosine_similarity(genre_matrix).astype(np.float32)
    movie_to_idx = {
        int(movie_id): idx for idx, movie_id in enumerate(movies["movie_id"].values)
    }
    return similarity, movie_to_idx


def load_or_build_similarity(movies, sim_path=SIM_PATH, idx_path=MOV2IDX_PATH):
    if sim_path.exists() and idx_path.exists():
        similarity = np.load(sim_path)
        with open(idx_path, "rb") as stream:
            movie_to_idx = pickle.load(stream)
        valid = (
            isinstance(movie_to_idx, dict)
            and similarity.ndim == 2
            and similarity.shape[0] == similarity.shape[1]
            and len(movie_to_idx) == similarity.shape[0]
            and set(movie_to_idx) == set(movies["movie_id"].astype(int))
        )
        if valid:
            print("Загружена матрица сходства:", similarity.shape)
            return similarity, movie_to_idx
        print("Кэш сходства не соответствует текущим данным; пересоздание.")

    similarity, movie_to_idx = build_similarity_and_index(movies)
    np.save(sim_path, similarity)
    with open(idx_path, "wb") as stream:
        pickle.dump(movie_to_idx, stream)
    print("Матрица сходства сохранена:", similarity.shape)
    return similarity, movie_to_idx


def make_genre_similarity(similarity_matrix, movie_to_idx):
    def genre_similarity(movie_a, movie_b):
        return float(
            similarity_matrix[
                movie_to_idx[int(movie_a)], movie_to_idx[int(movie_b)]
            ]
        )
    return genre_similarity
