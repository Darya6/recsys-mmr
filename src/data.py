"""Загрузка, проверка и временное разбиение MovieLens-1M."""

import zipfile
import numpy as np
import pandas as pd

from .config import (
    ZIP_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, TRAIN_FIT_PATH, VAL_DATA_PATH,
    MIN_RATINGS_PER_USER,
)


def load_movielens(zip_path=ZIP_PATH):
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Не найден архив {zip_path}. Поместите MovieLens-1M в data/Movielens.zip."
        )

    with zipfile.ZipFile(zip_path, "r") as archive:
        ratings = pd.read_csv(
            archive.open("ratings.dat"), sep="::", header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python", encoding="latin-1",
        )
        users = pd.read_csv(
            archive.open("users.dat"), sep="::", header=None,
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python", encoding="latin-1",
        )
        movies = pd.read_csv(
            archive.open("movies.dat"), sep="::", header=None,
            names=["movie_id", "title", "genres"],
            engine="python", encoding="latin-1",
        )

    duplicates = int(ratings.duplicated(subset=["user_id", "movie_id"]).sum())
    if duplicates:
        ratings = (
            ratings.sort_values("timestamp")
            .drop_duplicates(subset=["user_id", "movie_id"], keep="last")
            .reset_index(drop=True)
        )

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Обнаружены рейтинги вне диапазона 1–5.")

    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
    if ratings["timestamp"].isna().any():
        raise ValueError("В timestamp обнаружены пропуски или некорректные значения.")

    print("Загружено:", ratings.shape, users.shape, movies.shape)
    print("Пользователей:", ratings["user_id"].nunique())
    print("Фильмов:", ratings["movie_id"].nunique())
    print("Оценок:", len(ratings))
    if duplicates:
        print("Удалено дубликатов user_id/movie_id:", duplicates)

    return ratings, users, movies


def filter_users(ratings, min_ratings=MIN_RATINGS_PER_USER):
    counts = ratings.groupby("user_id").size()
    active_users = counts[counts >= min_ratings].index
    filtered = ratings[ratings["user_id"].isin(active_users)].copy()
    print(
        f"Пользователей до/после фильтрации: "
        f"{ratings['user_id'].nunique()} / {filtered['user_id'].nunique()}"
    )
    return filtered


def temporal_split(ratings, use_cache=True):
    paths = [TRAIN_DATA_PATH, TEST_DATA_PATH, TRAIN_FIT_PATH, VAL_DATA_PATH]
    if use_cache and all(path.exists() for path in paths):
        print("Загрузка существующих разбиений из cache/...")
        train_data, test_data, train_fit, val_data = [
            pd.read_csv(path) for path in paths
        ]
        return train_data, test_data, train_fit, val_data

    data = ratings.copy()
    data["rank"] = data.groupby("user_id")["timestamp"].rank(
        method="first", ascending=False
    )
    data["count"] = data.groupby("user_id")["user_id"].transform("count")

    test_mask = data["rank"] <= np.ceil(data["count"] * 0.2)
    test_data = data[test_mask].drop(columns=["rank", "count"])
    train_data = data[~test_mask].drop(columns=["rank", "count"]).copy()

    train_data["rank"] = train_data.groupby("user_id")["timestamp"].rank(
        method="first", ascending=False
    )
    train_data["count"] = train_data.groupby("user_id")["user_id"].transform("count")

    val_mask = train_data["rank"] <= np.ceil(train_data["count"] * 0.1)
    val_data = train_data[val_mask].drop(columns=["rank", "count"])
    train_fit = train_data[~val_mask].drop(columns=["rank", "count"])

    for frame, path in zip(
        [train_data, test_data, train_fit, val_data], paths
    ):
        frame.to_csv(path, index=False)

    print(f"Fit={len(train_fit)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, test_data, train_fit, val_data


def plot_eda(ratings, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 4))
    sns.countplot(x="rating", data=ratings)
    plt.title("Распределение оценок")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png", dpi=200)
    plt.close()

    activity = ratings.groupby("user_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(activity, bins=50, edgecolor="black")
    plt.title("Оценок на пользователя")
    plt.xlabel("Число оценок")
    plt.ylabel("Число пользователей")
    plt.tight_layout()
    plt.savefig(output_dir / "user_activity.png", dpi=200)
    plt.close()

    n_users = ratings["user_id"].nunique()
    n_items = ratings["movie_id"].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_items)
    print(f"Разреженность матрицы: {sparsity * 100:.2f}%")
    print(f"Доля оценок >= 4: {(ratings['rating'] >= 4).mean():.3f}")
