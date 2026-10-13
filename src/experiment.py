"""Настройка ALS, подбор lambda и разреженности."""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import (
    OUTPUT_DIR, ALS_GRID_PATH, ADAPTIVE_GRID_PATH, SPARSITY_RESULTS_PATH,
    TOP_N, RATING_THRESHOLD, K, N_NEG, MAX_USERS_EVAL, RANDOM_SEED,
    ALS_ITERATIONS, ALS_ALPHA, KEEP_RATIOS,
)
from .data import load_movielens, filter_users, temporal_split, plot_eda
from .similarity import load_or_build_similarity, make_genre_similarity
from .evaluator import make_validation_and_test_evaluators
from .metrics import f_tradeoff
from .models import ALSImplicitRecommender, ALSPlusMMRRecommender


def make_sparse_train(train_df, keep_ratio, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    pieces = []
    for user_id in train_df["user_id"].unique():
        user_rows = train_df[train_df["user_id"] == user_id]
        n_keep = max(1, int(len(user_rows) * keep_ratio))
        indices = rng.choice(len(user_rows), size=n_keep, replace=False)
        pieces.append(user_rows.iloc[indices])
    return pd.concat(pieces, ignore_index=True)


def fixed_lambda(user_id, **kwargs):
    return 0.5


def make_adaptive_lambda(thresholds, lambdas, activity_dict):
    if len(lambdas) != len(thresholds) + 1:
        raise ValueError("Число lambda должно быть на единицу больше числа порогов.")

    def adaptive(user_id, **kwargs):
        activity = activity_dict.get(user_id, 0)
        for idx, threshold in enumerate(thresholds):
            if activity < threshold:
                return lambdas[idx]
        return lambdas[-1]
    return adaptive


def tune_als(train_fit, val_evaluator):
    print("\nПодбор гиперпараметров ALS...")
    rows = []
    for factors in [32, 64, 128]:
        for regularization in [0.01, 0.05, 0.1]:
            model = ALSImplicitRecommender(
                rating_threshold=RATING_THRESHOLD,
                factors=factors, regularization=regularization,
                iterations=ALS_ITERATIONS, seed=RANDOM_SEED, alpha=ALS_ALPHA,
            ).fit(train_fit)
            metrics = val_evaluator.evaluate(model, k=K)
            score = metrics.get(f"ndcg@{K}", 0.0)
            rows.append({
                "factors": factors, "regularization": regularization,
                "ndcg": score,
            })
            print(f"factors={factors}, reg={regularization} -> NDCG={score:.4f}")

    grid = pd.DataFrame(rows)
    grid.to_csv(ALS_GRID_PATH, index=False)
    best = grid.loc[grid["ndcg"].idxmax()]
    best_factors = int(best["factors"])
    best_reg = float(best["regularization"])
    print(f"Лучшие параметры ALS: factors={best_factors}, reg={best_reg}")
    return best_factors, best_reg


def tune_adaptive_lambda(train_fit, val_evaluator, best_factors, best_reg,
                         genre_similarity):
    print("\nПодбор адаптивной функции lambda(n) на валидации...")
    activity = train_fit.groupby("user_id").size().to_dict()
    als = ALSImplicitRecommender(
        rating_threshold=RATING_THRESHOLD,
        factors=best_factors, regularization=best_reg,
        iterations=ALS_ITERATIONS, seed=RANDOM_SEED, alpha=ALS_ALPHA,
    ).fit(train_fit)

    attempts = [
        ("V1: 20/50, lambda=0.8/0.5/0.2", (20, 50), (0.8, 0.5, 0.2)),
        ("V2: 20/50, lambda=0.2/0.5/0.8", (20, 50), (0.2, 0.5, 0.8)),
        ("V3: 5/10/20/50/100, descending",
         (5, 10, 20, 50, 100), (0.8, 0.7, 0.6, 0.5, 0.4, 0.3)),
        ("V4: 5/10/20/50/100, ascending",
         (5, 10, 20, 50, 100), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
    ]
    results = []
    for name, thresholds, lambdas in attempts:
        adaptive = make_adaptive_lambda(thresholds, lambdas, activity)
        rows = []
        for user_id in val_evaluator.users_:
            lam = adaptive(user_id)
            reranker = ALSPlusMMRRecommender(
                als, lam, top_k=K, top_n=TOP_N, sim_func=genre_similarity
            )
            metrics = val_evaluator.evaluate_user(
                reranker, user_id, k=K, similarity_func=genre_similarity
            )
            metrics[f"f_tradeoff@{K}"] = f_tradeoff(
                metrics[f"ndcg@{K}"], metrics[f"ild@{K}"]
            )
            rows.append(metrics)

        frame = pd.DataFrame(rows)
        results.append({
            "attempt": name,
            "thresholds": repr(thresholds),
            "lambdas": repr(lambdas),
            f"f_tradeoff@{K}": frame[f"f_tradeoff@{K}"].mean(),
            f"ndcg@{K}": frame[f"ndcg@{K}"].mean(),
            f"ild@{K}": frame[f"ild@{K}"].mean(),
        })
        print(
            f"{name} -> F={results[-1][f'f_tradeoff@{K}']:.4f}, "
            f"NDCG={results[-1][f'ndcg@{K}']:.4f}, "
            f"ILD={results[-1][f'ild@{K}']:.4f}"
        )

    grid = pd.DataFrame(results).sort_values(
        by=f"f_tradeoff@{K}", ascending=False
    )
    grid.to_csv(ADAPTIVE_GRID_PATH, index=False)
    best = grid.iloc[0]
    thresholds = ast.literal_eval(best["thresholds"])
    lambdas = ast.literal_eval(best["lambdas"])
    print("Лучшая адаптивная конфигурация:", best["attempt"])
    return thresholds, lambdas, grid


def run_sparsity_experiment(train_data, test_data, all_items, test_evaluator,
                            best_factors, best_reg, thresholds, lambdas,
                            genre_similarity):
    print("\nЭксперимент по разреженности...")
    results = []
    for keep_ratio in KEEP_RATIOS:
        train_sparse = make_sparse_train(train_data, keep_ratio)
        removed_fraction = 1.0 - keep_ratio
        print(
            f"\nУдалено оценок: {removed_fraction:.0%}; "
            f"сохранено: {keep_ratio:.0%}"
        )

        activity = train_sparse.groupby("user_id").size().to_dict()
        adaptive = make_adaptive_lambda(thresholds, lambdas, activity)
        als = ALSImplicitRecommender(
            rating_threshold=RATING_THRESHOLD,
            factors=best_factors, regularization=best_reg,
            iterations=ALS_ITERATIONS, seed=RANDOM_SEED, alpha=ALS_ALPHA,
        ).fit(train_sparse)

        # Кандидаты, положительные тестовые объекты и набор пользователей
        # берутся из одного фиксированного test_evaluator для всех уровней.
        for strategy_name, strategy_fn in [
            ("fixed", fixed_lambda), ("adaptive", adaptive)
        ]:
            rows = []
            for user_id in tqdm(
                test_evaluator.users_, desc=strategy_name, leave=False
            ):
                lam = strategy_fn(user_id)
                reranker = ALSPlusMMRRecommender(
                    als, lam, top_k=K, top_n=TOP_N, sim_func=genre_similarity
                )
                metrics = test_evaluator.evaluate_user(
                    reranker, user_id, k=K, similarity_func=genre_similarity
                )
                metrics[f"f_tradeoff@{K}"] = f_tradeoff(
                    metrics[f"ndcg@{K}"], metrics[f"ild@{K}"]
                )
                metrics["user_id"] = user_id
                metrics["lambda"] = lam
                rows.append(metrics)

            frame = pd.DataFrame(rows)
            results.append({
                "keep_ratio": keep_ratio,
                "train_sparsity": removed_fraction,
                "strategy": strategy_name,
                "ild": frame[f"ild@{K}"].mean(),
                "ndcg": frame[f"ndcg@{K}"].mean(),
                f"f_tradeoff@{K}": frame[f"f_tradeoff@{K}"].mean(),
                "mean_lambda": frame["lambda"].mean(),
            })

        pd.DataFrame(results).to_csv(SPARSITY_RESULTS_PATH, index=False)
        print("Результаты сохранены:", SPARSITY_RESULTS_PATH)

    return pd.DataFrame(results)


def save_plots(grid, results):
    grid_plot = grid.copy()
    grid_plot["attempt_id"] = grid_plot["attempt"].str.extract(r"^(V\d+)")
    grid_plot = grid_plot.sort_values("attempt_id")

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(
        grid_plot["attempt_id"], grid_plot[f"f_tradeoff@{K}"],
        edgecolor="black", alpha=0.85,
    )
    plt.ylabel(f"F_tradeoff@{K}")
    plt.xlabel("Конфигурация адаптивной lambda(n)")
    plt.title("Сравнение адаптивных конфигураций на валидации")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, y + 0.005, f"{y:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "adaptive_grid_search_plot_ru.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 5))
    labels = {"fixed": "Фиксированная (lambda = 0.5)", "adaptive": "Адаптивная lambda(n)"}
    markers = {"fixed": "o", "adaptive": "s"}
    for strategy in ["fixed", "adaptive"]:
        subset = results[results["strategy"] == strategy].sort_values("train_sparsity")
        plt.plot(
            subset["train_sparsity"], subset[f"f_tradeoff@{K}"],
            marker=markers[strategy], label=labels[strategy],
            linewidth=2.2, markersize=7,
        )
    plt.xlabel("Доля удалённых обучающих оценок")
    plt.ylabel(f"F_tradeoff@{K}")
    plt.title("F_tradeoff@10 в зависимости от разреженности")
    plt.xticks(
        [0.0, 0.3, 0.5, 0.7, 0.9],
        ["0%\n(100% данных)", "30%\n(70% данных)", "50%\n(50% данных)",
         "70%\n(30% данных)", "90%\n(10% данных)"],
        fontsize=8.5,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f_tradeoff_plot_ru.png", dpi=300)
    plt.close()


def main():
    ratings, users, movies = load_movielens()
    filtered = filter_users(ratings)
    plot_eda(filtered, OUTPUT_DIR)
    train_data, test_data, train_fit, val_data = temporal_split(filtered)

    all_items = movies["movie_id"].unique()
    similarity_matrix, movie_to_idx = load_or_build_similarity(movies)
    genre_similarity = make_genre_similarity(similarity_matrix, movie_to_idx)

    val_evaluator, test_evaluator = make_validation_and_test_evaluators(
        train_fit, val_data, train_data, test_data, all_items,
        RATING_THRESHOLD, N_NEG, RANDOM_SEED, MAX_USERS_EVAL,
    )

    best_factors, best_reg = tune_als(train_fit, val_evaluator)
    thresholds, lambdas, grid = tune_adaptive_lambda(
        train_fit, val_evaluator, best_factors, best_reg, genre_similarity
    )

    results = run_sparsity_experiment(
        train_data, test_data, all_items, test_evaluator,
        best_factors, best_reg, thresholds, lambdas, genre_similarity,
    )
    save_plots(grid, results)
    print("\nЭксперимент завершён.")
    print("Таблица результатов:", SPARSITY_RESULTS_PATH)
    print("Графики:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
