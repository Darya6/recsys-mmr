"""Общие параметры проекта."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ZIP_PATH = DATA_DIR / "Movielens.zip"

SIM_PATH = CACHE_DIR / "similarity_matrix_genres.npy"
MOV2IDX_PATH = CACHE_DIR / "movie_to_idx.pkl"
CAND_VAL_PATH = CACHE_DIR / "eval_candidates_val.pkl"
CAND_TEST_PATH = CACHE_DIR / "eval_candidates_test.pkl"

TRAIN_DATA_PATH = CACHE_DIR / "train_data.csv"
TEST_DATA_PATH = CACHE_DIR / "test_data.csv"
TRAIN_FIT_PATH = CACHE_DIR / "train_fit.csv"
VAL_DATA_PATH = CACHE_DIR / "val_data.csv"

RES_TEST_PATH = OUTPUT_DIR / "results_test.csv"
SPARSITY_RESULTS_PATH = OUTPUT_DIR / "sparsity_results.csv"
ADAPTIVE_GRID_PATH = OUTPUT_DIR / "adaptive_grid_search_results.csv"
ALS_GRID_PATH = OUTPUT_DIR / "als_grid_search_results.csv"

TOP_N = 300
RATING_THRESHOLD = 4.0
MIN_RATINGS_PER_USER = 5
K = 10
N_NEG = 200
MAX_USERS_EVAL = 2000

RANDOM_SEED = 42
ALS_ITERATIONS = 30
ALS_ALPHA = 15
KEEP_RATIOS = [1.0, 0.7, 0.5, 0.3, 0.1]

for directory in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
