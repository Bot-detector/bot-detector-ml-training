import socket
import urllib.parse
import uuid

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.model_selection import cross_validate, StratifiedKFold

# --- Configuration ---
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "binary_classifier"

DATA_FILE = "../../data/2022-10-26_hiscore_data.parquet.gzip"
TARGET_COLUMN = "confirmed_ban"


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


PARAM_GUESS = {
    "colsample_bytree": 0.9651651365751598,
    "learning_rate": 0.05310861766637499,
    "max_depth": 23,
    "min_child_samples": 84,
    "min_child_weight": 0.004787755397014158,
    "min_split_gain": 0.022266875377142784,
    "n_estimators": 1445,
    "num_leaves": 255,
    "reg_alpha": 1.7177943743459194,
    "reg_lambda": 1.1992203482868355e-05,
    "subsample": 0.7224340426136543,
}

SCORING = [
    "accuracy",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
]

SKILLS = [
    "attack",
    "defence",
    "strength",
    "hitpoints",
    "ranged",
    "prayer",
    "magic",
    "cooking",
    "woodcutting",
    "fletching",
    "fishing",
    "firemaking",
    "crafting",
    "smithing",
    "mining",
    "herblore",
    "agility",
    "thieving",
    "slayer",
    "farming",
    "runecraft",
    "hunter",
    "construction",
]

MINIGAMES = [
    "league",
    "bounty_hunter_hunter",
    "bounty_hunter_rogue",
    "lms_rank",
    "soul_wars_zeal",
    "cs_all",
    "cs_beginner",
    "cs_easy",
    "cs_medium",
    "cs_hard",
    "cs_elite",
    "cs_master",
]

BOSSES = [
    "abyssal_sire",
    "alchemical_hydra",
    "barrows_chests",
    "bryophyta",
    "callisto",
    "cerberus",
    "chambers_of_xeric",
    "chambers_of_xeric_challenge_mode",
    "chaos_elemental",
    "chaos_fanatic",
    "commander_zilyana",
    "corporeal_beast",
    "crazy_archaeologist",
    "dagannoth_prime",
    "dagannoth_rex",
    "dagannoth_supreme",
    "deranged_archaeologist",
    "general_graardor",
    "giant_mole",
    "grotesque_guardians",
    "hespori",
    "kalphite_queen",
    "king_black_dragon",
    "kraken",
    "kreearra",
    "kril_tsutsaroth",
    "mimic",
    "nex",
    "nightmare",
    "phosanis_nightmare",
    "obor",
    "sarachnis",
    "scorpia",
    "skotizo",
    "tempoross",
    "the_gauntlet",
    "the_corrupted_gauntlet",
    "theatre_of_blood",
    "theatre_of_blood_hard",
    "thermonuclear_smoke_devil",
    "tombs_of_amascut",
    "tombs_of_amascut_expert",
    "tzkal_zuk",
    "tztok_jad",
    "venenatis",
    "vetion",
    "vorkath",
    "wintertodt",
    "zalcano",
    "zulrah",
]

FEATURE_COLUMNS = SKILLS + MINIGAMES + BOSSES


def server_running(uri="http://localhost:5000", timeout=1):
    parsed = urllib.parse.urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def load_data(file_path: str, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df


def log_run(output, params):
    mlflow.log_params(params)
    for metric, values in output.items():
        metric = metric.removeprefix("test_")

        mlflow.log_metric(f"mean_{metric}", values.mean())
        mlflow.log_metric(f"std_{metric}", values.std())


def objective(trial, X, y):

    params = {
        "objective": "binary",
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 24),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    out = cross_validate(
        model, X, y, cv=CV, scoring=SCORING, n_jobs=1
    )  # n_jobs is 1 because LGBM uses all cores
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        log_run(out, params=params)

    return out["test_roc_auc"].mean()


def main():
    if server_running(TRACKING_SERVER_URI):
        mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    else:
        mlflow.set_tracking_uri("file:./mlruns")

    experiment_id = f"{EXPERIMENT_NAME}_{uuid.uuid4().hex[:4]}"
    mlflow.set_experiment(experiment_id)

    df = load_data(DATA_FILE, FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=experiment_id,
    )

    study.enqueue_trial(PARAM_GUESS)

    study.optimize(
        lambda t: objective(t, X, y),
        n_trials=10,
    )

    # cross-validated metrics with the final params
    best_params = study.best_trial.params
    cv_model = LGBMClassifier(random_state=42, **best_params)
    out = cross_validate(cv_model, X, y, cv=CV, scoring=SCORING, n_jobs=1)

    # fit on full data for the artifact
    refit_model = LGBMClassifier(random_state=42, **best_params)
    refit_model.fit(X, y)

    with mlflow.start_run(run_name="refit_best"):
        log_run(out, best_params)
        sig = infer_signature(X.head(100), refit_model.predict_proba(X.head(100)))
        mlflow.sklearn.log_model(
            refit_model,
            name="refit_model",
            signature=sig,
            input_example=X.head(5),
        )


if __name__ == "__main__":
    main()
