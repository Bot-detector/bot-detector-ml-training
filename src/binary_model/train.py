import uuid
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import mlflow
import mlflow.sklearn
import optuna
from mlflow.tracking import MlflowClient

# --- Configuration ---
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "binary_classifier"

DATA_FILE = "data/2022-10-26_hiscore_data.parquet.gzip"
TARGET_COLUMN = "confirmed_ban"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

PARAM_GUESS = {
    'colsample_bytree': 0.9651651365751598,
    'learning_rate': 0.05310861766637499,
    'max_depth': 23,
    'min_child_samples': 84,
    'min_child_weight': 0.004787755397014158,
    'min_split_gain': 0.022266875377142784,
    'n_estimators': 1445,
    'num_leaves': 255,
    'reg_alpha': 1.7177943743459194,
    'reg_lambda': 1.1992203482868355e-05,
    'subsample': 0.7224340426136543
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


def load_data(file_path: str, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df


def objective(trial, X, y, experiment_id):

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

    # child run; inherit parent's experiment implicitly
    with mlflow.start_run(nested=True, experiment_id=experiment_id, run_name=f"trial_{trial.number}"):
        model = LGBMClassifier(**params)
        out = cross_validate(model, X, y, cv=CV, scoring=SCORING)

        mlflow.log_params(params)
        for metric, values in out.items():  #
            mlflow.log_metric(f"mean_{metric}", values.mean())
            mlflow.log_metric(f"std_{metric}", values.std())

        mean_auc = out["test_roc_auc"].mean()
        std_auc = out["test_roc_auc"].std()
        mlflow.log_metric("auc_mean", mean_auc)
        mlflow.log_metric("auc_std", std_auc)

    return mean_auc


def main():
    df = load_data(DATA_FILE, FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    experiment_id = mlflow.create_experiment(
        f"{EXPERIMENT_NAME}-{str(uuid.uuid4())[:4]}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{experiment_id}-study",
    )

    study.enqueue_trial(PARAM_GUESS)

    with mlflow.start_run(experiment_id=experiment_id, run_name="optuna_parent"):
        study.optimize(lambda t: objective(t, X, y, experiment_id), n_trials=10)

        best_params = study.best_trial.params
        best_value = float(study.best_value)
        mlflow.log_metric("best_cv_auc", best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # refit on full data as another child
        with mlflow.start_run(nested=True, run_name="refit_best", experiment_id=experiment_id):
            model = LGBMClassifier(random_state=42, **best_params)
            model.fit(X, y)
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(model, artifact_path="refit_model")


if __name__ == "__main__":
    main()
