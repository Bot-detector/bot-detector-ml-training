import os
import pickle
import tempfile
import time
import uuid

import mlflow
import optuna
from lightgbm import LGBMClassifier
from pydantic import ValidationError
from sklearn.model_selection import cross_validate, StratifiedKFold

from _features import feature_engineering
from _structs import FEATURE_COLUMNS, InputData, OutputData
from _utils import data_cleaning, load_data, server_running
from _wrapper import LGBMWrapper

# --- Configuration ---
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "multi_classifier"

DATA_FILE = "../../data/2025-08-30_hiscore_data.parquet.gzip"
TARGET_COLUMN = "player_label"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

PARAM_GUESS = {
    "n_estimators": 1134,
    "learning_rate": 0.003928837419116695,
    "num_leaves": 52,
    "max_depth": 7,
    "min_child_samples": 31,
    "min_child_weight": 0.007160950259502004,
    "subsample": 0.8643540359280603,
    "colsample_bytree": 0.6669806043037383,
    "reg_alpha": 0.012258874962110665,
    "reg_lambda": 0.006538683324341339,
    "min_split_gain": 0.15879816199038882,
}

# Multiclass-safe scorers
SCORING = [
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]


def log_run(output, params):
    mlflow.log_params(params)
    for metric, values in output.items():
        metric = metric.removeprefix("test_")
        mlflow.log_metric(f"mean_{metric}", values.mean())
        mlflow.log_metric(f"std_{metric}", values.std())


def objective(trial, X, y):
    params = {
        "objective": "multiclass",
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
    out = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=1)
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        log_run(out, params)
    # optimize for weighted F1 by default
    return out["test_f1_weighted"].mean()


def main():

    if server_running(TRACKING_SERVER_URI):
        mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    else:
        mlflow.set_tracking_uri("file:./mlruns")

    today_iso = time.strftime("%Y-%m-%dT%H:%M")
    experiment_name = f"{today_iso}_{EXPERIMENT_NAME}_{uuid.uuid4().hex[:4]}"
    mlflow.set_experiment(experiment_name)

    df = load_data(file_path=DATA_FILE, feature_columns=FEATURE_COLUMNS)
    df = data_cleaning(df)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    # data validation
    output_fields = set(OutputData.model_fields.keys())
    target_fields = set(y.unique())

    assert output_fields == target_fields, (
        f"Mismatch between OutputData fields and target columns in `y`.\n"
        f"Fields missing in `output data`: {output_fields - target_fields}\n"
        f"Extra fields in `target column`: {target_fields - output_fields}"
    )

    # data validation
    try:
        _ = [InputData(**x) for x in X[:5].to_dict(orient="records")]
    except ValidationError as e:
        print(e.json())
        raise e

    X, _ = feature_engineering(X)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=experiment_name,
    )

    study.enqueue_trial(PARAM_GUESS)

    study.optimize(lambda t: objective(t, X, y), n_trials=10)

    # cross-validated metrics with the final params
    best_params = study.best_trial.params

    cv_model = LGBMClassifier(**best_params)
    out = cross_validate(cv_model, X, y, cv=CV, scoring=SCORING, n_jobs=1)

    # fit on full data for the artifact
    refit_model = LGBMClassifier(**best_params)
    refit_model.fit(X, y)

    with mlflow.start_run(run_name="refit_best"):
        log_run(out, best_params)

        model_pkl = os.path.join(tempfile.mkdtemp(), "model.pkl")
        with open(model_pkl, "wb") as f:
            pickle.dump(refit_model, f)

        info = mlflow.pyfunc.log_model(
            python_model=LGBMWrapper.from_lgbm(refit_model),
            name="refit_model",
            artifacts={"model": model_pkl},
            code_paths=["_wrapper.py", "_features.py", "_structs.py", "_utils.py"],
        )

        mlflow.register_model(model_uri=info.model_uri, name=EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
