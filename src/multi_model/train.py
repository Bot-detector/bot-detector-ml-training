import os
import pickle
import time
import uuid

import mlflow
from pandas import DataFrame
from pydantic import ValidationError
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid, train_test_split

from _features import feature_engineering
from _structs import FEATURE_COLUMNS
from _utils import data_cleaning, load_data
from _wrapper import DecisionTreeWrapper, InputData, OutputData

# --- Configuration ---
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "multi_classifier"

DATA_FILE = "../../data/2025-08-30_hiscore_data.parquet.gzip"
TARGET_COLUMN = "player_label"


def get_metrics(
    model: DecisionTreeWrapper, X_test: DataFrame, y_test: DataFrame
) -> dict | None:
    y_pred = model.model.predict(X_test)

    report_dict = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0,
    )

    if not isinstance(report_dict, dict):
        return None

    metrics = {}
    for pred, v in report_dict.items():
        if not isinstance(v, dict):
            continue
        for _k, _v in v.items():
            metrics.update({f"{pred}.{_k}": round(_v, 4)})
    return metrics


def main():
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # data validation
    try:
        _ = [InputData(**x) for x in X_train[:5].to_dict(orient="records")]
    except ValidationError as e:
        print(e.json())
        raise e

    param_grid = ParameterGrid(
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 30, 50, 100, 200, 500],
            "min_samples_leaf": [5, 10, 20],
        }
    )

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    today_iso = time.strftime("%Y-%m-%dT%H:%M")
    uuid_str = str(uuid.uuid4())[:4]
    experiment_name = f"{today_iso}_{EXPERIMENT_NAME}_{uuid_str}"
    experiment_id = mlflow.create_experiment(name=experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as parent_run:
        _print = f"parent: {parent_run.info.run_id} - name={parent_run.info.run_name}"
        print(_print)
        for i, params in enumerate(param_grid):
            with mlflow.start_run(
                nested=True,
                experiment_id=experiment_id,
                parent_run_id=parent_run.info.run_id,
                log_system_metrics=False,
            ) as run:
                _print = f"child: {run.info.run_id} - name={run.info.run_name}"
                print(_print)
                print(f"Training with params: {params}")

                model = DecisionTreeWrapper(
                    params=params, feature_fn=feature_engineering
                )
                model.fit(X=X_train, y=y_train)

                metrics = get_metrics(model=model, X_test=X_test, y_test=y_test)
                assert isinstance(metrics, dict)

                mlflow.log_metrics(metrics=metrics, run_id=run.info.run_id)
                mlflow.log_params(params=params, run_id=run.info.run_id)

                model_path = f"{run.info.run_id}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model.model, f)

                # idk what is wrong here if i leave this out it works
                code_paths = ["_wrapper.py", "_features.py", "_structs.py", "_utils.py"]
                mlflow.pyfunc.log_model(
                    python_model=model,
                    name=f"{EXPERIMENT_NAME}_{i}",
                    artifacts={"model": model_path},
                    code_paths=code_paths,
                )
                os.remove(model_path)


if __name__ == "__main__":
    main()
