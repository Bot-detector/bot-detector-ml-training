import mlflow
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

# Import our reusable utility functions
from common.data_utils import load_and_split_data
from common.train_utils import evaluate_and_log_metrics, log_model_artifact

# --- Configuration ---
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "binary_classifier"

DATA_FILE = "data/2022-10-26_hiscore_data.parquet.gzip"
TARGET_COLUMN = "confirmed_ban"
FEATURE_COLUMNS = [
    "total",
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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def run_standalone_training():
    """
    Orchestrates training by managing the MLflow state directly within the script.
    """
    # 1. The script now takes full control of the MLflow state
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split_data(
        file_path=DATA_FILE,
        target_column=TARGET_COLUMN,
        feature_columns=FEATURE_COLUMNS,
    )

    param_grid = ParameterGrid(
        {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [10, 30, 50, 100, 200, 500],
            "classifier__min_samples_leaf": [5, 10, 20],
        }
    )

    # 2. The script now creates its own parent run
    with mlflow.start_run(run_name="DecisionTree_Hyperparameter_Search") as parent_run:
        print(f"Started parent run: {parent_run.info.run_id}")
        mlflow.log_param("data_file", DATA_FILE)

        # Loop through each hyperparameter combination
        for i, params in enumerate(param_grid):
            with mlflow.start_run(run_name="trial", nested=True) as child_run:
                print(f"  - Starting trial (child run): {child_run.info.run_id}")
                mlflow.log_params(params)

                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("classifier", DecisionTreeClassifier(random_state=42)),
                    ]
                )
                pipeline.set_params(**params)
                pipeline.fit(X_train, y_train)

                # Evaluate and log all metrics using our utility function
                evaluate_and_log_metrics(pipeline, X_test, y_test)
                log_model_artifact(
                    model=pipeline,
                    model_name=f"{EXPERIMENT_NAME}_{i}",
                    input_example=X_train.head(),
                )


if __name__ == "__main__":
    run_standalone_training()
