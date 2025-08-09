import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import uuid

# --- Configuration ---
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "binary_classifier"

DATA_FILE = "data/2022-10-26_hiscore_data.parquet.gzip"
TARGET_COLUMN = "confirmed_ban"
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


def load_data(file_path: str, feature_columns: list[str]):
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Data loaded with {len(df)} samples and {len(df.columns)} columns.")

    # Ensure all feature columns are present
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return df


def log_metrics(model: DecisionTreeClassifier, X_test, y_test):
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    mlflow.log_metric("true_positives", tp)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_negatives", fn)
    print(f"\t- TP: {tp}, FP: {fp}")

    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    mlflow.log_metric("accuracy", accuracy)
    print(f"\t- Accuracy: {accuracy}")

    auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    mlflow.log_metric("auc", auc)
    print(f"\t- auc: {auc}")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    if not isinstance(report_dict, dict):
        return

    report_dict: dict[str, dict | str]
    for pred, cf_rep in report_dict.items():
        if not isinstance(cf_rep, dict):
            continue
        for k, v in cf_rep.items():
            mlflow.log_metric(key=f"{pred}-{k}", value=round(v, 4))


def train(X_train, y_train, X_test, y_test, model_name, params: dict, experiment_id):
    with mlflow.start_run(nested=True, experiment_id=experiment_id) as run:
        print(f"Run ID: {run.info.run_id} - run_name={run.info.run_name}")
        print(f"Training with params: {params}")
        model = DecisionTreeClassifier(random_state=42)
        model.set_params(**params)
        model.fit(X=X_train, y=y_train)
        log_metrics(model=model, X_test=X_test, y_test=y_test)
        mlflow.sklearn.log_model(sk_model=model, name=model_name)
        mlflow.log_params(params)


def main():
    df = load_data(file_path=DATA_FILE, feature_columns=FEATURE_COLUMNS)
    df = feature_engineering(df)

    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = ParameterGrid(
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 30, 50, 100, 200, 500],
            "min_samples_leaf": [5, 10, 20],
        }
    )

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    experiment_id = mlflow.create_experiment(
        f"{EXPERIMENT_NAME}-{str(uuid.uuid4())[:4]}"
    )
    with mlflow.start_run(experiment_id=experiment_id) as parent_run:
        print(
            f"Run ID: {parent_run.info.run_id} - run_name={parent_run.info.run_name} - parent"
        )
        for i, params in enumerate(param_grid):
            train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_name=f"{EXPERIMENT_NAME}_{i}",
                params=params,
                experiment_id=experiment_id,
            )
            break

    #
    query = f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
    results = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["metrics.`weighted avg-f1-score` DESC"],
        max_results=5,
    )

    best_run_id = results.iloc[0]["run_id"]

    model_uri = f"runs:/{best_run_id}/model"
    print(model_uri)
    loaded_model: DecisionTreeClassifier = mlflow.sklearn.load_model(model_uri)
    # Get feature importances
    importances = loaded_model.feature_importances_

    # Select features with importance greater than a threshold
    threshold = 0.1  # Adjust as needed
    selected_features = X.columns[importances > threshold]
    print(selected_features)

    # Use only the selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    loaded_model.fit(X_train_selected, y_train)


if __name__ == "__main__":
    main()
