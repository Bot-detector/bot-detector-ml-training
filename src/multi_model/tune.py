import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow
from sklearn.metrics import classification_report
import uuid
import time
from mlflow.tracking import MlflowClient

##### Configuration #####
TRACKING_SERVER_URI = "http://localhost:5000"
EXPERIMENT_NAME = "2025-08-30_multi_classifier_559f"

DATA_FILE = "data/2025-08-30_hiscore_data.parquet.gzip"
TARGET_COLUMN = "player_label"
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
    # "league",
    # "bounty_hunter_hunter",
    # "bounty_hunter_rogue",
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
    # print(df.columns)

    # Ensure all feature columns are present
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    return df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # print(pd.DataFrame(df.player_label.value_counts()))
    mask = df["player_label_id"].isin([0, 89])
    df = df[~mask].copy()

    common_labels = (
        pd.DataFrame(df.player_label.value_counts())
        .query("count > 200")
        .index.to_list()
    )
    mask = df.player_label.isin(common_labels)
    df = df[mask].copy()
    # print(pd.DataFrame(df.player_label.value_counts()))
    return df


def get_ratio(
    df: pd.DataFrame,
    COLUMNS: list,
    total_column: str = None,
    column_suffix: str = "ratio",
) -> pd.DataFrame:
    _df = df.copy()
    TOTAL = df[COLUMNS].sum(axis=1)
    for column in COLUMNS:
        _df[f"{column}_{column_suffix}"] = df[column] / TOTAL
    if total_column:
        _df[total_column] = TOTAL
    return _df


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "total" in df.columns:
        df.drop(columns=["total"], inplace=True)
    df = get_ratio(df, SKILLS, total_column="skill_total")
    df = get_ratio(df, BOSSES, total_column="boss_total")
    df = get_ratio(df, MINIGAMES, total_column="minigame_total")
    new_feature_columns = [c for c in df.columns if c.endswith("_ratio")]
    new_feature_columns += ["skill_total", "boss_total", "minigame_total"]
    return df, new_feature_columns


def get_metrics(model: DecisionTreeClassifier, X_test, y_test) -> dict:
    # Predict and evaluate
    y_pred = model.predict(X_test)

    report_dict = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0,
    )
    # _ = {print("/t", {k: v}) for k, v in report_dict.items()}

    if not isinstance(report_dict, dict):
        return

    report_dict: dict[str, dict | str]
    metrics = {}
    for pred, v in report_dict.items():
        if not isinstance(v, dict):
            continue
        for _k, _v in v.items():
            # mlflow.log_metric(key=f"{pred}.{_k}", value=round(_v, 4))
            # print({f"{pred}.{_k}": round(_v, 4)})
            metrics.update({f"{pred}.{_k}": round(_v, 4)})
    return metrics


def train(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    params: dict,
    experiment_id,
    parent_run_id,
):
    with mlflow.start_run(
        nested=True,
        experiment_id=experiment_id,
        parent_run_id=parent_run_id,
        log_system_metrics=False,
    ) as run:
        print(f"Run ID: {run.info.run_id} - run_name={run.info.run_name} - child")
        print(f"Training with params: {params}")

        model = DecisionTreeClassifier(random_state=42)
        model.set_params(**params)
        model.fit(X=X_train, y=y_train)

        metrics = get_metrics(model=model, X_test=X_test, y_test=y_test)
        mlflow.log_metrics(metrics=metrics, run_id=run.info.run_id)
        mlflow.log_params(params=params, run_id=run.info.run_id)

        # idk what is wrong here if i leave this out it works
        mlflow.sklearn.log_model(sk_model=model, name=model_name, step=1)


def main():
    df = load_data(file_path=DATA_FILE, feature_columns=FEATURE_COLUMNS)
    df = data_cleaning(df)
    df, new_feature_columns = feature_engineering(df)

    X, y = df[FEATURE_COLUMNS + new_feature_columns], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    client = MlflowClient()

    ##### 1. Find best run #####
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment {EXPERIMENT_NAME} not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.`macro avg.f1-score` DESC"],
        max_results=1,
    )
    best_run = runs[0]
    print(
        f"Best run: {best_run.info.run_id} with f1={best_run.data.metrics.get('macro avg.f1-score')}"
    )

    ##### 2. Load best model #####
    best_model_uri = f"{best_run.info.artifact_uri}"
    print(best_model_uri)
    best_model = mlflow.sklearn.load_model(best_model_uri)

    ##### 3. Feature importance #####
    importances = best_model.feature_importances_
    feat_importances = sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
    )
    print("Top features:", feat_importances[:10])

    # Select features with nonzero importance
    selected_features = [f for f, imp in feat_importances if imp > 0]
    print(f"Selected {len(selected_features)} features")

    ##### 4. Retrain with selected features #####
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    ##### 5. Log new run #####
    today_iso = time.strftime("%Y-%m-%d")
    experiment_name = f"{today_iso}_{EXPERIMENT_NAME}_{str(uuid.uuid4())[:4]}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        tuned_model = DecisionTreeClassifier(random_state=42)
        tuned_model.fit(X_train_sel, y_train)

        metrics = get_metrics(tuned_model, X_test_sel, y_test)
        print("New model metrics:", metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_params({"selected_features": len(selected_features)})
        mlflow.sklearn.log_model(tuned_model, artifact_path="selected_model")


if __name__ == "__main__":
    main()
