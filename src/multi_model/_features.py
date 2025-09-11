import pandas as pd

from _structs import BOSSES, MINIGAMES, SKILLS


def get_ratio(
    df: pd.DataFrame,
    COLUMNS: list,
    total_column: str = "",
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
    df = get_ratio(df, SKILLS, total_column="skill_total")
    df = get_ratio(df, BOSSES, total_column="boss_total")
    df = get_ratio(df, MINIGAMES, total_column="minigame_total")
    new_feature_columns = [c for c in df.columns if c.endswith("_ratio")]
    new_feature_columns += ["skill_total", "boss_total", "minigame_total"]
    return df, new_feature_columns
