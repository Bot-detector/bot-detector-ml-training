import pandas as pd
from sklearn.model_selection import train_test_split


def get_ratio(
    df: pd.DataFrame,
    COLUMNS: list,
    total_column: str | None = None,
    column_suffix: str = "ratio",
) -> pd.DataFrame:
    """
    Calculate the ratio of each column in the given DataFrame to the sum of all columns in COLUMNS.

    Args:
        df: The DataFrame to calculate ratios for.
        COLUMNS: A list of columns to calculate ratios for.
        total_column: The name of the column to use for the total.
        column_suffix: The suffix to use for the ratio columns.

    Returns:
        A new DataFrame with the same index as df where each column in COLUMNS has been replaced
        with the corresponding ratio. Additionally, the function calculates the total of all columns
        in COLUMNS and adds this as a new column in the returned DataFrame with the name specified
        by total_column.
    """
    _df = pd.DataFrame(index=df.index)
    TOTAL = df[COLUMNS].sum(axis=1)
    if total_column:
        _df[total_column] = TOTAL

    for column in COLUMNS:
        _df[f"{column}_{column_suffix}"] = df[column] / TOTAL
    return _df


def load_and_split_data(
    file_path,
    target_column,
    feature_columns,
    test_size=0.2,
    random_state=42,
):
    """
    Loads data from a parquet file, splits it into features (X) and target (y),
    and then into training and testing sets.

    Args:
        file_path (str): The path to the parquet data file.
        target_column (str): The name of the column to be used as the target variable.
        feature_columns (list): A list of column names to be used as features.
        test_size (float): The proportion of the dataset to allocate to the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Data loaded with {len(df)} samples and {len(df.columns)} columns.")

    # Ensure all feature columns are present
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    # Ensure target is boolean/integer (0 or 1)
    df[target_column] = df[target_column].astype(int)

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    return X_train, X_test, y_train, y_test
