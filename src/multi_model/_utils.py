import pandas as pd
import socket
import urllib.parse


def server_running(uri="http://localhost:5000", timeout=1):
    parsed = urllib.parse.urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


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
    print(pd.DataFrame(df.player_label.value_counts()))
    mask = df["player_label_id"].isin([0, 89])
    df = df[~mask].copy()

    common_labels = (
        pd.DataFrame(df.player_label.value_counts())
        .query("count > 200")
        .index.to_list()
    )
    mask = df.player_label.isin(common_labels)
    df = df[mask].copy()
    print(pd.DataFrame(df.player_label.value_counts()))
    if "total" in df.columns:
        df.drop(columns=["total"], inplace=True)
    return df
