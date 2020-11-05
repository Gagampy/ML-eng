from typing import Dict, Tuple
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def join_datatables(datapath: Path, **kwargs) -> pd.DataFrame:
    """
    datapath: Path -- Path to the folder with splitted data.
    """
    dataframes = [pd.read_csv(filepath) for filepath in datapath.glob(pattern="*.csv")]
    return pd.concat(dataframes, axis=0).reset_index(drop=True)


def remove_outliers(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Removes outliers basing on quantiles. If `lower_q_value` or `upper_q_value` provided, they are used instead of
    calculated.

    X: pd.DataFrame
    """
    X_train_feature_quantiles_dict = get_feature_quantiles(X_train, lower_q, upper_q)

    X_train_filtered = remove_outliers_from_features(
        X_train, X_train_feature_quantiles_dict
    )
    X_val_filtered = remove_outliers_from_features(
        X_val, X_train_feature_quantiles_dict
    )
    X_test_filtered = remove_outliers_from_features(
        X_test, X_train_feature_quantiles_dict
    )

    return X_train_filtered, X_val_filtered, X_test_filtered


def get_feature_quantiles(
    X: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99, **kwargs
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates quantiles for every feature in X.
    """
    X_copy = X.copy()

    feature_quantiles_dict = {}
    for feature_col in X_copy.columns:
        q_low = X_copy[feature_col].quantile(lower_q)
        q_hi = X_copy[feature_col].quantile(upper_q)
        feature_quantiles_dict[feature_col] = (q_low, q_hi)
    return feature_quantiles_dict


def remove_outliers_from_features(
    X: pd.DataFrame, feature_quantiles_dict: Dict[str, Tuple[float, float]], **kwargs
) -> pd.DataFrame:
    """
    Using feature quantiles remove outliers from X.
    """
    X_copy = X.copy()

    for feature_col in X_copy.columns:
        if feature_col in feature_quantiles_dict:
            q_low = feature_quantiles_dict[feature_col][0]
            q_hi = feature_quantiles_dict[feature_col][1]
            X_copy = X_copy[(X_copy[feature_col] < q_hi) & (X_copy[feature_col] > q_low)]
    return X_copy


def split_data(
    X: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15, seed: int = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, test and validation.
    Test size is calculated according to train_ratio and valid_ratio.
    """
    valid_size = int(X.shape[0] * valid_ratio)

    X_train, X_val = train_test_split(X, train_size=train_ratio, random_state=seed)
    X_val, X_test = train_test_split(X_val, train_size=valid_size, random_state=seed)
    return X_train, X_val, X_test


def save_splitted_dataset(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    savefolder_path: Path,
    **kwargs
):
    savefolder_path.mkdir(exist_ok=True)

    X_train.to_csv(savefolder_path / 'train_filtered.csv', index=False)
    X_val.to_csv(savefolder_path / 'val_filtered.csv', index=False)
    X_test.to_csv(savefolder_path / 'test_filtered.csv', index=False)

