from pathlib import Path
from typing import Tuple, Dict, Union, List
import json
import pandas as pd
from rtn.preprocessing_utils import (
    join_datatables,
    remove_outliers,
    split_data,
    save_splitted_dataset,
    get_feature_quantiles,
    remove_outliers_from_features,
)
from rtn.client.models import HyperoptHPOptimizer, lasso_param_grid
from sklearn.linear_model import Lasso


def join_datatables_task(datapath: Path = None, **kwargs) -> pd.DataFrame:
    """
    Triggers `join_datatables`; loading train, val and test and joining them.
    """
    if datapath is None:
        X_train = kwargs["ti"].xcom_pull(task_ids="removing_outliers_train")
        X_val = kwargs["ti"].xcom_pull(task_ids="removing_outliers_valid")
        X_test = kwargs["ti"].xcom_pull(task_ids="removing_outliers_test")
        return pd.concat((X_train, X_val, X_test), axis=0).reset_index(drop=True)
    return join_datatables(datapath)


def split_data_task(
    train_ratio: float = 0.7, valid_ratio: float = 0.15, seed: int = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Triggers `split_data`; splitting X to train, val and test.
    """
    X = kwargs["ti"].xcom_pull(task_ids="joining_data")
    return split_data(X, train_ratio, valid_ratio, seed)


def remove_outliers_dataset_task(
    lower_q: float = 0.01, upper_q: float = 0.99, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Triggers `remove_outliers`; removing outliers from features of X_train, X_val and X_test.
    """
    X_train, X_val, X_test = kwargs["ti"].xcom_pull(task_ids="splitting_data")
    return remove_outliers(X_train, X_val, X_test, lower_q, upper_q)


def remove_outliers_from_train_task(**kwargs) -> pd.DataFrame:
    """
    Triggers `remove_outliers_from_features`; removing outliers from features of X_train only.
    """
    X_train_feature_quantiles = kwargs["ti"].xcom_pull(task_ids="calculating_feature_quantiles")
    X_train, _, _ = kwargs["ti"].xcom_pull(task_ids="splitting_data")
    return remove_outliers_from_features(X_train, X_train_feature_quantiles)


def remove_outliers_from_valid_task(**kwargs) -> pd.DataFrame:
    """
    Triggers `remove_outliers_from_features`; removing outliers from features of X_val only.
    """
    X_train_feature_quantiles = kwargs["ti"].xcom_pull(task_ids="calculating_feature_quantiles")
    _, X_val, _ = kwargs["ti"].xcom_pull(task_ids="splitting_data")
    return remove_outliers_from_features(X_val, X_train_feature_quantiles)


def remove_outliers_from_test_task(**kwargs) -> pd.DataFrame:
    """
    Triggers `remove_outliers_from_features`; removing outliers from features of X_test only.
    """
    X_train_feature_quantiles = kwargs["ti"].xcom_pull(task_ids="calculating_feature_quantiles")
    _, _, X_test = kwargs["ti"].xcom_pull(task_ids="splitting_data")
    return remove_outliers_from_features(X_test, X_train_feature_quantiles)


def get_feature_quantiles_task(lower_q: float = 0.01, upper_q: float = 0.99, **kwargs
) -> Dict[str, Tuple[float, float]]:
    """
    Triggers `get_feature_quantiles`; calculates quantiles for all features in X.
    """
    X_train, _, _ = kwargs["ti"].xcom_pull(task_ids="splitting_data")
    return get_feature_quantiles(X_train, lower_q, upper_q)


def save_splitted_dataset_task(savefolder_path: Path, source_task_id: Union[str, List[str]], **kwargs):
    """
    Triggers `save_splitted_dataset`; saving X_train, X_val and X_test.
    """
    if source_task_id == 'removing_outliers':
        X_train, X_val, X_test = kwargs["ti"].xcom_pull(task_ids=source_task_id)

    elif source_task_id == ['removing_outliers_train', 'removing_outliers_valid', 'removing_outliers_test']:
        X_train = kwargs["ti"].xcom_pull(task_ids='removing_outliers_train')
        X_val = kwargs["ti"].xcom_pull(task_ids='removing_outliers_valid')
        X_test = kwargs["ti"].xcom_pull(task_ids='removing_outliers_test')

    return save_splitted_dataset(X_train, X_val, X_test, savefolder_path)


def save_train_feature_quantiles(savefolder_path: Path, **kwargs):
    """
    Saves calculated train feature statistics as JSON.
    """
    feature_quantiles = kwargs["ti"].xcom_pull(task_ids="calculating_feature_quantiles")
    with open(savefolder_path/'train_feature_quantiles.json', 'w') as fs:
        json.dump(feature_quantiles, fs)


def train_model_and_get_predictions(savefolder_path: Path, **kwargs):
    """
    Runs HyperOpt on model's grid, fits with best params, gets MAE score on test
    and saves model instance with name 'lasso_model_test_mae_{test_score}.pkl'.
    """
    savefolder_path.mkdir(exist_ok=True)

    X_train = kwargs["ti"].xcom_pull(task_ids='removing_outliers_train')
    X_val = kwargs["ti"].xcom_pull(task_ids='removing_outliers_valid')
    X_test = kwargs["ti"].xcom_pull(task_ids='removing_outliers_test')

    y_train = X_train["title_len"]
    y_val = X_val["title_len"]
    y_test = X_test["title_len"]

    X_train.drop(columns="title_len", inplace=True)
    X_val.drop(columns="title_len", inplace=True)
    X_test.drop(columns="title_len", inplace=True)

    hyper_optimizer = HyperoptHPOptimizer(
        X_train,
        X_val,
        y_train,
        y_val,
        lasso_param_grid,
        model_class=Lasso,
        max_evals=20,
        tracking_uri="http://mlflow-server:5000",
        send_to_mlflow=True,
    )

    _ = hyper_optimizer.optimize()
