import argparse
from logging import warning
from typing import Union, Dict

from pandas import DataFrame
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from utils import get_train_and_test
import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', default='Lasso')
    parser.add_argument('-uri', default='http://rtn-mlflow-serv:5000')
    parser.add_argument('-en', default='rtn-title-len-regr')
    parser.add_argument('-m', default='mae_val')
    parser.add_argument('-tr_data', default='./split/train.csv')
    parser.add_argument('-tt_data', default='./split/test.csv')

    parser = parser.parse_args()
    return parser


def get_run_dataframe(experiment_name: str, uri: str) -> Union[None, DataFrame]:
    """ For specified experiment_name searches an experiment on MLFlow server and returns a run with it's ID. """
    experiment_id = None
    for experiment in mlflow.tracking.MlflowClient(tracking_uri=uri).list_experiments():
        if experiment.name == experiment_name:
            experiment_id = experiment.experiment_id

    if experiment_id is None:
        warning(f'Experiment with specified name: {experiment_name} not found. Aborting.')
        return None

    return mlflow.search_runs(experiment_id)


def get_params_from_run_df(run_df: DataFrame, model_name: str, metric: str) -> Dict[str, Union[str, float]]:
    """For specified model_name selects parameters corresponding to the best metric score from run_df. """
    run_df = run_df.copy().loc[run_df['tags.mlflow.runName'] == model_name]
    parameter_cols = [col for col in run_df.columns if col.startswith('params.')]

    params = run_df.iloc[run_df[f'metrics.{metric}'].idxmin()][parameter_cols].to_dict()
    params = {param.split('.')[-1]: value for param, value in params.items()}
    return params


if __name__ == '__main__':

    parser = parse_args()
    x_train, y_train, x_test, y_test = get_train_and_test(parser.tr_data, parser.tt_data)

    run_df = get_run_dataframe(parser.en, parser.uri)
    if run_df is not None:
        params = get_params_from_run_df(run_df, model_name=parser.mn, metric=parser.m)

        model = Lasso(**params)
        model.fit(X=x_train, y=y_train)
        prediction = model.predict(X=x_test)
        print("MAE on test:", mean_absolute_error(y_test, prediction))