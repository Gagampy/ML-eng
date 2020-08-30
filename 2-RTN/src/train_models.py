import argparse
from typing import Tuple

from models import HyperoptHPOptimizer, lasso_param_grid, gb_param_grid

from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from pandas import DataFrame, Series, read_csv

TRAINABLE_CLASSES = {
    Lasso: lasso_param_grid,
    GradientBoostingRegressor: gb_param_grid
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default='dataset_fc.csv')
    parser.add_argument('-me', default=20)
    parser.add_argument('-uri', default='http://rtn-mlflow-serv:5000')
    parser.add_argument('-en', default='rtn-title-len-regr')
    parser = parser.parse_args()
    return parser


def get_x_and_y(dataset_path: str) -> Tuple[DataFrame, Series]:
    dataset = read_csv(dataset_path)
    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


if __name__ == '__main__':

    parser = parse_args()
    x, y = get_x_and_y(parser.fp)

    for model_class, param_grid in TRAINABLE_CLASSES.items():
        hyper_optimizer = HyperoptHPOptimizer(x,
                                              y,
                                              param_grid,
                                              model_class=model_class,
                                              max_evals=parser.me,
                                              tracking_uri=parser.uri,
                                              experiment_name=parser.en
                                              )
        hyper_optimizer.optimize()

