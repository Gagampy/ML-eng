from typing import Tuple
from pandas import DataFrame, Series, read_csv


def get_train_and_test(train_path: str, test_path: str) -> Tuple[DataFrame, Series, DataFrame, Series]:
    """ Reads train and test datasets. """
    train_dataset = get_x_and_y(train_path)
    test_dataset = get_x_and_y(test_path)
    return train_dataset[0], train_dataset[1], test_dataset[0], test_dataset[1]


def get_x_and_y(dataset_path: str) -> Tuple[DataFrame, Series]:
    """ Reads a dataset and split it to X and Y. """
    dataset = read_csv(dataset_path)
    return dataset.iloc[:, :-1], dataset.iloc[:, -1]