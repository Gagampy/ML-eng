from typing import Dict, List

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
import numpy as np

import mlflow
from mlflow import sklearn as mlflow_sklearn


lasso_param_grid = {"alpha": hp.choice("alpha", np.linspace(0.001, 1, num=1000))}
gb_param_grid = {
    "learning_rate": hp.choice("learning_rate", np.linspace(0.001, 1, num=1000)),
    "n_estimators": hp.choice("n_estimators", np.array((75, 100, 150))),
}


class HyperoptHPOptimizer(object):
    def __init__(
        self,
        x_trn: DataFrame,
        x_val: DataFrame,
        y_trn: DataFrame,
        y_val: DataFrame,
        hyperparameters_space: Dict[str, List],
        model_class,
        max_evals: int,
        send_to_mlflow: bool = True,
        experiment_name: str = "rtn-title-len-regr",
        tracking_uri: str = "http://rtn-mlflow-serv:5000",
    ):

        self.trials = Trials()
        self.model_class = model_class
        self.run_name = model_class().__class__.__name__
        self.max_evals = max_evals
        self.hyperparameters_space = hyperparameters_space
        self.x_trn = x_trn
        self.x_val = x_val
        self.y_trn = y_trn
        self.y_val = y_val
        self.send_to_mlflow = send_to_mlflow

        if self.send_to_mlflow:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    def fit_model_and_return_loss(self, hyperparameters):
        model = self.model_class(**hyperparameters)
        model = model.fit(self.x_trn, self.y_trn)

        mae_val = mean_absolute_error(self.y_val, model.predict(self.x_val))
        mae_trn = mean_absolute_error(self.y_trn, model.predict(self.x_trn))
        return model, {"mae_train": mae_trn, "mae_val": mae_val}

    def _get_loss_with_mlflow(self, hyperparameters):
        # MLflow will track and save hyperparameters, loss, and scores.
        with mlflow.start_run(run_name=self.run_name):
            print("Training with the following hyperparameters: ")
            print(hyperparameters)

            for param_name, value in hyperparameters.items():
                mlflow.log_param(param_name, value)
            model, metrics = self.fit_model_and_return_loss(hyperparameters)

            # Log the various losses and metrics (on train and validation)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow_sklearn.log_model(model, self.run_name)

            # Use the last validation loss from the history object to optimize
            return {"loss": metrics["mae_val"], "status": STATUS_OK}

    def _get_loss(self, hyperparameters):
        model, metrics = self.fit_model_and_return_loss(hyperparameters)
        return {"loss": metrics["mae_val"], "status": STATUS_OK}

    def optimize(self):
        """
        This is the optimization function that given a space of
        hyperparameters and a scoring function, finds the best hyperparameters.
        """
        # Use the fmin function from Hyperopt to find the best hyperparameters
        # Here we use the tree-parzen estimator method.
        if self.send_to_mlflow:
            training_func = self._get_loss_with_mlflow
        else:
            training_func = self._get_loss

        best = fmin(
            training_func,
            self.hyperparameters_space,
            algo=tpe.suggest,
            trials=self.trials,
            max_evals=self.max_evals,
        )
        return best
