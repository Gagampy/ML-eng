import numpy as np

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import mlflow
from mlflow import lightgbm as mlflow_lgbm, sklearn as mlflow_sklearn


elastic_net_param_grid = {
    'l1_ratio': hp.choice('learning_rate', np.linspace(0.001, 1, num=1000)),
    'alpha': hp.choice('alpha', np.linspace(0.001, 1, num=1000))
}


class HyperoptHPOptimizer(object):

    def __init__(self, x_data, y_data, hyperparameters_space, model_class, max_evals, random_state=42,
                 experiment_name='rtn_title_len_prediction', tracking_uri: str = 'http://localhost:5001'):
        self.trials = Trials()
        self.model_class = model_class
        self.max_evals = max_evals
        self.hyperparameters_space = hyperparameters_space
        self.x_trn, self.x_val, self.y_trn, self.y_val = train_test_split(x_data,
                                                                          y_data,
                                                                          train_size=0.7,
                                                                          random_state=random_state)

        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(tracking_uri)

    def fit_model_and_return_loss(self, hyperparameters):
        model = self.model_class(**hyperparameters)
        model = model.fit(self.x_trn, self.y_trn)

        mae_val = mean_absolute_error(self.y_val, model.predict(self.x_val))
        mae_trn = mean_absolute_error(self.y_trn, model.predict(self.x_trn))
        return model, {'mae_train': mae_trn, 'mae_val': mae_val}

    def _get_loss_with_mlflow(self, hyperparameters):
        # MLflow will track and save hyperparameters, loss, and scores.
        with mlflow.start_run():
            print("Training with the following hyperparameters: ")
            print(hyperparameters)

            for param_name, value in hyperparameters.items():
                mlflow.log_param(param_name, value)
            model, metrics = self.fit_model_and_return_loss(hyperparameters)

            # Log the various losses and metrics (on train and validation)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # elif isinstance(model, ElasticNet):
            #     mlflow_sklearn.log_model(model, 'model', registered_model_name='ElasticNet')

            # Use the last validation loss from the history object to optimize
            return {'loss': metrics['mae_val'], 'status': STATUS_OK}

    def optimize(self):
        """
        This is the optimization function that given a space of
        hyperparameters and a scoring function, finds the best hyperparameters.
        """
        # Use the fmin function from Hyperopt to find the best hyperparameters
        # Here we use the tree-parzen estimator method.
        best = fmin(self._get_loss_with_mlflow, self.hyperparameters_space, algo=tpe.suggest,
                    trials=self.trials, max_evals=self.max_evals)
        return best




