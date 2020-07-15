import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score


SEED = 12
DATAPATH = '2-RTN/data/feature_eng/dataset_fe.csv'
METRICS_SAVEPATH = '2-RTN/metrics/regression.metric'


def run_kfold(model, splitter, X, y):
    r2 = {'train': [], 'test': []}
    mae = {'train': [], 'test': []}

    for train_idx, test_idx in splitter.split(X):
        model.fit(X[train_idx], y[train_idx])

        test_predicted = model.predict(X[test_idx])
        train_predicted = model.predict(X[train_idx])

        r2['test'].append(r2_score(y[test_idx], test_predicted))
        r2['train'].append(r2_score(y[train_idx], train_predicted))

        mae['test'].append(mean_absolute_error(y[test_idx], test_predicted))
        mae['train'].append(mean_absolute_error(y[train_idx], train_predicted))
    return r2, mae


def main():

    dataset = pd.read_csv(DATAPATH, lines=True)

    y = dataset['title_len'].values
    X = dataset.drop('title_len', axis=1)

    X = RobustScaler().fit_transform(X)
    kf = KFold(shuffle=True, random_state=SEED)
    ridge = Ridge()

    r2_metrics, mae_metrics = run_kfold(ridge, kf, X, y)

    metrics_message = """
    MAE:

    -- Train: {:<10} Test: {}
    =====================================

    R2 score:

    -- Train: {:<10} Test: {}
    =====================================
    """.format(mae_metrics['train'], mae_metrics['test'],
               r2_metrics['train'], r2_metrics['test']).strip()

    with open(METRICS_SAVEPATH, 'w') as f:
        f.writelines(metrics_message)


if __name__ == '__main__':
    main()
