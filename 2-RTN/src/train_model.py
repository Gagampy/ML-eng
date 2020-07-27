import argparse

from ..src.models import HyperoptHPOptimizer, elastic_net_param_grid

from sklearn.linear_model import ElasticNet
import pandas as pd


DATAPATH = './dataset_fe.csv'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH)
    parser.add_argument('-model_type', default='elastic_net')
    parser.add_argument('-experiment_name', default='rtn_title_len_elastic_net')

    parser = parser.parse_args()

    dataset = pd.read_csv(parser.fp)
    x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    if parser.model_type == 'elastic_net':
        hyper_optimizer = HyperoptHPOptimizer(x, y, elastic_net_param_grid, model_class=ElasticNet, max_evals=1000)
