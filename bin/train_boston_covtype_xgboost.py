#!/home/user/miniconda/envs/xgboost-1.0.2-cuda-10.1/bin/python
import click
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype, load_boston
from sklearn.model_selection import train_test_split
import time
import random
import os


@click.command()
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=25)
@click.option('--no-cuda', type=bool, default=False)
@click.option('--dataset', type=click.Choice(['boston', 'covertype']), default='covertype')
def train(seed, epochs, no_cuda, dataset):
    # Fetch dataset using sklearn
    if dataset == 'boston':
        dataset = load_boston()
        param = {
            # 'single_precision_histogram': True
        }
    elif dataset == 'covertype':
        dataset = fetch_covtype()
        param = {
            'objective': 'multi:softmax',
            'num_class': 8,
            'single_precision_histogram': True
        }

    X = dataset.data
    y = dataset.target

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

    # Set random seeds
    # random_seed(seed, param)
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['colsample_bylevel'] = 0.5

    # Convert input data from numpy to XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set CPU or GPU as training device
    if no_cuda:
        param['tree_method'] = 'hist'
    else:
        param['tree_method'] = 'gpu_hist'

    # Train on the chosen device
    results = {}
    gpu_runtime = time.time()
    xgb.train(param, dtrain, epochs, evals=[(dtest, 'test')], evals_result=results)
    if not no_cuda:
        print(f'GPU Run Time: {str(time.time() - gpu_runtime)} seconds')
    else:
        print(f'CPU Run Time: {str(time.time() - gpu_runtime)} seconds')

def random_seed(seed, param):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed)
    random.seed(seed) # Python random
    param['seed'] = seed


if __name__ == '__main__':
    train()