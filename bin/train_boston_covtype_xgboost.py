#!/home/user/miniconda/envs/xgboost-1.0.2-cuda-10.1/bin/python
from email.policy import default
import click
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype, load_boston
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time
import random
import os


@click.command()
@click.option('--setting', type=str, default='seed')
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=25)
@click.option('--no-cuda', type=bool, default=False)
@click.option('--dataset', type=click.Choice(['boston', 'covertype']), default='covertype')
def train(seed, epochs, setting, no_cuda, dataset):
    param = {}
    if setting == 'single_precision':
        param['single_precision_histogram'] = True
    if dataset == 'boston':
        dataset = load_boston()
    elif dataset == 'covertype':
        dataset = fetch_covtype()
        param['objective'] = 'multi:softmax'
        param['num_class'] = 8

    X = dataset.data
    y = dataset.target

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

    # Set random seeds
    if setting == 'seed' or setting == 'single_precision':
        random_seed(seed, param)
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
    
    feature_importances_dict = defaultdict(list)
    for i in range(10):
        random_seed(i + 40, param)
        
        model = xgb.train(param, dtrain, epochs, evals=[(dtest, 'test')], evals_result=results)
        if not no_cuda:
            print(f'GPU Run Time: {str(time.time() - gpu_runtime)} seconds')
        else:
            print(f'CPU Run Time: {str(time.time() - gpu_runtime)} seconds')
            
        feature_importances = model.get_score()
        
        for key, val in feature_importances.items():
            feature_importances_dict[key].append(val)
    
    for fis in feature_importances_dict.values():
        fi_std = np.std(fis)
        print(fi_std)
    
def random_seed(seed, param):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed)
    random.seed(seed) # Python random
    param['seed'] = seed


if __name__ == '__main__':
    train()
