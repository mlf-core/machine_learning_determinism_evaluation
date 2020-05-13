#!/home/user/miniconda/envs/xgboost-1.0.2-cuda-10.1/bin/python
import click
import xgboost as xgb
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import array as da
from xgboost.dask import DaskDMatrix
import dask.array as da
import numpy as np
from sklearn.datasets import fetch_covtype, load_boston
from dask_ml.model_selection import train_test_split
import time
import random
import os


@click.command()
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=10)
@click.option('--n-gpus', type=int, default=2)
@click.option('--dataset', type=click.Choice(['boston', 'covertype']), default='covertype')
def train(seed, epochs, n_gpus, dataset):
    with LocalCUDACluster(n_workers=n_gpus, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            # Fetch dataset using sklearn
            if dataset == 'boston':
                dataset = load_boston()
                param = {

                }
            elif dataset == 'covertype':
                dataset = fetch_covtype()
                param = {
                    'objective': 'multi:softmax',
                    'num_class': 8
                    # 'single_precision_histogram': True
                }
            
            param['verbosity'] = 2
            param['tree_method'] = 'gpu_hist'

            # Rechunking is required for the covertype dataset
            X = da.from_array(dataset.data, chunks=1000)
            y = da.from_array(dataset.target, chunks=1000)

            # Create 0.75/0.25 train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

            dtrain = DaskDMatrix(client, X_train, y_train)
            dtest = DaskDMatrix(client, X_test, y_test)

            random_seed(seed, param)
            
            gpu_runtime = time.time()
            model_training_results = xgb.dask.train(client,
                                    param,
                                    dtrain,
                                    num_boost_round=epochs,
                                    evals=[(dtest, 'test')])
            
            print(model_training_results)
            print(f'GPU Run Time: {str(time.time() - gpu_runtime)} seconds')


def random_seed(seed, param):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed)
    random.seed(seed) # Python random
    param['seed'] = seed


if __name__ == '__main__':
    train()