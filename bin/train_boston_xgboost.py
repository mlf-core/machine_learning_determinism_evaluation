#!/home/user/miniconda/envs/xgboost-1.0.2-cuda-10.1/bin/python
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time
import random

# Fetch dataset using sklearn
cov = fetch_covtype()
X = cov.data
y = cov.target

np.random.seed(0)
random.seed(0) # Python general seed

# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

# Leave most parameters as default
param = {'objective': 'multi:softmax',
         'num_class': 8,
         'seed': 0,
        # 'single_precision_histogram': True
         }

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Specify sufficient boosting iterations to reach a minimum
num_round = 100

# GPU Training
param['tree_method'] = 'gpu_hist'
gpu_res = {}
tmp = time.time()
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

# Repeat for CPU algorithm
# tmp = time.time()
# param['tree_method'] = 'hist'
# cpu_res = {}
# xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
# print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
