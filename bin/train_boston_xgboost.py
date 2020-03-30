#!/home/user/miniconda/envs/xgboost-1.0.2-cuda-10.1/bin/python
import xgboost as xgb
from sklearn.datasets import load_boston

boston = load_boston()

# XGBoost API example
params = {'tree_method': 'gpu_hist', 'max_depth': 3, 'learning_rate': 0.00001}
dtrain = xgb.DMatrix(boston.data, boston.target)
xgb.train(params, dtrain, evals=[(dtrain, "train")])

# sklearn API example
gbm = xgb.XGBRegressor(silent=False, n_estimators=100, tree_method='gpu_hist')
# just training 50 times to verify that the GPU is actually being used
gbm.fit(boston.data, boston.target, eval_set=[(boston.data, boston.target)])
