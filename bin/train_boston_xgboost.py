#!/home/user/miniconda/envs/xgboost-1.0.2-cuda-10.1/bin/python
import xgboost as xgb
from sklearn.datasets import load_boston

boston = load_boston()

params = {'silent': False, 'tree_method': 'gpu_hist',
          'n_estimators': 100, 'subsample': 0.5}

# sklearn API example
gbm = xgb.XGBRegressor(**params)
gbm.fit(boston.data, boston.target, eval_set=[(boston.data, boston.target)])

#
# Additional parameters for gpu_hist tree method
# single_precision_histogram, [default=``false``]
# Use single precision to build histograms. See document for GPU support for more details.

# deterministic_histogram, [default=``true``]

# Build histogram on GPU deterministically. Histogram building is not deterministic due to the non-associative aspect of floating point summation.
# We employ a pre-rounding routine to mitigate the issue, which may lead to slightly lower accuracy. Set to false to disable it.

# ________________________________________________________________________________________________________________________________________________________
# Choice of algorithm to fit linear model

# shotgun: Parallel coordinate descent algorithm based on shotgun algorithm. Uses ‘hogwild’ parallelism and therefore produces a nondeterministic solution on each run.
# -> This should be linted against!!!