"""
gridsearch to find best values for max_depth and min_samples_leaf in
a decision tree that predicts all the output measures.
"""

import numpy as np
import time

from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

from lignet_utils import gen_train_test
from constants import Y_COLUMNS

script_start_time = time.time()

x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()

dtr_full = tree.DecisionTreeRegressor(max_depth=20, min_samples_leaf=5)

# Set up the gridsearch
param_grid = {'max_depth': range(10, 40),
              'min_samples_leaf': range(1, 15),
              }

grid_search = GridSearchCV(dtr_full, param_grid, verbose=0, n_jobs=20,
                           pre_dispatch='2*n_jobs',
                           scoring='mean_squared_error')

# search the network parameters using a subset of the entire training set
grid_search.fit(x_train[:, :], y_train[:, :])

script_running_time = time.time() - script_start_time

with open('decision_tree_gridsearch_report.txt', 'w+') as report:
    print >>report, ('took % sec\n\n' % script_running_time)
    for entry in grid_search.grid_scores_:
       print >>report, entry
    print >>report, '\n\nbest: %s' % grid_search.best_params_
