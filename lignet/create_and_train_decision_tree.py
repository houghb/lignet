"""
Train the decision tree that we've determined gives the best performance on
our training data.  See the analyze_decision_tree ipython notebook for more
details.
"""

import cPickle as pickle
from sklearn import tree
from lignet_utils import gen_train_test

x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()

dtr_full = tree.DecisionTreeRegressor(max_depth=26, min_samples_leaf=2)
dtr_full = dtr_full.fit(x_train, y_train)

# pickle the trained regressor
with open('trained_networks/decision_tree.pkl', 'wb') as pkl:
    pickle.dump([dtr_full], pkl)