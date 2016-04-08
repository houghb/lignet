import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except:
    import pickle

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

stuff = pickle.load(open('single_lr_ann_objects.pkl', 'rb'))

net = stuff[0]
param_grid = stuff[1]
grid_search = stuff[2]

# this shows you the scores for all the different searches
for entry in grid_search.grid_scores_:
    print entry

# quickly show which was the estimator with the best score
print 'The best estimator was:\n'
print grid_search.best_params_

