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

# specify the seed for random number generation so we can get consistent
# shuffling and initialized weights
np.random.seed(6509)

# Read the training data for the neural network
# Input data is 250000x4 and output data is 250000x32
X = pd.read_csv('../../parameters_250000.txt', sep=' ')
Y = pd.read_csv('../../results.txt', sep=' ', index_col=False)
# These functional groups do not exist in my model
Y = Y.drop(['light_aromatic_C-C', 'light_aromatic_methoxyl'], axis=1)

X = X.values.astype(np.float32)
Y = Y.values.astype(np.float32)

# Shuffle the dataset (because x parameters are varied in a structured way)
combined = np.concatenate((X, Y), axis=1)
np.random.shuffle(combined)

# Separate the data into training (with included validation) and test sets.
# (Validation set is separated from the training set by nolearn)
test_fraction = 0.2
training = combined[:-int(test_fraction * combined.shape[0]), :]
test = combined[-int(test_fraction * combined.shape[0]):, :]

x_train = training[:, :4]
y_train = training[:, 4:]
x_test = test[:, :4]
y_test = test[:, 4:]

# Standardize the input parameters to have zero mean and unit variance
# x_scaler.transform() can be used later to transform any new data
# x_scaler.inverse_transform() can be used to get the original data back
x_scaler = preprocessing.StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

# Scale the output parameters to lie between 0 and 1
y_scaler = preprocessing.MinMaxScaler().fit(y_train)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)

# NeuralNet automatically splits the data provided into a training
# and a validation set, using 20% of the samples for validation.
# You can adjust this ratio by overriding the eval_size=0.2 parameter.
# default objective is lasagne.objectives.squared_error.
net = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('hidden0', layers.DenseLayer),
                ('hidden1', layers.DenseLayer),
                ('output', layers.DenseLayer)
                ],
            input_shape=(None, x_train.shape[1]),
            hidden0_num_units=15,
            hidden0_nonlinearity=nonlinearities.sigmoid,
            hidden1_num_units=17,
            hidden1_nonlinearity=nonlinearities.sigmoid,
            output_num_units=y_train.shape[1],
            output_nonlinearity=nonlinearities.linear,
            regression=True,
#             update=lasagne.updates.nesterov_momentum,
#             update_momentum=0.9,
            update_learning_rate=0.5,
            verbose=1,
            max_epochs=500,
#             train_split=TrainSplit(eval_size=0.2),
#             objective_l2=0.0001
            # use on_epoch_finished to update the learning rate during training
            # or use early stopping.
            # see https://github.com/dnouri/kfkd-tutorial/blob/master/kfkd.py
            )

# Use get_params to get all the hyperparameters that make up the estimator.
# Any of these parameters can be optimized with gridsearch
# net.get_params()
param_grid = {'hidden0_num_units': range(18, 30),
              'hidden1_num_units': range(18, 30),
              'update_learning_rate': [0.5, 0.9, 2]
              }
grid_search = GridSearchCV(net, param_grid, verbose=0, n_jobs=20,
                           pre_dispatch='2*n_jobs',
                           scoring='mean_squared_error')
grid_search.fit(x_train[:8000,:], y_train[:8000,:])

# quickly show which was the estimator with the best score
print 'The best estimator was:\n'
print grid_search.best_params_

# This is the NeuralNet object with the best fit
grid_search.best_estimator_.save_params_to('double_hidden_best_params.dat')
with open('double_hidden_ann_objects.pkl', 'wb') as pkl:
    pickle.dump([net, param_grid, grid_search], pkl)
