"""
This file creates the network that I've determined is best based on the
gridsearch results, then it trains that network using the full training set.

Gridsearch results showed that I should have:
22 nodes in hidden layer
update momentum of 0.8
update learning rate of 0.1
"""

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

# specify the seed for random number generation so we can get consistent
# shuffling and initialized weights
np.random.seed(6509)

# Read the training data for the neural network
# Input data is 250000x4 and output data is 250000x32
X = pd.read_csv('../parameters_250000.txt', sep=' ')
Y = pd.read_csv('../results.txt', sep=' ', index_col=False)
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
                ('output', layers.DenseLayer)
                ],
            input_shape=(None, x_train.shape[1]),
            hidden0_num_units=22,
            hidden0_nonlinearity=nonlinearities.sigmoid,
            output_num_units=y_train.shape[1],
            output_nonlinearity=nonlinearities.linear,
            regression=True,
#             update=lasagne.updates.nesterov_momentum,
            update_momentum=0.8,
            update_learning_rate=0.1,
            verbose=1,
            max_epochs=5000,
#             train_split=TrainSplit(eval_size=0.2),
#             objective_l2=0.0001
            # use on_epoch_finished to update the learning rate during training
            # or use early stopping.
            # see https://github.com/dnouri/kfkd-tutorial/blob/master/kfkd.py
            )

# Train the network parameters using the entire training set
net.fit(x_train[:, :], y_train[:, :])

# Save trained network and other relevant objects for later
net.save_params_to('ann_params.dat')
with open('ann_objects.pkl', 'wb') as pkl:
    pickle.dump([net, x_scaler, y_scaler],
                pkl)

