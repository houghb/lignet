"""
Perform a gridsearch to identify the best number of nodes and batch size
"""

import sys
import time
import numpy as np
import pandas as pd

import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne.nonlinearities import ScaledTanH
from nolearn.lasagne import NeuralNet, TrainSplit, RememberBestWeights
from sklearn import preprocessing

script_start_time = time.time()

# first argument passed is the column of the output measure to train
output_col = int(sys.argv[1])

# specify the seed for random number generation so we can get consistent
# shuffling and initialized weights
np.random.seed(6509)

# Read the training data for the neural network
# Input data is 250000x4 and output data is 250000x32
X = pd.read_csv('../parameters_250000.txt', sep=' ')
Y = pd.read_csv('../results.txt', sep=' ', index_col=False)
# These functional groups do not exist in my model
Y = Y.drop(['light_aromatic_C-C', 'light_aromatic_methoxyl'], axis=1)
y_columns = Y.columns.values
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

# Standardize the input & output parameters to have zero mean and unit variance
# x_scaler.transform() can be used later to transform any new data
# x_scaler.inverse_transform() can be used to get the original data back
x_scaler = preprocessing.StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
y_scaler = preprocessing.StandardScaler().fit(y_train)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)

# set up the Scaled tanh parameters.  See nonlinearities.py for usage notes.
# I am following the guidance of LeCun et al. for these values
scaled_tanh = ScaledTanH(scale_in=2./3, scale_out=1.7159)

# Use RememberBestWeights to use the parameters that yielded the best loss
# on the validation set.
rbw = RememberBestWeights()

# implement early stopping so we don't have to reach max_epochs
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

# NeuralNet automatically splits the data provided into a training
# and a validation set, using 20% of the samples for validation.
# You can adjust this ratio by overriding the eval_size=0.2 parameter.
# .
# adagrad scales the learning rates by dividing by the square root of
# accumulated squared gradients.  This is an adaptive learning rate
# technique.  See the references in updates.py for details.
# .
# setting the first dimension of input_shape to None translates to using a
# variable batch size
# .
# the initial weights are initialized from a uniform distribution with a
# cleverly chosen interval - Lasagne figures out this interval using
# Glorot-style initialization
# (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
# .
# Mean squared error is the default objective function to minimize
net = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('hidden0', layers.DenseLayer),
                ('hidden1', layers.DenseLayer),
                ('output', layers.DenseLayer)
                ],
            input_shape=(None, x_train.shape[1]),
            hidden0_num_units=15,
            hidden0_nonlinearity=scaled_tanh,
            hidden1_num_units=15,
            hidden1_nonlinearity=scaled_tanh,
            output_num_units=1,
            output_nonlinearity=nonlinearities.linear,
            regression=True,
            verbose=1,
            max_epochs=500,
            update=lasagne.updates.adagrad,
            on_epoch_finished=[EarlyStopping(patience=100), rbw],
            on_training_finished=[rbw.restore],
            train_split=TrainSplit(eval_size=0.3),
            batch_iterator_train=BatchIterator(batch_size=128),
            )

# Set up the gridsearch
param_grid = {'hidden0_num_units': range(5, 32),
			  'hidden1_num_units': range(5, 32),
              'batch_iterator_train': [BatchIterator(batch_size=80),
									   BatchIterator(batch_size=128),
									   BatchIterator(batch_size=200)]
              }
grid_search = GridSearchCV(net, param_grid, verbose=0, n_jobs=20,
                           pre_dispatch='2*n_jobs',
                           scoring='mean_squared_error')

# select only the output measure you want to train
y_train2 = y_train[:, output_col]
# Train the network parameters using a subset of the entire training set
grid_search.fit(x_train[:6000, :], y_train2[:6000])

script_running_time = time.time() - script_start_time

output = []
output.append('%s - %s took % sec\n\n' % (output_col, y_columns[output_col],
									script_running_time))

for entry in grid_search.grid_scores_:
    output.append(entry + '\n')

output.append('\n\n%s' % grid_search.best_params_)

with open('updated_gridsearch_single_report.txt', 'a') as report:
	report.writelines(output)



