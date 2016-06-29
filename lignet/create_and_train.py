"""
This file creates the network that I've determined is best based on the
gridsearch results, then it trains that network using the full training set.

To run this file for the full network (and train all the output measures at 
once), pass 999 as the first argument.

Gridsearch results for each output measure are in the `gridsearches` folder,
and encoded into the bash scripts that call this file script, which passes
them as arguments to this file.
"""

import sys
import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except:
    import pickle
import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne.nonlinearities import ScaledTanH
from nolearn.lasagne import NeuralNet, TrainSplit, RememberBestWeights
#from nolearn.lasagne import BatchIterator
from sklearn import preprocessing


class EarlyStopping(object):
    """implement early stopping so we don't have to reach max_epochs"""
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


if __name__ == '__main__':
    # first argument passed is the column of the output measure to train
    output_col = int(sys.argv[1])
    # second and third arguments are the number of nodes in the hidden layers
    hidden0_units = int(sys.argv[2])
    hidden1_units = int(sys.argv[3])

    # specify the seed for random number generation so we can get consistent
    # shuffling and initialized weights
    np.random.seed(6509)

    # Read the training data for the neural network
    # Input data is 250000x4 and output data is 250000x32
    X = pd.read_csv('../../parameters_250000.txt', sep=' ')
    Y = pd.read_csv('../../results.txt', sep=' ', index_col=False)
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

    # Standardize input & output to have zero mean and unit variance
    # x_scaler.transform() can be used later to transform any new data
    # x_scaler.inverse_transform() can be used to get the original data back
    x_scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_scaler = preprocessing.StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    # set up the Scaled tanh parameters. See nonlinearities.py for usage notes.
    # I am following the guidance of LeCun et al. for these values
    scaled_tanh = ScaledTanH(scale_in=2./3, scale_out=1.7159)

    # Use RememberBestWeights to use the parameters that yielded the best loss
    # on the validation set.
    rbw = RememberBestWeights()

    # Set the number of output units
    if output_col == 999:
        out_units = y_train.shape[1]
    else:
        out_units = 1

    # NeuralNet automatically splits the data provided into a training
    # and a validation set, using the eval_size below.
    # .
    # adagrad scales the learning rates by dividing by the square root of
    # accumulated squared gradients.  This is an adaptive learning rate
    # technique.  See the references in updates.py for details.
    # .
    # setting the first dimension of input_shape to None translates to using a
    # variable batch size
    # .
    # the weights are initialized from a uniform distribution with a
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
                hidden0_num_units=hidden0_units,
                hidden0_nonlinearity=scaled_tanh,
                hidden1_num_units=hidden1_units,
                hidden1_nonlinearity=scaled_tanh,
                output_num_units=out_units,
                output_nonlinearity=nonlinearities.linear,
                regression=True,
                verbose=1,
                max_epochs=6000,
                update=lasagne.updates.adagrad,
                on_epoch_finished=[EarlyStopping(patience=900), rbw],
                on_training_finished=[rbw.restore],
                train_split=TrainSplit(eval_size=0.3),
                #batch_iterator_train=BatchIterator(batch_size=128),
                )

    # select only the output measure(s) you want to train, fit the network
    # with all the training data, and pickle the result.
    if output_col == 999:
        net.fit(x_train[:, :], y_train[:, :])
        with open('trained_networks/full_net.pkl', 'wb') as pkl:
            pickle.dump([net, x_scaler, y_scaler], pkl)

    else:
        y_train2 = y_train[:, output_col]
        net.fit(x_train[:, :], y_train2[:])
        with open('trained_networks/final-%s-%s_net.pkl' %
                          (output_col, y_columns[output_col]),
                  'wb') as pkl:
            pickle.dump([net, x_scaler, y_scaler], pkl)
