"""
Make a learning curve for the full neural net trained on all 30 output
measures.  The point of this graph is to investigate how much training data
is needed to achieve various MSE values.
"""
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne.nonlinearities import ScaledTanH
from nolearn.lasagne import NeuralNet, TrainSplit
from sklearn.learning_curve import learning_curve

from lignet_utils import gen_train_test

x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()

# set up the Scaled tanh parameters. See nonlinearities.py for usage notes.
# I am following the guidance of LeCun et al. for these values
scaled_tanh = ScaledTanH(scale_in=2./3, scale_out=1.7159)

# Make a learning curve to find out how much training data to use
train_size = int(1 * x_train.shape[0])
xt = x_train[:train_size, :]
yt = y_train[:train_size, :]

train_sizes, train_scores, valid_scores = learning_curve(
    NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden0', layers.DenseLayer),
            ('hidden1', layers.DenseLayer),
            ('output', layers.DenseLayer)
            ],
        input_shape=(None, x_train.shape[1]),
        hidden0_num_units=18,
        hidden0_nonlinearity=scaled_tanh,
        hidden1_num_units=20,
        hidden1_nonlinearity=scaled_tanh,
        output_num_units=y_train.shape[1],
        output_nonlinearity=nonlinearities.linear,
        regression=True,
        verbose=1,
        max_epochs=4000,
        update=lasagne.updates.adagrad,
        train_split=TrainSplit(eval_size=0.3),
        ),
    xt, yt,
    train_sizes=[500, 1500, 5000, 15000, 35000, 75000, 133333],
    scoring='mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

with open('learning_curve.pkl', 'wb') as pkl:
    pickle.dump([train_scores_mean, train_scores_std,
                 valid_scores_mean, valid_scores_std,
                 train_sizes], pkl)

