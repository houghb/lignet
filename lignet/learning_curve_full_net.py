"""
Make a learning curve for the full neural net trained on all 30 output
measures.  The point of this graph is to investigate how much training data
is needed to achieve various levels of fit.
"""
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne.nonlinearities import ScaledTanH
from nolearn.lasagne import NeuralNet, TrainSplit, RememberBestWeights
from sklearn.learning_curve import learning_curve

from lignet_utils import gen_train_test


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


x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()

# set up the Scaled tanh parameters. See nonlinearities.py for usage notes.
# I am following the guidance of LeCun et al. for these values
scaled_tanh = ScaledTanH(scale_in=2./3, scale_out=1.7159)

# Use RememberBestWeights to use the parameters that yielded the best loss
# on the validation set.
rbw = RememberBestWeights()

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
#        on_epoch_finished=[EarlyStopping(patience=700), rbw],
#        on_training_finished=[rbw.restore],
        train_split=TrainSplit(eval_size=0.3),
        ),
    xt, yt,
    train_sizes=[500, 1500, 5000, 15000, 35000, 75000, 133333],
    scoring='mean_squared_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

with open('learning_curve_larger.pkl', 'wb') as pkl:
    pickle.dump([train_scores_mean, train_scores_std,
                 valid_scores_mean, valid_scores_std,
                 train_sizes], pkl)

