"""
Functions used to analyze the trained neural networks.
"""

try:
    import cPickle as pickle
except:
    import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import lasagne
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet, TrainSplit
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from create_and_train import EarlyStopping
from constant import Y_COLUMNS


def gen_train_test(params='default', results='default'):
    """
    Re-create the scaled training (includes validation) and test sets that
    were used in training the networks.  This function assumes
    there are four parameters that are varied in your input layer and that
    your results file contains two extra columns: 'light_aromatic_C-C', and
    'light_aromatic_methoxyl'.

    Parameters
    ----------
    params   : str, optional
               the relative or absolute path to the text file of parameters
               used in the input layer of the network
    results  : str, optional
               the relative or absolute path to the text file of expected
               values for the output layer of the network

    Returns
    -------
    x_train  : numpy.ndarray
               the scaled training data used for the input layer of the network
               (standardized to have zero mean and unit variance).  This array
               is split into a training and a validation set by nolearn
    x_test   : numpy.ndarray
               scaled data you can use as a test set that the network has never
               seen before.  It is 20% of the data in params.
    y_train  : numpy.ndarray
               the scaled training data used for the output layer of the
               network (standardized to have zero mean and unit variance).
    y_test   : numpy.ndarray
               scaled expected output values that the network has not seen
               before to use with x_test.
    x_scaler : sklearn.preprocessing.data.StandardScaler
               a scaler that can be used later to scale new data or recover
               the original values (x_scaler.transform(),
               x_scaler.inverse_transform())
    y_scaler : sklearn.preprocessing.data.StandardScaler
               a scaler that can be used later to scale new data or recover
               the original values (y_scaler.transform(),
               y_scaler.inverse_transform())
    """
    # use the same random seed that was used initially
    np.random.seed(6509)

    # set the paths to appropriate files
    if params == 'default':
        params = '../../parameters_250000.txt'
    if results == 'default':
        results = '../../results.txt'

    # Read the training data for the neural network
    # Default input data is 250000x4 and output data is 250000x32
    X = pd.read_csv(params, sep=' ')
    Y = pd.read_csv(results, sep=' ', index_col=False)
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

    # Standardize input & output to have zero mean and unit variance
    # x_scaler.transform() can be used later to transform any new data
    # x_scaler.inverse_transform() can be used to get the original data back
    x_scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_scaler = preprocessing.StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test, x_scaler, y_scaler


def load_nets(myglob):
    """
    Load a dictionary of trained neural networks with filenames
    matching a unix-style pattern.  Filenames of networks trained for
    individual output measures should be named,
    `final-OutputCol-NameOfOutputCol_net.pkl` and filenames for a network
    trained for the complete set of output measures should be `full_net.pkl`.

    Parameters
    ----------
    myglob  : str
              the unix-style pattern (including abs or rel path) for the
              filenames you want to load.  Wildcards accepted.

    Returns
    -------
    nets  : dict
            a dictionary with the trained networks.  Keys are the columns
            from y_train that the network is trained on.
    """
    files_list = glob.glob(myglob)
    nets = {}

    if [x for x in files_list if 'full' in x]:
        with open(files_list[0], 'rb') as pkl:
            nets['all'] = pickle.load(pkl)[0]
    else:
        for filename in files_list:
            output_col = int(filename.split('/')[-1].split('-')[1])
            with open(filename, 'rb') as pkl:
                nets[output_col] = pickle.load(pkl)[0]

    return nets


def calc_r_squared(nets, x_train, y_train, x_test, y_test):
    """
    Calculate the r squared value of neural nets for the training, validation,
    and test sets.

    Parameters
    ----------
    nets    : dict
              a dictionary with the trained neural nets
    x_train : numpy.ndarray
              the scaled training data used for the input layer of the
              network (standardized to have zero mean and unit variance)
    y_train : numpy.ndarray
              the scaled training data used for the output layer of the
              network (standardized to have zero mean and unit variance)
    x_test  : numpy.ndarray
              scaled data you can use as a test set that the network has never
              seen before.  It is 20% of the data in params.
    y_test  : numpy.ndarray
              scaled expected output values that the network has not seen
              before to use with x_test.

    Returns
    -------
    r_squared : dict
                a dictionary with the r^2 values for each output measure
    """
    r_squared = {}

    if len(nets) == 1:
        net = nets['all']
        # get the same train/test split that was used in setting up the net
        Xt, Xv, yt, yv = net.train_split(x_train, y_train, net)
        # calculate the values predicted by the network
        y_pred_train = net.predict(Xt)
        y_pred_valid = net.predict(Xv)
        y_pred_test = net.predict(x_test)

        for i, title in enumerate(Y_COLUMNS):
            # calculate r**2 values
            r2_train = round(r2_score(yt[:, i], y_pred_train[:, i]), 5)
            r2_valid = round(r2_score(yv[:, i], y_pred_valid[:, i]), 5)
            r2_test = round(r2_score(y_test[:, i], y_pred_test[:, i]), 5)
            r_squared[title] = r2_train, r2_valid, r2_test

    else:
        for key in nets:
            net = nets[key]
            title = Y_COLUMNS[key]
            # get the same train/test split that was used in setting up the net
            Xt, Xv, yt, yv = net.train_split(x_train, y_train[:, key], net)
            # calculate the values predicted by the network
            y_pred_train = net.predict(Xt)
            y_pred_valid = net.predict(Xv)
            y_pred_test = net.predict(x_test)
            # calculate r**2 values
            r2_train = round(r2_score(yt[:], y_pred_train[:]), 5)
            r2_valid = round(r2_score(yv[:], y_pred_valid[:]), 5)
            r2_test = round(r2_score(y_test[:, key], y_pred_test[:]), 5)
            r_squared[title] = r2_train, r2_valid, r2_test

    return r_squared


def calc_r_squared_dt(dtr, x_train, x_test, y_train, y_test):
    """
    Calculate the r squared value from the decision tree regressor for the
    training and test sets.

    Parameters
    ----------
    dtr     : sklearn.tree.tree.DecisionTreeRegressor
              a trained decision tree
    x_train : numpy.ndarray
              the scaled training data used for the input layer of the
              network (standardized to have zero mean and unit variance)
    y_train : numpy.ndarray
              the scaled training data used for the output layer of the
              network (standardized to have zero mean and unit variance)
    x_test  : numpy.ndarray
              scaled data you can use as a test set that the network has never
              seen before.  It is 20% of the data in params.
    y_test  : numpy.ndarray
              scaled expected output values that the network has not seen
              before to use with x_test.

    Returns
    -------
    r_squared : dict
                a dictionary with the r^2 values for each output measure
    """

    r_squared = {}
    ytrainpred = dtr.predict(x_train)
    ytestpred = dtr.predict(x_test)
    for i, title in enumerate(Y_COLUMNS):
        r2_train = round(r2_score(y_train[:, i], ytrainpred[:, i]), 5)
        r2_test = round(r2_score(y_test[:, i], ytestpred[:, i]), 5)
        r_squared[title] = r2_train, r2_test

    r2_train_overall = round(r2_score(y_train.ravel(),
                                      ytrainpred.ravel()), 5)
    r2_test_overall = round(r2_score(y_test.ravel(),
                                     ytestpred.ravel()), 5)
    r_squared['ALL'] = r2_train_overall, r2_test_overall

    return r_squared


def print_r_squared(r_squared, title=''):
    """
    Print the R squared report in a easy to read way.

    Parameters
    ----------
    r_squared : dict
                dictionary returned by `calc_r_squared()`
    title     : str, optional
                an optional title to prepend to the printed description
    Returns
    -------
    None
    """
    keylist = r_squared.keys()
    keylist.sort()
    print '%s R**2 values for training, validation, and test sets\n' % title
    for x in keylist:
        print '%s:%s %s %s' % ('{: <28}'.format(x),
                               '{: >10}'.format(r_squared[x][0]),
                               '{: >10}'.format(r_squared[x][1]),
                               '{: >10}'.format(r_squared[x][2]))
    print '\n'

    return None


def transform_pred_to_actual(pred, output_col, scaler):
    """
    Transform a set of predicted values back to their actual values before
    scaling.

    Parameters
    ----------
    pred       : numpy.ndarray
                 a set of predicted values for one specific output measure
    output_col : int
                 the index of the output measure you predicted
    scaler     : sklearn.preprocessing scaler
                 the scaler that was originally used to scale your data

    Returns
    -------
    actual : numpy.ndarray
             the unscaled values of your prediction
    """
    rows = pred.shape[0]
    temp = np.zeros((rows, 30))
    temp[:, output_col] = pred.ravel()
    actual = scaler.inverse_transform(temp)[:, output_col]

    return actual
