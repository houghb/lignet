"""
Functions to plot information relating to the trained neural nets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from constants import Y_COLUMNS


def plot_one_learning_curve(output_col, nets):
    """
    Plot the learning curves for a single output measure.

    Parameters
    ----------
    output_col : int or str
                 the key of the network in 'nets' that you want to plot
    nets       : a dictionary with the trained networks.  Keys are the columns
                 from y_train that the network is trained on.

    Returns
    -------
    None
    """
    if output_col == 'all':
        title = 'Full Network'
    else:
        title = Y_COLUMNS[output_col]
    net = nets[output_col]

    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])

    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.title('%s Learning Curves' % title)

    return None


def pplot_one_output(ax, net, x_train, y_train, x_test, y_test, output_col,
                     title, ub=None):
    """
    Make a parity plot for the training (blue), validation (green),
    and test (red) set predictions of a specific output measure.
    This function can be called for a single plot (`f, ax = plt.subplots()`),
    or to plot many subplots in the same figure, but it only works for networks
    that are trained on individual output measures (not the full network).

    Parameters
    ----------
    ax         : matplotlib.axes._subplots.AxesSubplot
                 axes object to plot on
    net        : nolearn.lasagne.base.NeuralNet
                 a trained neural network
    x_train    : numpy.ndarray
                 the scaled training data used for the input layer of the
                 network (standardized to have zero mean and unit variance)
    x_test     : numpy.ndarray
                 scaled data you can use as a test set that the network has
                 never seen before
    y_train    : numpy.ndarray
                 the scaled training data used for the output layer of the
                 network (standardized to have zero mean and unit variance)
    y_test     : numpy.ndarray
                 scaled expected output values that the network has not seen
                 before to use with x_test
    output_col : int
                 the column of the output measure in y_train to plot
    title      : string
                 the name of the output measure being plotted
    ub         : int
                 the number of points you would like to include in plot

    Returns
    -------
    None
    """
    # get the same train/test split that was used in setting up the network
    Xt, Xv, yt, yv = net.train_split(x_train, y_train[:, output_col], net)
    # calculate the values predicted by the network
    y_pred_train = net.predict(Xt)
    y_pred_valid = net.predict(Xv)
    y_pred_test = net.predict(x_test)

    r2_train = round(r2_score(yt[:], y_pred_train[:]), 4)
    r2_valid = round(r2_score(yv[:], y_pred_valid[:]), 4)
    r2_test = round(r2_score(y_test[:, output_col], y_pred_test[:]), 4)

    if ub is None:
        ub = y_pred_train.shape[0]

    # If you are making just a single plot
    try:
        if ax.numRows:
            ax.plot([np.min(yt[:ub]), np.max(yt[:ub])],
                    [np.min(yt[:ub]), np.max(yt[:ub])], c='black')
            ax.scatter(y_pred_train[:ub].flatten(), yt[:ub].flatten(),
                       s=0.1, alpha=0.16, c='b', marker='x', label='train')
            ax.scatter(y_pred_valid[:ub].flatten(), yv[:ub].flatten(),
                       s=0.1, alpha=0.16, c='g', marker='x', label='valid')
            ax.scatter(y_pred_test[:ub].flatten(),
                       y_test[:ub, output_col].flatten(),
                       s=0.1, alpha=0.16, c='r', marker='x', label='test')
            ax.set_title('%s - r**2=%s, %s, %s' %
                         (title, r2_train, r2_valid, r2_test))
    except AttributeError:
        ax[output_col].plot([np.min(yt[:ub]), np.max(yt[:ub])],
                            [np.min(yt[:ub]), np.max(yt[:ub])], c='black')
        ax[output_col].scatter(y_pred_train[:ub].flatten(),
                               yt[:ub].flatten(),
                               s=0.1, alpha=0.16,
                               c='b', marker='x', label='train')
        ax[output_col].scatter(y_pred_valid[:ub].flatten(),
                               yv[:ub].flatten(),
                               s=0.1, alpha=0.16,
                               c='g', marker='x', label='valid')
        ax[output_col].scatter(y_pred_test[:ub].flatten(),
                               y_test[:ub, output_col].flatten(),
                               s=0.1, alpha=0.16,
                               c='r', marker='x', label='test')
        ax[output_col].set_title('%s - r**2=%s, %s, %s' %
                                 (title, r2_train, r2_valid, r2_test))
    return None


def pplot_one_output_full(ax, yt, yv, ytest, ytpred, yvpred, ytestpred,
                          output_col, ub=None):
    """
    Make a parity plot for the training (blue), validation(green),
    and test (red) set predictions of a specific output measure within the
    full network (trained on all the output measures at once).

    Parameters
    ----------
    ax         : matplotlib.axes._subplots.AxesSubplot
                 axes object to plot on
    yt         : numpy ndarray
                 training set actual values
    yv         : numpy ndarray
                 validation set actual values
    ytest      : numpy ndarray
                 test set actual values
    ytpred     : numpy ndarray
                 training set predicted values
    yvpred     : numpy ndarray
                 validation set predicted values
    ytestpred  : numpy ndarray
                 test set predicted values
    output_col : int
                 the column of the output measure you want to plot
    ub         : int
                 the number of points you would like to include in plot

    Returns
    -------
    None
    """

    r2_train = round(r2_score(yt[:, output_col],
                              ytpred[:, output_col]), 4)
    r2_valid = round(r2_score(yv[:, output_col],
                              yvpred[:, output_col]), 4)
    r2_test = round(r2_score(ytest[:, output_col],
                             ytestpred[:, output_col]), 4)

    if ub is None:
        ub = ytpred.shape[0]
    ax[output_col].plot([np.min(yt[:ub, output_col]),
                         np.max(yt[:ub, output_col])],
                        [np.min(yt[:ub, output_col]),
                         np.max(yt[:ub, output_col])], c='black')
    ax[output_col].scatter(ytpred[:ub, output_col].flatten(),
                           yt[:ub, output_col].flatten(),
                           s=0.1, alpha=0.16,
                           c='b', marker='x', label='train')
    ax[output_col].scatter(yvpred[:ub, output_col].flatten(),
                           yv[:ub, output_col].flatten(),
                           s=0.1, alpha=0.16,
                           c='g', marker='x', label='valid')
    ax[output_col].scatter(ytestpred[:ub, output_col].flatten(),
                           ytest[:ub, output_col].flatten(),
                           s=0.1, alpha=0.16,
                           c='r', marker='x', label='test')
    ax[output_col].set_title('%s - r**2=%s, %s, %s' %
                             (Y_COLUMNS[output_col], r2_train, r2_valid,
                              r2_test))

    return None


def hplot_one_output(yt, yv, ytest, ytpred, yvpred, ytestpred,
                     title, ax, ax_pos=0):
    """
    Make a histogram for the training (blue), validation (green),
    and test (red) mean squared errors of a specific output measure (or
    of the full network for all outputs if you flatten the input arrays).

    Note: these plots are expensive and take some time to make.

    Parameters
    ----------
    yt         : numpy ndarray
                 training set actual values
    yv         : numpy ndarray
                 validation set actual values
    ytest      : numpy ndarray
                 test set actual values
    ytpred     : numpy ndarray
                 training set predicted values
    yvpred     : numpy ndarray
                 validation set predicted values
    ytestpred  : numpy ndarray
                 test set predicted values
    title      : str
                 the title to display for this plot
    ax         : matplotlib.axes._subplots.AxesSubplot
                 axes object to plot on
    ax_pos     : int, optional
                 the index of the ax instance to put this plot on. Use
                 this parameter if you are plotting multiple subplots

    Returns
    -------
    None
    """
    train_mse = []
    for i, value in enumerate(yt):
        train_mse.append(mean_squared_error([yt[i]],
                                            [ytpred[i]]))
    valid_mse = []
    for i, value in enumerate(yv):
        valid_mse.append(mean_squared_error([yv[i]],
                                            [yvpred[i]]))
    test_mse = []
    for i, value in enumerate(ytest):
        test_mse.append(mean_squared_error([ytest[i]],
                                           [ytestpred[i]]))
    try:
        if ax.numRows:
            ax.hist([train_mse, valid_mse, test_mse], bins=50,
                    stacked=True, label=['train', 'valid', 'test'],
                    log=True)
            ax.set_title('%s' % title)
            ax.legend()
    except AttributeError:
        ax[ax_pos].hist([train_mse, valid_mse, test_mse], bins=50,
                        stacked=True, label=['train', 'valid', 'test'],
                        log=True)
        ax[ax_pos].set_title('%s' % title)
        ax[ax_pos].legend()
