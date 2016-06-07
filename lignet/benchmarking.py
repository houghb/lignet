"""
This module performs benchmarking to compare the computational costs of
generating predictions using the ligpy model, trained neural nets for the
full set of outputs and collections of individual outputs, and the trained
decision tree estimator.
"""
import sys
sys.path.append('../../../ligpy/ligpy')
import os
import copy
from subprocess import call

import numpy as np
import cPickle as pickle

import ligpy_utils as utils
import ddasac_utils as ddasac
from analysis_tools import load_results, generate_report
from constants import MW
from lignet_utils import gen_train_test, load_nets
from create_and_train import EarlyStopping


# Pre-load the testing data and machine learning estimators
x_train, x_test, y_train, y_test, x_scaler, y_scaler = gen_train_test()
nets = load_nets('trained_networks/updated*')
full_net = load_nets('trained_networks/full*')
with open('trained_networks/decision_tree.pkl', 'rb') as pkl:
    dtr_full = pickle.load(pkl)[0]
with open('ligpy_benchmarking_files/ligpy_args.txt', 'rb') as args:
    ligpy_args = args.readlines()
    ligpy_args = ligpy_args[1:]

# reset the random seed generator
np.random.seed()
rand_sample = np.random.randint(0, 199999)
# the row of input data to use in tests
rand_input = x_train[rand_sample:rand_sample+1, :]


def predict_full_net(input_data=rand_input, net=full_net['all']):
    """
    Predict the output measures using the network trained on all 30
    output measures at once.

    Parameters
    ----------
    input_data : numpy.ndarray, optional
                 an array of input values to predict.  This can be a single
                 row or many rows.
    net        : nolearn.lasagne.base.NeuralNet, optional
                 the trained neural net for all 30 output measures
    Returns
    -------
    predicted : numpy.ndarray
                an array of the predicted values
    """
    return net.predict(input_data)


def predict_single_net(input_data=rand_input, net=nets[5]):
    """ 
    Predict the value for a single output measure.

    Parameters
    ----------
    input_data : numpy.ndarray, optional
                 an array of input values to predict.  This can be a single
                 row or many rows.
    net        : nolearn.lasagne.base.NeuralNet, optional
                 a trained neural net for a single output measure
    Returns
    -------
    predicted : numpy.ndarray
                an array of the predicted values
    """
    return net.predict(input_data)


def predict_30_single_nets(input_data=rand_input, nets=nets):
    """
    Predict the output measures using 30 individually trained neural nets.

    Parameters
    ----------
    input_data : numpy.ndarray, optional
                 an array of input values to predict.  This can be a single
                 row or many rows.
    nets       : dict, optional
                 dictionary with the trained neural nets for all 30 output
                 measures

    Returns
    -------
    predicted : numpy.ndarray
                an array of the predicted values
    """

    predicted = np.zeros((input_data.shape[0], 30))
    for i in nets.keys():
        predicted[:, i] = nets[i].predict(input_data).ravel()

    return predicted


def predict_decision_tree(input_data=rand_input, tree=dtr_full):
    """
    Predict the output measures using a decision tree trained on all 30
    output measures at once.

    Parameters
    ----------
    input_data  : numpy.ndarray, optional
                  an array of input values to predict.  This can be a single
                  row or many rows.
    tree        : sklearn.tree.tree.DecisionTreeRegressor, optional
                  the trained decision tree for all 30 output measures
    Returns
    -------
    predicted : numpy.ndarray
                an array of the predicted values
    """
    return tree.predict(input_data)


def setup_predict_ligpy(end_time=end_time, output_time_step=output_time_step,
                  cool_time=cool_time, initial_T=initial_T,
                  heat_rate=heat_rate, maximum_T=maximum_T, plant=plant):
    """
    Create the proper environment to run predict_ligpy() and set up the
    kinetic model.

    Parameters
    ----------
    standard arguments passed to `ligpy.py`

    Returns
    -------
    standard arguments for `ddasac.run_ddasac()`
    """
    call('cp ligpy_benchmarking_files/sa_compositionlist.dat '
         '../../../ligpy/ligpy/data/compositionlist.dat;', shell=True
         )

    absolute_tolerance = float(1e-10)
    relative_tolerance = float(1e-8)

    # These are the files and paths that will be referenced in this program:
    (file_completereactionlist, file_completerateconstantlist,
     file_compositionlist) = utils.set_paths()
    working_directory = 'results_dir'
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    # pickle the arguments used for this program to reference during analysis
    prog_params = [end_time, output_time_step, initial_T, heat_rate, maximum_T,
                   absolute_tolerance, relative_tolerance, plant, cool_time]
    with open('%s/prog_params.pkl' % working_directory, 'wb') as pkl:
        pickle.dump(prog_params, pkl)

    # Get lists of all the species in the kinetic scheme and their indices
    specieslist = utils.get_specieslist(file_completereactionlist)
    # Get kmatrix
    kmatrix = utils.build_k_matrix(file_completerateconstantlist)

    # Set the initial composition of the lignin polymer
    PLIGC_0, PLIGH_0, PLIGO_0 = utils.define_initial_composition(
        file_compositionlist, plant)

    # Set the initial conditions for the DDASAC solver
    y0_ddasac = np.zeros(len(specieslist))
    y0_ddasac[:3] = [PLIGC_0, PLIGH_0, PLIGO_0]

    return (file_completereactionlist, kmatrix, working_directory,
                      y0_ddasac, specieslist, absolute_tolerance,
                      relative_tolerance, initial_T, heat_rate, end_time,
                      maximum_T, output_time_step, cool_time)


def teardown_predict_ligpy():
    """
    Clean up after running predict_ligpy().

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    call('rm -rf bsub.c bsub.o ddat.in fort.11 f.out greg10.in jacobian.c '
         'jacobian.o model.c model.o net_rates.def parest rates.def '
         'results_dir/', shell=True)


def get_random_ligpy_args():
    """
    Get the arguments from a random row of ligpy_args to pass to
    predict_ligpy.

    Returns
    -------
    end_time
    output_time_step
    cool_time
    initial_T
    heat_rate
    maximum_T
    plant
    """
    rand_index = np.random.randint(0, 249999)
    args = ligpy_args[rand_index]
    end_time = float(args.split(' ')[0])
    output_time_step = float(args.split(' ')[1])
    cool_time = int(args.split(' ')[2])
    initial_T = float(args.split(' ')[3])
    heat_rate = float(args.split(' ')[4])
    maximum_T = float(args.split(' ')[5])
    plant = str(args.split(' ')[8]).rstrip()

    return (end_time, output_time_step, cool_time, initial_T, heat_rate,
            maximum_T, plant)


# the set of arguments for the first predict_ligpy() call
(end_time, output_time_step, cool_time, initial_T, heat_rate,
    maximum_T, plant) = get_random_ligpy_args()


def predict_ligpy(file_completereactionlist, kmatrix, working_directory,
                      y0_ddasac, specieslist, absolute_tolerance,
                      relative_tolerance, initial_T, heat_rate, end_time,
                      maximum_T, output_time_step, cool_time):
    """
    This function is a modified version of `ligpy.py` in the `ligpy` package.
    It sets up and solves the ODE model for lignin pyrolysis, then calculates
    the set of outputs that are predicted by the machine learning models
    developed in `lignet`.

    Parameters
    ----------
    standard arguments passed to `ligpy.py`

    Returns
    -------
    None
    """
    # Solve the model with DDASAC
    ddasac.run_ddasac(file_completereactionlist, kmatrix, working_directory,
                      y0_ddasac, specieslist, absolute_tolerance,
                      relative_tolerance, initial_T, heat_rate, end_time,
                      maximum_T, output_time_step, cool_time)

    # Load the program parameters and results from the selected folder
    (end_time, output_time_step, initial_T, heating_rate, max_T, atol, rtol,
     plant, cool_time, y, t, T, specieslist, speciesindices,
     indices_to_species) = load_results('.')

    # create a new matrix of mass fractions (instead of concentrations)
    m = copy.deepcopy(y)
    for species in specieslist:
        # make an array of mass concentration (g/L)
        m[:, speciesindices[species]] = (y[:, speciesindices[species]] *
                                         MW[species][0])

    generate_report(speciesindices, specieslist, y, m, t, 'temp_result')


if __name__ == '__main__':
    import timeit

    tot_time = (timeit.timeit('predict_full_net()',
                setup='from __main__ import predict_full_net',
                number=1000))
    print('predict_full_net: %s sec per call' % (tot_time/1000))

    tot_time = (timeit.timeit('predict_single_net()',
                setup='from __main__ import predict_single_net',
                number=1000))
    print('predict_single_net: %s sec per call' % (tot_time/1000))

    tot_time = (timeit.timeit('predict_30_single_nets()',
                setup='from __main__ import predict_30_single_nets',
                number=1000))
    print('predict_30_single_nets: %s sec per call' % (tot_time/1000))

    tot_time = (timeit.timeit('predict_decision_tree()',
                setup='from __main__ import predict_decision_tree',
                number=1000))
    print('predict_decision_tree: %s sec per call' % (tot_time/1000))

    num_runs = 1000
    times = np.zeros(num_runs)
    for i in range(0, num_runs):
        setup_predict_ligpy()
        tot_time = (timeit.timeit('predict_ligpy()',
                    setup='from __main__ import predict_ligpy',
                    number=1))
        times[i] = tot_time
        teardown_predict_ligpy()
    print('predict_ligpy: %s sec per call' % times.mean())

