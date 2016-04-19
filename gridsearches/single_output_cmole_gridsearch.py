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

# get the arguments passed by the user from CLI
# first argument passed is the column of the output measure to train
output_col = sys.argv[1]

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
            hidden0_num_units=10,
            hidden0_nonlinearity=nonlinearities.sigmoid,
            output_num_units=1,
            output_nonlinearity=nonlinearities.linear,
            regression=True,
            update_learning_rate=0.9,
            verbose=1,
            max_epochs=400,
            update=lasagne.updates.adagrad,
            )

# Set up the gridsearch
param_grid = {'hidden0_num_units': range(4, 33),
              'update_learning_rate': [x/10.0 for x in range(1, 21)]
              }
grid_search = GridSearchCV(net, param_grid, verbose=0, n_jobs=20,
                           pre_dispatch='2*n_jobs',
                           scoring='mean_squared_error')

# select only the output measure you want to train
y_train2 = y_train[:, output_col]
grid_search.fit(x_train[:6000, :], y_train2[:6000])

# Pickle
with open('gs2_%s_ann_objects.pkl' % y_columns[output_col], 'wb') as pkl:
    pickle.dump([net, param_grid, grid_search], pkl)

# Print the scores to a file
with open('gs2_%s_scores.txt' % y_columns[output_col], 'w+') as report:
    # this shows you the scores for all the different searches
    for entry in grid_search.grid_scores_:
        print >>report, entry

    # quickly show which was the estimator with the best score
    print >>report, 'The best estimator was:\n'
    print >>report, grid_search.best_params_
