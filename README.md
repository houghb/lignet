### Artificial Neural Networks for Modeling Lignin Pyrolysis
----
<p align="center">
   <img src="lignet.png">
</p>

----
Comprehensive models of biomass pyrolysis are needed to develop renewable fuels and chemicals from biomass. Unfortunately, the detailed kinetic schemes required to optimize industrial biomass pyrolysis processes are too computationally expensive to include in models that account for both kinetics and transport within reacting particles. *lignet* is a project to train neural nets (and a decision tree) that can reproduce the results of my detailed ODE-based kinetic model for lignin pyrolysis, *ligpy* [(available here)](https://github.com/houghb/ligpy). The trained neural networks generalize very well, predicting the outputs of the detailed kinetic model with over 99.9% accuracy on new data, and **reduce the computational cost by four orders of magnitude.**

*lignet* was written to generate results for our 2016 paper, **Application of machine learning to pyrolysis reaction networks: reducing model solution time to enable process optimization** (a link to the paper will be added when it is published).

**Cite lignet:** link coming soon

-------
### Software dependencies and license information

**Programming language:**  
Python version 2.7 (https://www.python.org)

**Python packages needed:**
Version numbers that were used in generating our data are listed, but for popular packages (NumPy, sklearn, pandas, etc) it is likely that updated versions will work fine.

- numpy 1.10.4
- matplotlib 1.5.1
- sklearn 0.17
- pandas 0.17.1
- lasagne 0.2
- nolearn 0.7


**License information:**   
*lignet* is licensed under a BSD 2-clause “Simplified” License. The objective behind this choice of licensing is to make the content reproducible and make it useful for as many people as possible. We want to maximize the two-way collaborations with minimum restrictions, so that developers of other projects can easily utilize, patch, improve, and cite this code. Please refer to the [license](https://github.com/houghb/lignet/blob/master/LICENSE) for full details.

-------
### Summary of folder contents

**[gridsearches](https://github.com/houghb/lignet/tree/master/gridsearches)** - Contains the results of gridsearches used to optimize the network architectures for the various neural nets and decision tree, as well as the python scripts to run these gridsearches.

**[lignet](https://github.com/houghb/lignet/tree/master/lignet)** - Contains modules to build, train, and analyze the neural nets and decision tree.

- *[ligpy_benchmarking_files](https://github.com/houghb/lignet/tree/master/lignet/ligpy_benchmarking_files)*: includes files that provide the input arguments required for benchmarking the original *ligpy* model that *lignet* was developed to reproduce.
- *[trained_networks](https://github.com/houghb/lignet/tree/master/lignet/trained_networks)*: contains pickles of the trained neural networks. The trained decision tree is not included in this folder because it is too large to upload to github, but it can be generated (and placed in this folder) by running `create_and_train_decision_tree.py`.
- `EES_plots.ipynb`: an ipython notebook that generates the plots we used in our journal article.
- `analyze_decision_tree.ipynb`: an ipython notebook that explores various learning and validation curves we looked at while choosing the best decision tree architecture (in addition to the results in the gridsearch folder). Parity plots for all of the output measures predicted by the trained decision tree are also shown.
- `analyze_nets.ipynb`: an ipython notebook that explores the trained neural networks.  Learning curves, parity plots, and histograms of the MSE are shown.
- `benchmark_report_summary.txt`: the summarized results of running `benchmarkign.py` that show the average time (of 1000 function calls) for predicting the kinetic model results using the various machine learning estimators compared to the original ODE-based kinetic model.
- `benchmarking.py`: a python script to benchmark the various predictors developed in this package.
- `constant.py`: contains objects with constant values used by other modules.
- `create_and_train.py`: a script that sets up, trains, and pickles nolearn neural net objects.
- `create_and_train_decision_tree.py`: script that trains the decision tree with the optimal architecture (described in `analyze_decision_tree.ipynb`).
- `final_single_output_script.sh`: a bash script that runs `create_and_train.py` with the optimal architectures for all the single nets, as determined by gridsearch results.
- `learning_curve.pkl`: a pickle with the information needed to plot a learning curve for the full_net.
- `learning_curve_full_net.py`: script to generate the learning curve data for the full_net architecture.
- `lignet_utils.py`: contains functions used to analyze the trained neural networks and split the training data into training and test sets.
- `plotting_utils.py`: contains functions for generating the various plots in the ipython notebooks mentioned above.
