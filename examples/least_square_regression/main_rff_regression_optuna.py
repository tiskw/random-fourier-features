#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 08, 2021
#################################### SOURCE START ###################################

"""
Overview:
  Train Random Fourier Feature least square regression and plot results.

Usage:
    main_rff_regression.py [--n_trials <int>] [--seed <int>]
    main_rff_regression.py (-h | --help)

Options:
    --n_trials <int>  Number of trials in hyper parameter tuning.  [default: 300]
    --seed <int>      Random seed.                                 [default: 111]
    -h, --help        Show this message.
"""


import os
import sys

import docopt
import numpy as np
import matplotlib.pyplot as mpl
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing


### Create Boston housing dataset instance.
def generate_boston_housing_dataset():

    ### Load Boston Housing data from sklearn.
    data = sklearn.datasets.load_boston()

    ### Split data to train and valid.
    Xs_train, Xs_valid, ys_train, ys_valid \
        = sklearn.model_selection.train_test_split(data["data"], data["target"], test_size = 0.2, random_state = 111)

    ### Data standardization.
    scaler  = sklearn.preprocessing.StandardScaler().fit(Xs_train)
    X_train = scaler.transform(Xs_train)
    X_valid = scaler.transform(Xs_valid)

    return (Xs_train, Xs_valid, ys_train, ys_valid)


### Main procedure
def main(args):

    ### Fix seed for random fourier feature calclation
    rfflearn.seed(int(args["--seed"]))

    ### Prepare training data
    with utils.Timer("Creating dataset: "):
        Xs_train, Xs_valid, ys_train, ys_valid = generate_boston_housing_dataset()

    ### Hyper parameter tuning.
    with utils.Timer("Hyper parameter tuning: "):
        study = rfflearn.RFFRegression_tuner(train_set = (Xs_train, ys_train), valid_set = (Xs_valid, ys_valid),
                                             range_dim_kernel = (16, 96), range_std_kernel = (1.0E-10, 1.0E-4),
                                             n_jobs = -1, n_trials = int(args["--n_trials"]))

    ### Show the result of the hyper parameter tuning.
    print("  - study.best_params:", study.best_params)
    print("  - study.best_value:",  study.best_value)
    print("  - study.best_model:",  study.user_attrs["best_model"])

    ### Conduct prediction for the test data
    with utils.Timer("Prediction with the best model: "):
        best_model = study.user_attrs["best_model"]
        ys_valid_p = best_model.predict(Xs_valid)
        score_r2   = sklearn.metrics.r2_score(ys_valid, ys_valid_p)

    ### Show the R2 score of the best model.
    print("  - R2 score of the best model: ", score_r2)

    ### Draw figure and save it.
    with utils.Timer("Drawing figure: "):
        mpl.Figure(figsize = (10, 5))
        mpl.scatter(ys_valid_p, ys_valid, alpha = 0.5)
        mpl.plot([0, 50], [0, 50], "--", color = "#666666")
        mpl.title("Regression of Boston Housing Dataset (R2 = %.4f)" % score_r2)
        mpl.xlabel("Predicted price MEDV ($1000s)")
        mpl.ylabel("True price MEDV ($1000s)")
        mpl.grid()
        mpl.savefig("figure_rff_regression_optuna.png")
    print("  - Saved to 'figure_rff_regression_optuna.png'")


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Add path to 'rfflearn/' directory.
    ### The followings are not necessary if you copied 'rfflearn/' to the current
    ### directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    import rfflearn.cpu   as rfflearn
    import rfflearn.utils as utils

    ### Run main procedure.
    main(args)

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
