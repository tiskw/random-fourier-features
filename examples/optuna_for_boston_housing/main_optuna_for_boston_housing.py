#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 10, 2021
#################################### SOURCE START ###################################

"""
Overview:
    Automatic hyper parameter tuning using optuna.

Usage:
    main_optuna_for_boston_housing.py [--n_trials <int>] [--seed <int>] [--visualize]
    main_optuna_for_boston_housing.py (-h | --help)

Options:
    --n_trials <int>    Number of trials in hyper parameter tuning. [default: 600]
    --seed <int>        Random seed.                                [default: 111]
    --visualize         Enable visualization.
    -h, --help          Show this message.
"""


import os
import subprocess
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

    return (Xs_train, Xs_valid, ys_train, ys_valid, data.feature_names)


### Main procedure
def main(args):

    ### Define range of hyper parameters to be tuned.
    RANGE_DIM_KERNEL = (16, 128)
    RANGE_STD_KERNEL = (1.0E-10, 1.0E-3)

    ### Fix seed for random fourier feature calclation
    rfflearn.seed(int(args["--seed"]))

    ### Prepare training data
    Xs_train, Xs_valid, ys_train, ys_valid, feature_names = generate_boston_housing_dataset()

    ### Hyper parameter tuning.
    ### The returned value `study` contains the results of hyper parameter tuning,
    ### including the best parameters (study.best_params) and best model (= study.user_attrs["best_model"]).
    study = rfflearn.RFFRegression_tuner(train_set = (Xs_train, ys_train),
                                         valid_set = (Xs_valid, ys_valid),
                                         range_dim_kernel = RANGE_DIM_KERNEL,
                                         range_std_kernel = RANGE_STD_KERNEL,
                                         n_trials = int(args["--n_trials"]),
                                         n_jobs = -1)

    ### Show the result of the hyper parameter tuning.
    print("- study.best_params:", study.best_params)
    print("- study.best_value:",  study.best_value)
    print("- study.best_model:",  study.user_attrs["best_model"])

    ### Run prediction of the best model for the test data.
    best_model = study.user_attrs["best_model"]
    ys_valid_p = best_model.predict(Xs_valid)
    score_r2   = sklearn.metrics.r2_score(ys_valid, ys_valid_p)
    print("- R2 score of the best model: ", score_r2)

    ### The following code is visualization of the tuning process.
    if not args["--visualize"]: return

    ### Create plotting contents, where
    ###  - xs: parameter 1 (dim_kernel of RFFRegression)
    ###  - ys: parameter 2 (std_kernel of RFFRegression)
    ###  - zs: R2 score of the model.
    xs = np.array([trial.params["dim_kernel"] for trial in study.get_trials()])
    ys = np.array([trial.params["std_kernel"] for trial in study.get_trials()])
    zs = np.array([max(0, trial.value)        for trial in study.get_trials()])

    ### Create output directory if not exists.
    os.makedirs("figures", exist_ok = True)

    for index in range(len(xs)):
        mpl.figure(figsize = (5, 4))
        mpl.title("Behavior of Hyper Parameter Tuning (step %3d/%3d)" % (index + 1, len(xs) + 1))
        mpl.xlabel("dim_kernel")
        mpl.ylabel("std_kernel")
        mpl.scatter(xs[:index+1], ys[:index+1], 20, zs[:index+1], cmap = "jet", vmin = 0.0, vmax = 1.0)
        mpl.xlim(*RANGE_DIM_KERNEL)
        mpl.ylim(*RANGE_STD_KERNEL)
        mpl.yscale("log")
        mpl.colorbar()
        mpl.grid()
        mpl.tight_layout()
        mpl.savefig("figures/hyper_parameter_search_%04d.png" % index)
        mpl.close()

    ### Run command for creating the animation git.
    command = "convert -delay 8 figures/hyper_parameter_search_*.png figures/hyper_parameter_search.gif"
    subprocess.run(command, shell = True)
    print("figures/hyper_parameter_search.gif")


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
