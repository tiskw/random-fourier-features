#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 29, 2021
#################################### SOURCE START ###################################

"""
Overview:
    Calculation and visualization of feature importance.

Usage:
    main_optuna_and_shap_for_boston_housing.py [--model_type <str>] [--n_trials <int>] [--seed <int>]
    main_optuna_and_shap_for_boston_housing.py (-h | --help)

Options:
    --model_type <str>  Model type (rffgpr or rffregression).       [default: rffgpr]
    --n_trials <int>    Number of trials in hyper parameter tuning. [default: 300]
    --seed <int>        Random seed.                                [default: 111]
    -h, --help          Show this message.
"""

import os
import sys

import docopt
import matplotlib.pyplot as mpl
import sklearn.datasets
import sklearn.inspection
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
    scaler   = sklearn.preprocessing.StandardScaler().fit(Xs_train)
    Xs_train = scaler.transform(Xs_train)
    Xs_valid = scaler.transform(Xs_valid)

    return (Xs_train, Xs_valid, ys_train, ys_valid, data.feature_names)

### Main procedure
def main(args):

    ### Fix seed for random fourier feature calclation
    rfflearn.seed(int(args["--seed"]))

    ### Prepare training data
    Xs_train, Xs_valid, ys_train, ys_valid, feature_names = generate_boston_housing_dataset()

    ### Hyper parameter tuning.
    ### The returned value `study` contains the results of hyper parameter tuning,
    ### including the best parameters (study.best_params) and best model (= study.user_attrs["best_model"]).
    if args["--model_type"] == "rffgpr":
        study = rfflearn.RFFGPR_tuner(
                    train_set = (Xs_train, ys_train),
                    valid_set = (Xs_valid, ys_valid),
                    range_dim_kernel = (16, 256),
                    range_std_kernel = (1.0E-10, 1.0E-3),
                    range_std_error  = (1.0E-5, 1.0E-2),
                    n_trials = int(args["--n_trials"])
                )
    elif args["--model_type"] == "rffregression":
        study = rfflearn.RFFRegression_tuner(
                    train_set = (Xs_train, ys_train),
                    valid_set = (Xs_valid, ys_valid),
                    range_dim_kernel = (16, 128),
                    range_std_kernel = (1.0E-10, 1.0E-3),
                    n_trials = int(args["--n_trials"]),
                    n_jobs = -1
                )
    else: raise NotImplementedError("model type should be 'rffgpr' or 'rffregression'.")

    ### Show the result of the hyper parameter tuning.
    print("- study.best_params:", study.best_params)
    print("- study.best_value:",  study.best_value)
    print("- study.best_model:",  study.user_attrs["best_model"])

    ### Conduct prediction for the test data
    best_model = study.user_attrs["best_model"]
    ys_valid_p = best_model.predict(Xs_valid)
    score_r2   = sklearn.metrics.r2_score(ys_valid, ys_valid_p)
    print("- R2 score of the best model: ", score_r2)

    ### Calculate feature importance (SHAP and permutation importance).
    shap_values = rfflearn.shap_feature_importance(best_model, Xs_valid)
    perm_values = rfflearn.permutation_feature_importance(best_model, Xs_valid, ys_valid)

    ### Draw regression result.
    mpl.figure(0)
    mpl.scatter(ys_valid_p, ys_valid, alpha = 0.5)
    mpl.plot([0, 50], [0, 50], "--", color = "#666666")
    mpl.title("Regression of Boston Housing Dataset (R2 = %.4f)" % score_r2)
    mpl.xlabel("Predicted price MEDV ($1000s)")
    mpl.ylabel("True price MEDV ($1000s)")
    mpl.grid()

    ### Visualize SHAP importance.
    mpl.figure(1)
    rfflearn.shap_plot(shap_values, Xs_valid, feature_names, show = False)

    ### Visualize permurtation importance.
    mpl.figure(2)
    rfflearn.permutation_plot(perm_values, feature_names, show = False)

    ### Show all figures.
    mpl.show()

if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Add path to 'rfflearn/' directory.
    ### The followings are not necessary if you copied 'rfflearn/' to the current
    ### directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    import rfflearn.cpu as rfflearn

    ### Run main procedure.
    main(args)

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
