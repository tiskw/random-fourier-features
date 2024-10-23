"""
Overview:
    Automatic hyper parameter tuning using optuna.

Usage:
    optuna_for_california_housing.py [--n_trials <int>] [--seed <int>]
    optuna_for_california_housing.py (-h | --help)

Options:
    --n_trials <int>   Number of trials in hyper parameter tuning. [default: 10]
    --seed <int>       Random seed.                                [default: 111]
    -h, --help         Show this message.
"""

# Import standard libraries.
import subprocess
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import numpy as np
import matplotlib.pyplot as mpl
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing


def generate_boston_housing_dataset(args):
    """
    Create Boston housing dataset instance.
    """
    # Load Boston Housing data from sklearn.
    data = sklearn.datasets.fetch_california_housing()

    # Split data to train and test.
    Xs_train, Xs_test, ys_train, ys_test \
        = sklearn.model_selection.train_test_split(data["data"], data["target"], test_size=0.2, random_state=int(args["--seed"]))

    # Split training data to train and valid (0.25 x 0.8 = 0.2).
    Xs_train, Xs_valid, ys_train, ys_valid \
        = sklearn.model_selection.train_test_split(Xs_train, ys_train, test_size=0.25, random_state=int(args["--seed"]))

    # Data standardization.
    scaler   = sklearn.preprocessing.StandardScaler().fit(Xs_train)
    Xs_train = scaler.transform(Xs_train)
    Xs_valid = scaler.transform(Xs_valid)
    Xs_test  = scaler.transform(Xs_test)

    return (Xs_train, Xs_valid, Xs_test, ys_train, ys_valid, ys_test, data.feature_names)


def main(args):
    """
    Main procedure
    """
    # Define range of hyper parameters to be tuned.
    RANGE_DIM_KERNEL = {"low": 16, "high": 512}
    RANGE_STD_KERNEL = {"low": 1.0E-3, "high": 10.0}
    RANGE_STD_ERROR  = {"low": 1e-4, "high": 0.1, "log": True}

    # Fix seed for random fourier feature calclation
    rfflearn.seed(int(args["--seed"]))

    # Prepare training data
    Xs_train, Xs_valid, Xs_test, ys_train, ys_valid, ys_test, feature_names \
        = generate_boston_housing_dataset(args)

    for mode in ["linear", "gpr"]:

        # Hyper parameter tuning.
        # The returned value `study` contains the results of hyper parameter tuning,
        # including the best parameters (study.best_params) and best model (= study.user_attrs["best_model"]).
        if mode == "linear":
            study = rfflearn.RFFRegressor_tuner(train_set=(Xs_train, ys_train),
                                                 valid_set=(Xs_valid, ys_valid),
                                                 range_dim_kernel=RANGE_DIM_KERNEL,
                                                 range_std_kernel=RANGE_STD_KERNEL,
                                                 n_trials=int(args["--n_trials"]))
        elif mode == "gpr":
            study = rfflearn.RFFGPR_tuner(train_set=(Xs_train, ys_train),
                                          valid_set=(Xs_valid, ys_valid),
                                          range_dim_kernel=RANGE_DIM_KERNEL,
                                          range_std_kernel=RANGE_STD_KERNEL,
                                          range_std_error=RANGE_STD_ERROR,
                                          n_trials=int(args["--n_trials"]))

        # Show the result of the hyper parameter tuning.
        print("- study.best_params:", study.best_params)
        print("- study.best_value:",  study.best_value)
        print("- study.best_model:",  study.user_attrs["best_model"])

        # Run prediction of the best model for the test data.
        best_model = study.user_attrs["best_model"]
        ys_test_p = best_model.predict(Xs_test)
        score_r2   = sklearn.metrics.r2_score(ys_test, ys_test_p)
        print("- R2 score of the best model: ", score_r2)


if __name__ == "__main__":

    # Append path to rfflearn directory.
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Import rfflearn.
    import rfflearn.cpu as rfflearn

    # Run main procedure.
    main(args)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
