#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFRegression class which is a class
# for least square regression using RFF. Interface of RFFRegression is quite close to
# sklearn.linear_model.LinearRegression.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 13, 2020
#################################### SOURCE START ###################################

"""
Overview:
  Train Random Fourier Feature least square regression and plot results.

Usage:
    main_rff_regression.py [--random_type <str>] [--kdim <int>] [--std_kernel <float>]
                           [--rtype <str>] [--n_train <int>] [--n_test <int>] [--seed <int>]
    main_rff_regression.py (-h | --help)

Options:
    --rtype <str>        Random matrix type (rff ot orf).   [default: rff]
    --kdim <int>         Dimention of RFF/ORF.              [default: 8]
    --std_kernel <float> Standard deviation of RFF/ORF.     [default: 0.5]
    --n_train <int>      Number of training data points.    [default: 21]
    --n_test <int>       Number of test data points.        [default: 101]
    --seed <int>         Random seed.                       [default: 111]
    -h, --help           Show this message.
"""

import os
import sys

import docopt
import numpy as np
import matplotlib.pyplot as mpl


### Main procedure
def main(args):

    ### Fix seed for random fourier feature calclation
    rfflearn.seed(111)

    ### Create classifier instance
    if   args["--rtype"] == "rff": reg = rfflearn.RFFRegression(dim_kernel = args["--kdim"], std_kernel = args["--std_kernel"])
    elif args["--rtype"] == "orf": reg = rfflearn.ORFRegression(dim_kernel = args["--kdim"], std_kernel = args["--std_kernel"])
    else                         : raise RuntimeError("Error: 'random_type' must be 'rff' or 'orf'.")

    ### Prepare training data
    with utils.Timer("Creating dataset: "):
        Xs_train = np.linspace(0, 3, args["--n_train"]).reshape((args["--n_train"], 1))
        ys_train = np.sin(Xs_train**2)
        Xs_test  = np.linspace(0, 3, args["--n_test"]).reshape((args["--n_test"], 1))
        ys_test  = np.sin(Xs_test**2)

    ### Train regression with random fourier features
    with utils.Timer("Train regressor: "):
        reg.fit(Xs_train, ys_train)

    ### Conduct prediction for the test data
    with utils.Timer("Prediction: "):
        predict = reg.predict(Xs_test)

    ### Plot regression results
    mpl.figure(0)
    mpl.title("Regression for function y = sin(x^2) with RFF")
    mpl.xlabel("X")
    mpl.ylabel("Y")
    mpl.plot(Xs_train, ys_train, "o")
    mpl.plot(Xs_test,  ys_test,  ".")
    mpl.plot(Xs_test,  predict,  "-")
    mpl.legend(["Training data", "Test data", "Prediction by RFF regression"])
    mpl.grid()
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

    import rfflearn.cpu   as rfflearn
    import rfflearn.utils as utils

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
