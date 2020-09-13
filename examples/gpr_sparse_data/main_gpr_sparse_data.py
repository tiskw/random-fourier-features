#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFGaussianProcess
# class which is a class for Gaussian process regressor using RFF.
# Interface of RFFSVC is quite close to sklearn.gaussian_process.GaussianProcessRegressor.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : March 15, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
  An example of Gaussian Process Regression with Random Fourier Feature (RFF GPR).
  As a comparison with the noemal GPR, this script has a capability to run the normal GPR under the same condition with RFF GPR.

Usage:
    main_gpr_sparse_data.py kernel [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    main_gpr_sparse_data.py rff [--kdim <int>] [--std_kernel <float>] [--std_error <float>]
                                [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    main_gpr_sparse_data.py -h|--help

Options:
    kernel               Run normal Gaussian Process.
    rff                  Run Gaussian process with Random Fourier Features.
    --kdim <int>         Hyper parameter of RFF SVM (dimention of RFF). [default: 16]
    --std_kernel <float> Hyper parameter of RFF SVM (stdev of RFF).     [default: 5.0]
    --std_error <float>  Hyper parameter of RFF SVM (stdev of error).   [default: 1.0]
    --n_train <int>      Number of training samples.                    [default: 10000]
    --n_test <int>       Number of test samples.                        [default: 101]
    --no_pred_std        Run standard deviation prediction.
    --seed <int>         Random seed.                                   [default: 111]
    -h, --help           Show this message.
"""


import os
import pickle
import sys

import docopt
import matplotlib.pyplot as mpl
import numpy             as np
import sklearn           as skl
import sklearn.gaussian_process


### Main procedure.
def main(args):

    ### Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    ### Fix seed for random fourier feature calclation.
    pyrff.seed(args["--seed"])

    ### Create classifier instance.
    if args["kernel"]:
        kf = skl.gaussian_process.kernels.RBF(1.0 / args["--kstd"]) + skl.gaussian_process.kernels.WhiteKernel(args["--estd"])
        gpr = skl.gaussian_process.GaussianProcessRegressor(kernel = kf, random_state = args["--seed"])
    elif args["rff"]:
        gpr = pyrff.RFFGPR(args["--kdim"], args["--std_kernel"], std_error = args["--std_error"])

    ### Load training data.
    with utils.Timer("Generating training/testing data: "):
        Xs_train = np.random.randn(args["--n_train"], 1)
        ys_train = np.sin(Xs_train**2)
        Xs_test  = np.linspace(-4, 4, args["--n_test"]).reshape((args["--n_test"], 1))
        ys_test  = np.sin(Xs_test**2)

    ### Train SVM with orthogonal random features.
    with utils.Timer("GPR learning: ", unit = "ms"):
        gpr.fit(Xs_train, ys_train)

    ### Conduct prediction for the test data.
    if args["--no_pred_std"]:
        with utils.Timer("GPR inference: ", unit = "us", devide_by = args["--n_test"]):
            pred = gpr.predict(Xs_test)
            pstd = None
    else:
        with utils.Timer("GPR inference: ", unit = "us", devide_by = args["--n_test"]):
            pred, pstd = gpr.predict(Xs_test, return_std = True)
            pred = pred.reshape((pred.shape[0],))
            pstd = pstd.reshape((pstd.shape[0],))

    ### Plot regression results.
    mpl.figure(0)
    mpl.title("Regression for function y = sin(x^2) using Gaussian Process w/ RFF")
    mpl.xlabel("X")
    mpl.ylabel("Y")
    mpl.plot(Xs_train, ys_train, ".")
    mpl.plot(Xs_test,  ys_test,  ".")
    mpl.plot(Xs_test,  pred,     "-")
    if pstd is not None:
        mpl.fill_between(Xs_test.reshape((Xs_test.shape[0],)),  pred - pstd, pred + pstd, facecolor = "#DDDDDD")
    mpl.legend(["Training data", "Test data GT", "Prediction", "1-sigma area"])
    mpl.grid()
    mpl.show()


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Add path to PyRFF.py.
    ### The followings are not necessary if you copied PyRFF.py to the current directory
    ### or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../source")
    sys.path.append(module_path)

    import PyRFF as pyrff
    import utils

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
