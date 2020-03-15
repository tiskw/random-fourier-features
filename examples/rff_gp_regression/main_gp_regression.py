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
  Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    main_rff_svc_for_mnist.py [--kdim <int>] [--kstd <float>] [--estd <float>] [--samples <int>] [--seed <int>]
    main_rff_svc_for_mnist.py -h|--help

Options:
    --kdim <int>     Hyper parameter of RFF SVM (dimention of RFF)  [default: 16]
    --kstd <float>   Hyper parameter of RFF SVM (stdev of RFF)      [default: 5.0]
    --estd <float>   Hyper parameter of RFF SVM (stdev of error)    [default: 1.0]
    --samples <int>  Number of training samples.                    [default: 10000]
    --seed <int>     Random seed.                                   [default: 111]
    -h, --help       Show this message.
"""


import sys
import os

### Add path to PyRFF.py.
### The followings are not necessary if you copied PyRFF.py to the current directory
### or other directory which is included in the Python path.
current_dir = os.path.dirname(__file__)
module_path = os.path.join(current_dir, "../../source")
sys.path.append(module_path)

import time
import pickle
import docopt
import numpy             as np
import sklearn           as skl
import matplotlib.pyplot as mpl
import PyRFF             as pyrff


### Class for measure elasped time using 'with' sentence.
class Timer:

    def __init__(self, message = "", unit = "s", devide_by = 1):
        self.message   = message
        self.time_unit = unit
        self.devide_by = devide_by

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        dt = (time.time() - self.t0) / self.devide_by
        if   self.time_unit == "ms": dt *= 1E3
        elif self.time_unit == "us": dt *= 1E6
        print("%s%f [%s]" % (self.message, dt, self.time_unit))


### Main procedure.
def main(args):

    ### Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    ### Fix seed for random fourier feature calclation.
    # pyrff.seed(args["--seed"])
    pyrff.seed(111)

    ### Create classifier instance.
    gp = pyrff.RFFGaussianProcessRegression(dim_output = args["--kdim"], std_kernel = args["--kstd"], std_error = args["--estd"])

    ### Load training data.
    with Timer("Generating training/testing data: "):
        Xs_train = np.random.randn(args["--samples"], 1)
        ys_train = np.sin(Xs_train**2)
        Xs_test  = np.linspace(-4, 4, 101).reshape((101, 1))
        ys_test  = np.sin(Xs_test**2)

    ### Train SVM with orthogonal random features.
    with Timer("GP learning: "):
        gp.fit(Xs_train, ys_train)

    ### Conduct prediction for the test data
    pred, pstd = gp.predict(Xs_test, return_std = True)

    ### Plot regression results
    mpl.figure(0)
    mpl.title("Regression for function y = sin(x^2) using Gaussian Process w/ RFF")
    mpl.xlabel("X")
    mpl.ylabel("Y")
    mpl.plot(Xs_train, ys_train, ".")
    mpl.plot(Xs_test,  ys_test,  ".")
    mpl.plot(Xs_test,  pred,     "-")
    mpl.fill_between(Xs_test.reshape((Xs_test.shape[0],)),  pred - pstd, pred + pstd, facecolor = "#DDDDDD")
    mpl.legend(["Training data", "Test data GT", "Prediction", "1-sigma area"])
    mpl.grid()
    mpl.show()


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
