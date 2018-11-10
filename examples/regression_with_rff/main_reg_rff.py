#!/usr/bin/env python3
#
# Python script
#
# Author: 
# Date  : Oct 20, 2018
#################################### SOURCE START ###################################

import numpy as np
import matplotlib.pyplot as mpl

import Utils as utils
import Timer as timer
import PyRFF as pyrff

if __name__ == "__main__":
# {{{

    ### Fix seed for random fourier feature calclation
    pyrff.seed(111)

    ### Create classifier instance
    reg = pyrff.RFFRegression(dim_output = 16, std = 0.5)

    ### Load training data
    with timer.Timer("Generating training/test data: "):
        Xs_train = np.linspace(0, 3, 21).reshape((21, 1))
        ys_train = np.sin(Xs_train**2)
        Xs_test  = np.linspace(0, 3, 101).reshape((101, 1))
        ys_test  = np.sin(Xs_test**2)

    ### Train regression with random fourier features
    with timer.Timer("RFF regression learning time: "):
        reg.fit(Xs_train, ys_train)

    ### Calculate score for test data
    with timer.Timer("RFF regression prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
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

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
