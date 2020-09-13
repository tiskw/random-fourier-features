#!/usr/bin/env python3
#
# Python script for running 
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Nov 16, 2019
#################################### SOURCE START ###################################

import sys
import os

### Add path to PyRFF.py
### The followings are not necessary if you copied PyRFF.py to the current directory
### or other directory which is included in the Python path
current_dir = os.path.dirname(__file__)
module_path = os.path.join(current_dir, "../../source")
sys.path.append(module_path)

import numpy             as np
import matplotlib.pyplot as mpl
import PyRFF             as pyrff


### Main procedure
def main():

    ### Fix seed for random fourier feature calclation
    pyrff.seed(111)

    ### Prepare training data
    Xs_train = np.linspace(0, 3, 21).reshape((21, 1))
    ys_train = np.sin(Xs_train**2)
    Xs_test  = np.linspace(0, 3, 101).reshape((101, 1))
    ys_test  = np.sin(Xs_test**2)

    ### Create classifier instance
    reg = pyrff.RFFRegression(dim_output = 8, std = 0.5)

    ### Train regression with random fourier features
    reg.fit(Xs_train, ys_train)

    ### Conduct prediction for the test data
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
    try   : main()
    except: traceback.print_exc()


#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
