#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFSVC class which is a class for
# SVM classifier using RFF. Interface of RFFSVC is quite close to sklearn.svm.SVC.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 19, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
  Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    main_rff_svc_for_mnist.py kernel [--input <str>] [--output <str>] [--pcadim <int>] [--kernel <str>] [--gamma <float>] [--C <float>] [--seed <int>]
    main_rff_svc_for_mnist.py rff [--input <str>] [--output <str>] [--pcadim <int>] [--kdim <int>] [--stdev <float>] [--seed <int>]
    main_rff_svc_for_mnist.py orf [--input <str>] [--output <str>] [--pcadim <int>] [--kdim <int>] [--stdev <float>] [--seed <int>]
    main_rff_svc_for_mnist.py -h|--help

Options:
    kernel           Run kernel SVM classifier.
    rff              Run RFF SVM classifier.
    orf              Run ORF SVM classifier.
    --input <str>    Directory path to the MNIST dataset.                [default: ../../dataset/mnist]
    --output <str>   File path to the output pickle file.                [default: result.pickle]
    --pcadim <int>   Output dimention of Principal Component Analysis.   [default: 256]
    --kernel <str>   Hyper parameter of kernel SVM (type of kernel)      [default: rbf]
    --gamma <float>  Hyper parameter of kernel SVM (softness of kernel). [default: auto]
    --C <float>      Hyper parameter of kernel SVM (margin allowance).   [default: 1.0]
    --kdim <int>     Hyper parameter of RFF SVM (dimention of RFF)       [default: 1024]
    --stdev <float>  Hyper parameter of RFF SVM (stdev of RFF)           [default: 0.05]
    --seed <int>     Random seed.                                        [default: 111]
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
import numpy   as np
import sklearn as skl
import PyRFF   as pyrff


### Load train/test image data.
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0


### Load train/test label data.
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)


### PCA analysis for dimention reduction.
def mat_transform_pca(Xs, dim):
    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])


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
    pyrff.seed(args["--seed"])

    ### Create classifier instance.
    if   args["kernel"]: svc = skl.svm.SVC(kernel = args["--kernel"], gamma = args["--gamma"])
    elif args["rff"]   : svc = pyrff.RFFSVC(dim_output = args["--kdim"], std = args["--stdev"], tol = 1.0E-3)
    elif args["orf"]   : svc = pyrff.ORFSVC(dim_output = args["--kdim"], std = args["--stdev"], tol = 1.0E-3)
    else               : exit("Error: First argument must be 'kernel' or 'rff'.")

    ### Load training data.
    with Timer("Loading training data: "):
        Xs_train = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_train_images.npy"))
        ys_train = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_train_labels.npy"))

    ### Load test data.
    with Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy"))
        ys_test = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy"))

    ### Create matrix for principal component analysis.
    with Timer("Calculate PCA matrix: "):
        T = mat_transform_pca(Xs_train, dim = args["--pcadim"])

    ### Train SVM with orthogonal random features.
    with Timer("SVM learning: "):
        svc.fit(Xs_train.dot(T), ys_train)

    ### Calculate score for test data.
    with Timer("SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * svc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)

    ### Save training results.
    with Timer("Saving model: "):
        with open(args["--output"], "wb") as ofp:
            pickle.dump({"svc":svc, "pca":T, "args":args}, ofp)


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
