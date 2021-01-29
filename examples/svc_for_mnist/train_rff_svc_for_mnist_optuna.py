#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFSVC class which is a class for
# SVM classifier using RFF. Interface of RFFSVC is quite close to sklearn.svm.SVC.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 08, 2021
##################################################### SOURCE START #####################################################

"""
Overview:
  Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    main_rff_svc_for_mnist.py [--input <str>] [--output <str>] [--pcadim <int>] [--n_trials <int>] [--seed <int>] [--cpus <int>]
    main_rff_svc_for_mnist.py (-h | --help)

Options:
    --input <str>     Directory path to the MNIST dataset.        [default: ../../dataset/mnist]
    --output <str>    File path to the output pickle file.        [default: result.pickle]
    --pcadim <int>    Output dimention of PCA.                    [default: 128]
    --n_trials <int>  Number of trials in hyper parameter tuning. [default: 100]
    --seed <int>      Random seed.                                [default: 111]
    --cpus <int>      Number of available CPUs.                   [default: -1]
    -h, --help        Show this message.
"""

import os
import pickle
import sys

import docopt
import numpy as np

### Load train/test image data.
def vectorise_MNIST_images(filepath):

    Xs = np.load(filepath) / 255.0
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])])

### Load train/test label data.
def vectorise_MNIST_labels(filepath):

    return np.load(filepath)

### PCA analysis for dimention reduction.
def mat_transform_pca(Xs, dim):

    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])

### Main procedure.
def main(args):

    ### Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    ### Fix seed for random fourier feature calclation.
    rfflearn.seed(int(args["--seed"]))

    ### Load training data.
    with utils.Timer("Loading training data: "):
        Xs_train = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_train_images.npy"))
        ys_train = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_train_labels.npy"))

    ### Load test data.
    with utils.Timer("Loading test data: "):
        Xs_valid = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy"))
        ys_valid = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy"))

    ### Create matrix for principal component analysis.
    with utils.Timer("Calculate PCA matrix and transform train/valid data: "):
        T = mat_transform_pca(Xs_train, dim = int(args["--pcadim"]))
        Xs_train = Xs_train.dot(T)
        Xs_valid = Xs_valid.dot(T)

    study = rfflearn.RFFSVC_tuner(train_set = (Xs_train, ys_train), valid_set = (Xs_valid, ys_valid),
                                  tol = 1.0E-3, verbose = 0, n_jobs = int(args["--cpus"]), n_trials = int(args["--n_trials"]))

    print("study.best_params:", study.best_params)
    print("study.best_value:", study.best_value)
    print("study.best_trial:", study.best_trial)

    ### Save training results.
    with utils.Timer("Saving model: "):
        with open(args["--output"], "wb") as ofp:
            pickle.dump({"svc":study.user_attrs["best_model"], "pca":T, "args":args}, ofp)

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

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
