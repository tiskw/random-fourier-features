#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFGPC class which is a class for
# Gaussian process classifier using RFF. Interface of RFFGPC is quite close to
# sklearn.gaussian_process.GaussianProcessClassifier.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
  Train Gaussian process classifire with RFF/OFR.
  Before running this script, make sure to create MNIST dataset.

Usage:
    main_rff_gpc_for_mnist.py cpu [--input <str>] [--output <str>] [--pcadim <int>] [--rtype <str>]
                                  [--kdim <int>] [--std_kernel <float>] [--std_error <float>] [--seed <int>]
    main_rff_gpc_for_mnist.py gpu [--input <str>] [--output <str>] [--pcadim <int>] [--rtype <str>]
                                  [--kdim <int>] [--std_kernel <float>] [--std_error <float>] [--seed <int>]
    main_rff_gpc_for_mnist.py (-h | --help)

Options:
    --input <str>        Directory path to the MNIST dataset.                [default: ../../dataset/mnist]
    --output <str>       File path to the output pickle file.                [default: result.pickle]
    --pcadim <int>       Output dimention of Principal Component Analysis.   [default: 128]
    --rtype <str>        Type of random matrix (rff or orf).                 [default: rff]
    --kdim <int>         Hyper parameter of RFF SVM (dimention of RFF).      [default: 128]
    --std_kernel <float> Hyper parameter of RFF SVM (stdev of RFF).          [default: 0.05]
    --std_error <float>  Hyper parameter of RFF SVM (stdev of RFF).          [default: 0.05]
    --seed <int>         Random seed.                                        [default: 111]
    --cpus <int>         Number of available CPUs.                           [default: -1]
    -h, --help           Show this message.
"""

import os
import pickle
import sys

import docopt
import numpy   as np
import sklearn as skl


### Load train/test image data.
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0 - 0.5


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
    rfflearn.seed(args["--seed"])

    ### Create classifier instance.
    if   args["--rtype"] == "rff": gpc = rfflearn.RFFGPC(args["--kdim"], args["--std_kernel"], args["--std_error"])
    elif args["--rtype"] == "orf": gpc = rfflearn.ORFGPC(args["--kdim"], args["--std_kernel"], args["--std_error"])
    else                         : raise RuntimeError("Error: 'random_type' must be 'rff' or 'orf'.")

    ### Load training data.
    with utils.Timer("Loading training data: "):
        Xs_train = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_train_images.npy"))
        ys_train = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_train_labels.npy"))

    ### Load test data.
    with utils.Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy"))
        ys_test = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy"))

    ### Create matrix for principal component analysis.
    with utils.Timer("Calculate PCA matrix: "):
        T = mat_transform_pca(Xs_train, dim = args["--pcadim"])

    ### Train SVM with orthogonal random features.
    with utils.Timer("GPC learning: "):
        gpc.fit(Xs_train.dot(T), ys_train)

    ### Calculate score for test data.
    with utils.Timer("GPC prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * gpc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)

    ### Save training results.
    with utils.Timer("Saving model: "):
        with open(args["--output"], "wb") as ofp:
            pickle.dump({"gpc": gpc, "pca": T, "args": args}, ofp)


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Add path to 'rfflearn/' directory.
    ### The followings are not necessary if you copied 'rfflearn/' to the current
    ### directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    if   args["cpu"]: import rfflearn.cpu as rfflearn
    elif args["gpu"]: import rfflearn.gpu as rfflearn
    import rfflearn.utils as utils

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
