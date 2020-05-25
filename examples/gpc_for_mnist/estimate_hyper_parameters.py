#!/usr/bin/env python3
#
# Estimate hyper parameters for Gaussian process. The folowing parameters will be estimated:
#   - scale factor of the RBF kernel (scale)
#   - standers deviation of the RBF kernel (s_k)
#   - standers deviation of the measurement error (s_e)
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : May 25, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
    Estimate hyper parameters for Gaussian process. The folowing parameters will be estimated:
      - scale factor of the RBF kernel (scale)
      - standers deviation of the RBF kernel (s_k)
      - standers deviation of the measurement error (s_e)

Usage:
    estimate_hyper_parameters.py [--input <str>] [--output <str>] [--pcadim <int>]
                                 [--scale <float>] [--std_kernel <float>] [--std_error <float>]
                                 [--epoch <int>] [--batch_size <int>] [--lr <float>] [--seed <int>]
    estimate_hyper_parameters.py -h|--help

Options:
    --input <str>        Directory path to the MNIST dataset.               [default: ../../dataset/mnist]
    --output <str>       File path to the output pickle file.               [default: None]
    --pcadim <int>       Output dimention of Principal Component Analysis.  [default: 128]
    --scale <float>      Hyper parameter of Gaussian process.               [default: 1.0]
    --std_kernel <float> Hyper parameter of Gaussian process.               [default: 0.1]
    --std_error <float>  Hyper parameter of Gaussian process.               [default: 0.5]
    --epoch <int>        Number of epochs.                                  [default: 10]
    --batch_size <int>   Size of batch.                                     [default: 256]
    --lr <float>         Learning rate of the momentum SGD optimizer.       [default: 1.0E-4]
    --seed <int>         Random seed.                                       [default: 111]
    -h, --help           Show this message.
"""

import os
import pickle
import sys
import time

import docopt


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

    ### Load training data.
    Xs_train = vectorise_MNIST_images("../../dataset/mnist/MNIST_train_images.npy")
    ys_train = vectorise_MNIST_labels("../../dataset/mnist/MNIST_train_labels.npy")

    ### Create matrix for principal component analysis.
    T = mat_transform_pca(Xs_train, dim = args["--pcadim"])

    ### Run hyper parameter estimation.
    init_val_s_k = args["--std_kernel"]
    init_val_s_e = args["--std_error"]
    init_val_sca = args["--scale"]
    gpkpe = pyrff.GPKernelParameterEstimator(args["--epoch"], args["--batch_size"], args["--lr"], init_val_s_k, init_val_s_e, init_val_sca)
    s_k, s_e, scale = gpkpe.fit(Xs_train.dot(T), ys_train)

    ### Save results.
    if args["--output"]:
        with open(args["--output"], "wb") as ofp:
            pickle.dump({"pca":T, "s_k":s_k, "s_e":s_e, "scale":scale, "args":args}, ofp)


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Add path to PyRFF.py.
    ### The followings are not necessary if you copied PyRFF.py to the current directory
    ### or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../source")
    sys.path.append(module_path)

    import numpy as np
    import sklearn as skl
    import tensorflow as tf

    import PyRFF_GPU as pyrff
    import utils

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
