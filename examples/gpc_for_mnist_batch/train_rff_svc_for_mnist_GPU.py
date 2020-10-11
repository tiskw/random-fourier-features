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
  Train Random Fourier Feature Gaussian process.
  Before running this script, make sure to create MNIST dataset.

Usage:
    main_rff_svc_for_mnist_GPU.py [--input <str>] [--output <str>] [--pcadim <int>] [--kdim <int>]
                                  [--std_kernel <float>] [--std_error <float>] [--steps <int>]
                                  [--batch_size <int>] [--seed <int>] [--save]
    main_rff_svc_for_mnist_GPU.py -h|--help

Options:
    --input <str>         Directory path to the MNIST dataset.       [default: ../../dataset/mnist]
    --output <str>        File path to the output pickle file.       [default: result.pickle]
    --pcadim <int>        Output dimention of PCA.                   [default: 128]
    --kdim <int>          Hyper parameter of RFF SVM (dim of RFF).   [default: 256]
    --std_kernel <float>  Hyper parameter of RFF SVM (stdev of RFF). [default: 0.01]
    --std_error <float>   Hyper parameter of RFF SVM (stdev of RFF). [default: 0.5]
    --steps <int>         Number of iterations.                      [default: 1000]
    --batch_size <int>    Size of batch.                             [default: 1024]
    --seed <int>          Random seed.                               [default: None]
    -h, --help            Show this message.
"""

import sys
import os
import pickle
import random

import docopt
import numpy   as np
import sklearn as skl
import tensorflow as tf


### Load train/test image data.
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28**2, )) for n in range(Xs.shape[0])]) / 255.0 - 0.5
    # return np.array([Xs[n, :, :].reshape((32**2 * 3, )) for n in range(Xs.shape[0])]) / 255.0 - 0.5


### Load train/test label data.
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)


### PCA analysis for dimention reduction.
def mat_transform_pca(Xs, dim):
    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])


@tf.function
def train(X, y, W, a, S, s_e):

    ### Generate random matrix W and identity matrix I on CPU.
    I = tf.eye(S.shape[0], dtype = tf.float64)

    Z  = tf.matmul(X, W)
    Ft = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
    F  = tf.transpose(Ft)
    P  = tf.matmul(F, Ft)
    s  = (s_e)**2

    T = tf.linalg.pinv(S)
    M = tf.matmul(S, I - tf.matmul(P, tf.linalg.pinv(P + s * T))) / s
    m = tf.matmul(Ft, a)

    da = tf.matmul(M, tf.matmul(F, y - m))
    dS = - tf.matmul(M, tf.matmul(P, S))

    return (a + da, S + dS)


def predict(X_cpu, W, a, S):

    ### Move matrix to GPU.
    X = tf.constant(X_cpu, dtype = tf.float64)

    Z  = tf.matmul(X, W)
    Ft = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
    F  = tf.transpose(Ft)

    m = tf.matmul(Ft, a)
    V = tf.matmul(tf.matmul(Ft, S), F)

    return (m.numpy(), np.diag(V.numpy()), V.numpy())


### Main procedure.
def main(args):

    ### Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    ### Load training data.
    Xs_train = vectorise_MNIST_images("../../dataset/mnist/MNIST_train_images.npy")
    ys_train = vectorise_MNIST_labels("../../dataset/mnist/MNIST_train_labels.npy")

    ### Load test data.
    Xs_test = vectorise_MNIST_images("../../dataset/mnist/MNIST_test_images.npy")
    ys_test = vectorise_MNIST_labels("../../dataset/mnist/MNIST_test_labels.npy")

    ### Create matrix for principal component analysis.
    T = mat_transform_pca(Xs_train, dim = args["--pcadim"])

    W_cpu = args["--std_kernel"] * np.random.randn(args["--pcadim"], args["--kdim"])
    ys_train_oh = np.eye(int(np.max(ys_train) + 1))[ys_train]

    X_cpu = Xs_train.dot(T)
    y_cpu = ys_train_oh

    ### Move matrices to GPU.
    W = tf.constant(W_cpu, dtype = tf.float64)
    a = tf.constant(np.zeros((2 * args["--kdim"], ys_train_oh.shape[1])), dtype = tf.float64)
    S = tf.constant(np.eye(2 * args["--kdim"]), dtype = tf.float64)

    batch_size = int(args["--batch_size"])
    total_size = Xs_train.shape[0]

    for step in range(args["--steps"]):

        mask = [True] * batch_size + [False] * (total_size - batch_size)
        random.shuffle(mask)
        mask = np.array(mask)

        X = tf.constant(X_cpu[mask, :], dtype = tf.float64)
        y = tf.constant(y_cpu[mask, :], dtype = tf.float64)

        ### Calculate score for test data.
        a, S = train(X, y, W, a, S, args["--std_error"])

        m, v, V = predict(Xs_test.dot(T), W, a, S)
        score = 100.0 * np.mean(np.argmax(m, axis = 1) == ys_test)
        print("Score = %.2f [%%]" % score)
        sys.stdout.flush()

    ### Save training results.
    with open(args["--output"], "wb") as ofp:
        pickle.dump({"W":W, "mean":a, "cov":S, "pca":T, "args":args}, ofp)


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Set random seed
    random.seed(args["--seed"])
    np.random.seed(args["--seed"])
    tf.random.set_seed(args["--seed"])

    ### Run main procedure.
    main(args)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
