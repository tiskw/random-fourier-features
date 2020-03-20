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
    main_rff_svc_for_mnist_GPU.py [--input <str>] [--output <str>] [--pcadim <int>] [--kdim <int>]
                                  [--std_kernel <float>] [--std_error <float>] [--seed <int>]
    main_rff_svc_for_mnist_GPU.py -h|--help

Options:
    --input <str>        Directory path to the MNIST dataset.                [default: ../../dataset/mnist]
    --output <str>       File path to the output pickle file.                [default: result.pickle]
    --pcadim <int>       Output dimention of Principal Component Analysis.   [default: 256]
    --kdim <int>         Hyper parameter of RFF SVM (dimention of RFF)       [default: 128]
    --std_kernel <float> Hyper parameter of RFF SVM (stdev of RFF)           [default: 0.05]
    --std_error <float>  Hyper parameter of RFF SVM (stdev of RFF)           [default: 0.05]
    --seed <int>         Random seed.                                        [default: 111]
    --cpus <int>         Number of available CPUs.                           [default: -1]
    -h, --help           Show this message.
"""

import sys
import os
import time
import pickle

import docopt
import numpy   as np
import sklearn as skl
import tensorflow as tf


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


def train(X_cpu, y_cpu, kdim, s_k, s_e):

    ### Generate random matrix W and identity matrix I on CPU.
    W_cpu = s_k * np.random.randn(X_cpu.shape[1], kdim)
    I_cpu = np.eye(2 * kdim)

    ### Move matrices to GPU.
    I = tf.constant(I_cpu, dtype = tf.float64)
    W = tf.constant(W_cpu, dtype = tf.float64)
    X = tf.constant(X_cpu, dtype = tf.float64)
    y = tf.constant(y_cpu, dtype = tf.float64)

    Z   = tf.matmul(X, W)
    F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
    F   = tf.transpose(F_T)
    P   = tf.matmul(F, F_T)
    s   = (s_e)**2

    M = I - tf.matmul(tf.linalg.pinv(P + s * I), P)
    a = tf.matmul(tf.matmul(tf.transpose(y), F_T), M) / s
    S = tf.matmul(P, M) / s

    return (W.numpy(), a.numpy(), S.numpy())


def predict(X_cpu, W_cpu, a_cpu, S_cpu):

    ### Move matrix to GPU.
    X = tf.constant(X_cpu, dtype = tf.float64)
    W = tf.constant(W_cpu, dtype = tf.float64)
    a = tf.constant(a_cpu, dtype = tf.float64)
    S = tf.constant(S_cpu, dtype = tf.float64)

    Z   = tf.matmul(X, W)
    F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
    F   = tf.transpose(F_T)

    p = tf.matmul(a, F)
    p = tf.transpose(p)
    V = tf.matmul(tf.matmul(F_T, S), F)

    return (p.numpy(), np.diag(V.numpy()), V.numpy())


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

    ### Calculate score for test data.
    ys_train_oh = np.eye(int(np.max(ys_train) + 1))[ys_train]
    W, a, S = train(Xs_train.dot(T), ys_train_oh, args["--kdim"], args["--std_kernel"], args["--std_error"])

    m, v, V = predict(Xs_test.dot(T), W, a, S)
    score = 100.0 * np.mean(np.argmax(m, axis = 1) == ys_test)
    print("Score = %.2f [%%]" % score)

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


    ### Run main procedure.
    main(args)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
