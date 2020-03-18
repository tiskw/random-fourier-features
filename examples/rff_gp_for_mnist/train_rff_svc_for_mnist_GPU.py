#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFSVC class which is a class for
# SVM classifier using RFF. Interface of RFFSVC is quite close to sklearn.svm.SVC.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 19, 2020
##################################################### SOURCE START #####################################################

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


def train(X_cpu, y_cpu, kdim = 3000, s_k = 0.1, s_e = 0.2):

    I_cpu = np.eye(2 * kdim)
    I = tf.constant(I_cpu, dtype = tf.float64)

    W_cpu = s_k * np.random.randn(X_cpu.shape[1], kdim)
    W = tf.constant(W_cpu, dtype = tf.float64)

    X = tf.constant(X_cpu, dtype = tf.float64)
    y = tf.constant(y_cpu, dtype = tf.float64)

    Z   = tf.matmul(X, W)
    F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
    F   = tf.transpose(F_T)

    P = tf.matmul(F, F_T)
    s = (s_e)**2

    M = I - tf.matmul(tf.linalg.pinv(P + s * I), P)
    a = tf.matmul(tf.matmul(tf.transpose(y), F_T), M) / s
    S = tf.matmul(P, M) / s

    return (W, a, S)


def predict(X_cpu, W, a, S):

    X = tf.constant(X_cpu, dtype = tf.float64)

    Z   = tf.matmul(X, W)
    F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
    F   = tf.transpose(F_T)

    p = tf.matmul(a, F)
    p = tf.transpose(p)

    # pred_var = [clip_flt(F[:, n].T @ (np.eye(2 * self.dim) - self.S) @ F[:, n]) for n in range(F.shape[1])]
    # return np.sqrt(np.array(pred_var))
    # return F @ self.S @ F

    return (p, None, None)


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
    T = mat_transform_pca(Xs_train, dim = 128)

    ### Calculate score for test data.
    ys_train_oh = np.eye(int(np.max(ys_train) + 1))[ys_train]
    W, a, S = train(Xs_train.dot(T), ys_train_oh)

    mean, variance, _ = predict(Xs_test.dot(T), W, a, S)
    print(np.argmax(mean.numpy()[:10], axis = 1))
    print(ys_test[:10])
    score = 100.0 * np.mean(np.argmax(mean.numpy(), axis = 1) == ys_test)
    print("Score = %.2f [%%]" % score)


if __name__ == "__main__":

    ### Run main procedure.
    main({})


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
