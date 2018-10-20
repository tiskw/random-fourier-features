#!/usr/bin/env python3
#
# Python script
#
# Author: 
# Date  : Oct 20, 2018
#################################### SOURCE START ###################################

import numpy as np

### Load train/test image data
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0

### Load train/test label data
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)

### PCA analysis
def mat_transform_pca(Xs, dim = 100):
    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
