#!/usr/bin/env python3
#
# Python script
#
# Author: 
# Date  : Oct 20, 2018
#################################### SOURCE START ###################################

import Utils as utils
import Timer as timer
import sklearn.svm

if __name__ == "__main__":
# {{{

    ### Create classifier instance
    svc = sklearn.svm.SVC()

    ### Load training data
    with timer.Timer("Loading training data: "):
        Xs_train = utils.vectorise_MNIST_images("../data/MNIST_train_images.npy")
        ys_train = utils.vectorise_MNIST_labels("../data/MNIST_train_labels.npy")

    ### Load test data
    with timer.Timer("Loading test data: "):
        Xs_test = utils.vectorise_MNIST_images("../data/MNIST_test_images.npy")
        ys_test = utils.vectorise_MNIST_labels("../data/MNIST_test_labels.npy")

    ### Create matrix for principal component analysis
    with timer.Timer("Calculate PCA matrix: "):
        T = utils.mat_transform_pca(Xs_train, dim = 256)

    ### Train SVM with RBF kernel
    with timer.Timer("Kernel SVM learning time: "):
        svc.fit(Xs_train.dot(T), ys_train)

    ### Calculate score for test data
    with timer.Timer("Kernel SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * svc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
