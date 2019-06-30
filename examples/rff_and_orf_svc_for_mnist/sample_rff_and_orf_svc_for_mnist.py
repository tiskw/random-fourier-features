#!/usr/bin/env python3
#
# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct 20, 2018
#################################### SOURCE START ###################################

import sys
import os

### Add path to PyRFF.py
### The followings are not necessary if you copied PyRFF.py to the current directory
### or other directory which is included in the Python path
current_dir = os.path.dirname(__file__)
module_path = os.path.join(current_dir, "../../source")
sys.path.append(module_path)

import time
import traceback
import numpy   as np
import sklearn as skl
import PyRFF   as pyrff


### Load train/test image data
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0


### Load train/test label data
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)


### PCA analysis for dimention reduction
def mat_transform_pca(Xs, dim = 100):
    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])


### Class for measure elasped time using 'with' sentence
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


### Main procedure
def main():

    print("Program starts:", sys.argv)

    ### Fix seed for random fourier feature calclation
    pyrff.seed(111)

    ### Create classifier instance
    if   len(sys.argv) < 2      : exit("Error: First argument must be 'kernel', 'rff' or 'orf'.")
    elif sys.argv[1] == "kernel": svc = skl.svm.SVC(kernel = "rbf", gamma = "auto")
    elif sys.argv[1] == "rff"   : svc = pyrff.RFFSVC(dim_output = int(1024), std = float(0.05), tol = 1.0E-3)
    elif sys.argv[1] == "orf"   : svc = pyrff.ORFSVC(dim_output = int(1024), std = float(0.05), tol = 1.0E-3)
    else                        : exit("Error: First argument must be 'kernel', 'rff' or 'orf'.")

    ### Load training data
    with Timer("Loading training data: "):
        Xs_train = vectorise_MNIST_images("../../data/MNIST_train_images.npy")
        ys_train = vectorise_MNIST_labels("../../data/MNIST_train_labels.npy")

    ### Load test data
    with Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images("../../data/MNIST_test_images.npy")
        ys_test = vectorise_MNIST_labels("../../data/MNIST_test_labels.npy")

    ### Create matrix for principal component analysis
    with Timer("Calculate PCA matrix: "):
        T = mat_transform_pca(Xs_train, dim = 256)

    ### Train SVM with orthogonal random features
    with Timer("SVM learning: "):
        svc.fit(Xs_train.dot(T), ys_train)

    ### Calculate score for test data
    with Timer("SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * svc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)


if __name__ == "__main__":
    try   : main()
    except: traceback.print_exc()


#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
