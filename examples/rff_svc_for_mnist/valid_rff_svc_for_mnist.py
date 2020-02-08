#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFSVC class which is a class for
# SVM classifier using RFF. Interface of RFFSVC is quite close to sklearn.svm.SVC.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 07, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
  Validate Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    main_rff_svc_for_mnist.py cpu [--input <str>] [--model <str>]
    main_rff_svc_for_mnist.py tensorflow [--input <str>] [--model <str>] [--batch_size <int>]
    main_rff_svc_for_mnist.py -h|--help

Options:
    cpu                 Run inference on CPU.
    tensorflow          Run inference on GPU using Tensorflow 2.
    --input <str>       Directory path to the MNIST dataset.   [default: ../../dataset/mnist]
    --model <str>       File path to the output pickle file.   [default: result.pickle]
    --batch_size <int>  Batch size for GPU inference.          [default: 2000]
    -h, --help          Show this message.
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

### NOTE: Tensorflow is imported inside 'main_tensorflow' for users who run only CPU inference and don't have tensorflow-gpu.


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


### Inference using CPU.
def main_cpu(args):

    ### Load test data.
    with Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy"))
        ys_test = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy"))

    ### Load pickled result.
    with Timer("Loading model: "):
        with open(args["--model"], "rb") as ifp:
            result = pickle.load(ifp)
        svc = result["svc"]
        T   = result["pca"]

    ### Calculate score for test data.
    with Timer("SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * svc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)


### Inference using GPU.
def main_tensorflow(args):

    import tensorflow as tf

    ### Function for building Tensorflow model of RFF.
    def build_model_tensorflow(svc, M_pca, batch_size):

        ### Create parameters on CPU at first.
        ###   - W: Random matrix of Random Fourier Features.
        ###        If PCA applied, combine it to the random matrix for high throughput.
        ###   - A: Coefficients of Linear SVC.
        ###   - b: Intercepts of Linear SVC.
        W_cpu = M_pca.dot(svc.W)
        A_cpu = svc.svm.coef_.T
        b_cpu = svc.svm.intercept_.T

        ### Cureate psuedo input on CPU for GPU variable creation.
        x_cpu = np.zeros((batch_size, W_cpu.shape[0]), dtype = np.float32)

        ### Create GPU variables.
        x_gpu = tf.Variable(x_cpu, dtype = tf.float32)
        W_gpu = tf.constant(W_cpu, dtype = tf.float32)
        A_gpu = tf.constant(A_cpu, dtype = tf.float32)
        b_gpu = tf.constant(b_cpu, dtype = tf.float32)

        ### Model is only a tuple of necessary variables and constants.
        model = (x_gpu, W_gpu, A_gpu, b_gpu)

        ### Run the GPU model for creating the graph (because we are in the eager-mode here).
        _ = run_model_tensorflow(x_cpu, *model)

        return model

    ### Function for running the Tensorflow model of RFF.
    @tf.function
    def run_model_tensorflow(x_cpu, x_gpu, W_gpu, A_gpu, b_gpu):
        x_gpu.assign(x_cpu)
        z = tf.matmul(x_gpu, W_gpu)
        z = tf.concat([tf.cos(z), tf.sin(z)], 1)
        return tf.matmul(z, A_gpu) + b_gpu

    ### Load test data.
    with Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy")).astype(np.float32)
        ys_test = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy")).astype(np.float32)

    ### Load trained model.
    with Timer("Loading model: "):
        with open(args["--model"], "rb") as ifp:
            result = pickle.load(ifp)
        svc = result["svc"]
        T   = result["pca"]

    ### Only RFFSVC support GPU inference.
    if type(svc) != pyrff.RFFSVC:
        exit("GPU inference is available only PyRFF.RFFSVC class.")

    ### TODO: One-versus-one classifier is not supported now.
    if svc.svm.get_params()["estimator__multi_class"] != "ovr":
        exit("Sorry, current implementation support only One-versus-the-rest classifier.")

    ### Calsulate batch size and butch numbers.
    b_size = args["--batch_size"]
    b_num  = Xs_test.shape[0] // b_size

    ### Warning: fractions of the batches are ignored.
    ###          For guaranteeing the correct accuracy, this software will abort is fraction is not zero.
    if Xs_test.shape[0] % b_size != 0:
        exit("Batch size must be a divisor of total data number (= %d)." % Xs_test.shape[0])

    ### Build tensorflow model.
    model = build_model_tensorflow(svc, T, args["--batch_size"])

    ### Prepare a vector to store the inference results.
    result = np.zeros(ys_test.shape)

    ### Run inference for each batch.
    with Timer("SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        for n in range(b_num):
            batch_Xs = Xs_test[b_size*n:b_size*(n+1), :]
            batch_ys = ys_test[b_size*n:b_size*(n+1)]
            logits   = run_model_tensorflow(batch_Xs, *model)
            result[b_size*n:b_size*(n+1)] = np.argmax(logits, 1)
        total_score = 100.0 * np.mean(result == ys_test)
    print("Score = %.2f [%%]" % total_score)


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    if   args["cpu"]       : main_cpu(args)
    elif args["tensorflow"]: main_tensorflow(args)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
