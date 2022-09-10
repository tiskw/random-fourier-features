#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFGPC class which is a class for
# Gaussian process classifier using RFF. Interface of RFFGPC is quite close to
# sklearn.gaussian_process.GaussianProcessClassifier.
##################################################### SOURCE START #####################################################

"""
Overview:
  Validate Gaussian process classifire with RFF/OFR.
  Before running this script, make sure to create MNIST dataset.

Usage:
    valid_rff_gpc_for_mnist.py cpu [--input <str>] [--model <str>] [--cpus <int>]
    valid_rff_gpc_for_mnist.py gpu [--input <str>] [--model <str>] [--cpus <int>]
    valid_rff_gpc_for_mnist.py (-h | --help)

Options:
    --input <str>   Directory path to the MNIST dataset.    [default: ../../dataset/mnist]
    --model <str>   File path to the model pickle file.     [default: result_mnist.pickle]
    --cpus <int>    Number of available CPUs.               [default: -1]
    -h, --help      Show this message.
"""

import os
import pickle
import sys

import docopt
import numpy as np

### Load train/test image data.
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0 - 0.5

### Load train/test label data.
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)

### Main procedure.
def main(args):

    ### Load model file.
    with open(args["--model"], "rb") as ofp:
        data = pickle.load(ofp)
        W, b = (data["W"], data["b"])
        a, S = (data["a"], data["S"])
        T    = data["pca"]
        args = data["args"]

    ### Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    ### Create classifier instance.
    gpc = rfflearn.RFFGPC(args["--kdim"], args["--std_kernel"], args["--std_error"], W=W, b=b, a=a, S=S)

    ### Load test data.
    with utils.Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy"))
        ys_test = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy"))

    ### Idle CPU/GPU for accurate measurement of inference time.
    gpc.predict(np.zeros(Xs_test.shape).dot(T))

    ### Calculate score for test data.
    with utils.Timer("GPC prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * gpc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)

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
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
