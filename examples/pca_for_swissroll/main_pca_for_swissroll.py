#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 29, 2021
##################################################### SOURCE START #####################################################

"""
Overview:
  Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    main_rff_pca_for_swissroll.py linear [--samples <int>] [--seed <int>]
    main_rff_pca_for_swissroll.py kernel [--samples <int>] [--kernel <str>] [--gamma <float>] [--seed <int>]
    main_rff_pca_for_swissroll.py rff [--samples <int>] [--kdim <int>] [--stdev <float>] [--seed <int>]
    main_rff_pca_for_swissroll.py orf [--samples <int>] [--kdim <int>] [--stdev <float>] [--seed <int>]
    main_rff_pca_for_swissroll.py (-h | --help)

Options:
    linear           Run linese PCA.
    kernel           Run kernel PCA.
    rff              Run RFF PCA.
    orf              Run ORF PCA.
    --samples <int>  Number of swiss roll data points.                   [default: 10000]
    --kernel <str>   Hyper parameter of kernel SVM (type of kernel).     [default: rbf]
    --gamma <float>  Hyper parameter of kernel SVM (softness of kernel). [default: 1.0]
    --kdim <int>     Hyper parameter of RFF SVM (dimention of RFF).      [default: 1024]
    --stdev <float>  Hyper parameter of RFF SVM (stdev of RFF).          [default: 0.05]
    --seed <int>     Random seed.                                        [default: 111]
    -h, --help       Show this message.
"""

import sys
import os

import docopt
import sklearn as skl
import sklearn.datasets
import matplotlib.pyplot as mpl

def main(args):

    ### Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    ### Fix seed for random fourier feature calclation.
    rfflearn.seed(args["--seed"])

    ### Create swiss roll data.
    with utils.Timer("Creating swiss roll data: "):
        X, color = skl.datasets.make_swiss_roll(args["--samples"], random_state = args["--seed"])

    ### Create PCA class instance.
    if   args["linear"]: pca = skl.decomposition.PCA(n_components=2)
    elif args["kernel"]: pca = skl.decomposition.KernelPCA(n_components = 2, kernel = args["--kernel"], gamma = args["--gamma"])
    elif args["rff"]   : pca = rfflearn.RFFPCA(n_components = 2, dim_kernel = args["--kdim"], std_kernel = args["--stdev"])
    elif args["orf"]   : pca = rfflearn.ORFPCA(n_components = 2, dim_kernel = args["--kdim"], std_kernel = args["--stdev"])
    else               : raise NotImplementedError("No PCA type selected.")

    ### Run PCA.
    with utils.Timer("PCA training: "):
        X_p = pca.fit_transform(X)

    ### Draw input data in 3D.
    fig = mpl.figure(1)
    axs = fig.add_subplot(111, projection = "3d")
    axs.scatter(X[:, 0], X[:, 1], X[:, 2], c = color, cmap = mpl.cm.rainbow)
    axs.set_title("Swiss Roll in 3D")

    ### Draw PCA results.
    mpl.figure(2)
    mpl.scatter(X_p[:, 0], X_p[:, 1], c = color, cmap = mpl.cm.rainbow)
    mpl.title("First 2 principal components of PCA")
    mpl.xlabel("1st Principal Component")
    mpl.ylabel("2nd Principal Component")
    mpl.grid()

    mpl.show()

if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Add path to 'rfflearn/' directory.
    ### The followings are not necessary if you copied 'rfflearn/' to the current
    ### directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    import rfflearn.cpu   as rfflearn
    import rfflearn.utils as utils

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
