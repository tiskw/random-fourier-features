#!/usr/bin/env python3

"""
Overview:
  Train Random Fourier Feature canonical correlation analysis and plot results.

Usage:
    main_cca_for_artificial_data.py [--kdim <int>] [--std_kernel <float>] [--n_samples <int>] [--seed <int>]
    main_cca_for_artificial_data.py (-h | --help)

Options:
    --kdim <int>         Hyper parameter of RFF SVM (dimention of RFF).   [default: 16]
    --std_kernel <float> Hyper parameter of RFF SVM (stdev of RFF).       [default: 1.0]
    --n_samples <int>    Number of data samples.                          [default: 250]
    --seed <int>         Random seed.                                     [default: 111]
    -h, --help           Show this message.
"""

import os
import sys

import docopt
import numpy as np
import sklearn.cross_decomposition
import matplotlib.pyplot as mpl


def generate_artificial_data(n_samples):
    """
    This function generate the data pair, x and y.
    The data x = [x1, x2] and y = [y1, y2] has a strong correlation, y1 = x1**2.
    This means that it is theoretically possible to extract correlated subspace of this data.
    """
    xs1 = np.random.uniform(-1, 1, size = (n_samples,))
    xs2 = np.random.randn(n_samples)
    xs3 = np.random.randn(n_samples)

    data1 = np.array([xs1,    xs2]).T
    data2 = np.array([xs1**2, xs3]).T

    return (data1, data2)


def main(args):
    """
    Main function.
    """
    # Fix seed for random fourier feature calclation
    rfflearn.seed(args["--seed"])

    # Generate dataset.
    with utils.Timer("Generate dataset: "):
        X, Y = generate_artificial_data(args["--n_samples"])

    # Linear CCA
    with utils.Timer("Linear CCA training and inference: "):
        cca = sklearn.cross_decomposition.CCA(n_components = 1)
        cca.fit(X, Y)
        lin_z1, lin_z2 = cca.transform(X, Y)
    score_lin = cca.score(X, Y)
    print("Score of linear CCA:", score_lin)

    # RFF CCA
    with utils.Timer("RFF CCA trainig and inference: "):
        cca = rfflearn.RFFCCA(args["--kdim"], args["--std_kernel"], n_components = 1)
        cca.fit(X, Y)
        rff_z1, rff_z2 = cca.transform(X, Y)
    score_rff = cca.score(X, Y)
    print("Score of RFF CCA:", score_rff)

    mpl.figure(figsize = (8.4, 6.4))

    mpl.subplot(2, 2, 1)
    mpl.title("X1 vs Y1 (correlated part)")
    mpl.xlabel("X1 (= X[:, 0])")
    mpl.ylabel("Y1 (= Y[:, 0])")
    mpl.plot(X[:, 0], Y[:, 0], '.', color = "C1")
    mpl.grid()

    mpl.subplot(2, 2, 2)
    mpl.title("X2 vs Y2 (noise part)")
    mpl.xlabel("X2 (= X[:, 1])")
    mpl.ylabel("Y2 (= Y[:, 1])")
    mpl.plot(X[:, 1], Y[:, 1], '.', color = "C2")
    mpl.grid()

    mpl.subplot(2, 2, 3)
    mpl.title("linear CCA (score = %.2e)" % score_lin)
    mpl.xlabel("1st component of X")
    mpl.ylabel("1st component of Y")
    mpl.plot(lin_z1, lin_z2, '.', color = "C3")
    mpl.grid()

    mpl.subplot(2, 2, 4)
    mpl.title("RFF CCA (score = %.2e)" % score_rff)
    mpl.xlabel("1st component of X")
    mpl.ylabel("1st component of Y")
    mpl.plot(rff_z1, rff_z2, '.', color = "C4")
    mpl.grid()

    mpl.tight_layout()
    mpl.show()


if __name__ == "__main__":

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Add path to 'rfflearn/' directory.
    # The followings are not necessary if you copied 'rfflearn/' to the current
    # directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    import rfflearn.cpu   as rfflearn
    import rfflearn.utils as utils

    # Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    # Run main procedure.
    main(args)


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
