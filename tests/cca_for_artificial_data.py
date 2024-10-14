#!/usr/bin/env python3

"""
Overview:
  Train Random Fourier Feature canonical correlation analysis and plot results.

Usage:
    cca_for_artificial_data.py [--rtype <str>] [--kdim <int>] [--std_kernel <float>]
                               [--n_samples <int>] [--seed <int>]
    cca_for_artificial_data.py (-h | --help)

Options:
    --rtype <str>        Type of random matrix (rff/orf/qrf).             [default: rff]
    --kdim <int>         Hyper parameter of RFF SVM (dimention of RFF).   [default: 16]
    --std_kernel <float> Hyper parameter of RFF SVM (stdev of RFF).       [default: 1.0]
    --n_samples <int>    Number of data samples.                          [default: 250]
    --seed <int>         Random seed.                                     [default: 111]
    -h, --help           Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import numpy as np
import sklearn.cross_decomposition


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
    X, Y = generate_artificial_data(args["--n_samples"])

    # Linear CCA.
    cca = sklearn.cross_decomposition.CCA(n_components=1)
    cca.fit(X, Y)
    lin_z1, lin_z2 = cca.transform(X, Y)
    score_lin = cca.score(X, Y)
    print("Score of linear CCA:", score_lin)

    # RFF CCA.
    if   args["--rtype"] == "rff": cca = rfflearn.RFFCCA(args["--kdim"], args["--std_kernel"], n_components=1)
    elif args["--rtype"] == "orf": cca = rfflearn.ORFCCA(args["--kdim"], args["--std_kernel"], n_components=1)
    elif args["--rtype"] == "qrf": cca = rfflearn.QRFCCA(args["--kdim"], args["--std_kernel"], n_components=1)

    cca.fit(X, Y)
    rff_z1, rff_z2 = cca.transform(X, Y)
    score_rff = cca.score(X, Y)
    print("Score of RFF CCA:", score_rff)


if __name__ == "__main__":

    # Append path to rfflearn directory.
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Import rfflearn.
    import rfflearn.cpu as rfflearn

    # Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    # Run main procedure.
    main(args)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
