#!/usr/bin/env python3

"""
Overview:
  An example of Gaussian Process Regression with Random Fourier Feature.
  As a comparison with the noemal GPR, this script has a capability to run the normal GPR under the same condition with RFF GPR.

Usage:
    main_gpr_sparse_data.py kernel [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    main_gpr_sparse_data.py rff [--kdim <int>] [--std_kernel <float>] [--std_error <float>]
                                [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    main_gpr_sparse_data.py orf [--kdim <int>] [--std_kernel <float>] [--std_error <float>]
                                [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    main_gpr_sparse_data.py (-h | --help)

Options:
    kernel               Run normal Gaussian Process.
    rff                  Run Gaussian process with Random Fourier Features.
    orf                  Run Gaussian process with Orthogonal Random Features.
    --kdim <int>         Hyper parameter of RFF SVM (dimention of RFF).       [default: 32]
    --std_kernel <float> Hyper parameter of RFF SVM (stdev of RFF).           [default: 5.0]
    --std_error <float>  Hyper parameter of RFF SVM (stdev of error).         [default: 1.0]
    --n_train <int>      Number of training samples.                          [default: 10000]
    --n_test <int>       Number of test samples.                              [default: 101]
    --no_pred_std        Run standard deviation prediction.
    --seed <int>         Random seed.                                         [default: 111]
    -h, --help           Show this message.
"""

import os
import sys

import docopt
import matplotlib.pyplot as mpl
import numpy             as np
import sklearn
import sklearn.gaussian_process

def main(args):
    """
    Main procedure.
    """
    # Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    # Fix seed for random fourier feature calclation.
    rfflearn.seed(args["--seed"])

    # Create classifier instance.
    if args["kernel"]:
        kf  = sklearn.gaussian_process.kernels.RBF(1.0 / args["--std_kernel"]) \
            + sklearn.gaussian_process.kernels.WhiteKernel(args["--std_error"])
        gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kf, random_state=args["--seed"])
    elif args["rff"]:
        gpr = rfflearn.RFFGPR(args["--kdim"], args["--std_kernel"], std_error = args["--std_error"])
    elif args["orf"]:
        gpr = rfflearn.ORFGPR(args["--kdim"], args["--std_kernel"], std_error = args["--std_error"])

    # Load training data.
    with utils.Timer("Generating training/testing data: "):
        Xs_train = np.random.randn(args["--n_train"], 1)
        ys_train = np.sin(Xs_train**2)
        Xs_test  = np.linspace(-4, 4, args["--n_test"]).reshape((args["--n_test"], 1))
        ys_test  = np.sin(Xs_test**2)

    # Train SVM with orthogonal random features.
    with utils.Timer("GPR learning: ", unit="ms"):
        gpr.fit(Xs_train, ys_train)

    # Conduct prediction for the test data.
    if args["--no_pred_std"]:
        with utils.Timer("GPR inference: ", unit="us", devide_by=args["--n_test"]):
            pred = gpr.predict(Xs_test)
            pstd = None
    else:
        with utils.Timer("GPR inference: ", unit="us", devide_by=args["--n_test"]):
            pred, pcov = gpr.predict(Xs_test, return_cov=True)
            pred = pred.reshape((pred.shape[0],))
            pstd = np.diag(pcov).reshape((pred.shape[0],))

    print("R2 score:", gpr.score(Xs_test, ys_test))

    # Plot regression results.
    with utils.Timer("Drawing figure: "):
        mpl.figure(figsize=(6, 3.5))
        mpl.title("Regression of y = sin(x^2) using Gaussian Process w/ RFF")
        mpl.xlabel("X")
        mpl.ylabel("Y")
        mpl.plot(Xs_train[::100], ys_train[::100], ".")
        mpl.plot(Xs_test,         ys_test,         ".")
        mpl.plot(Xs_test,         pred,            "-")
        if pstd is not None:
            mpl.fill_between(Xs_test.reshape((Xs_test.shape[0],)),  pred - pstd, pred + pstd, facecolor="#DDDDDD")
        mpl.legend(["Training data", "Test data GT", "Prediction", "1-sigma area"])
        mpl.grid()
        mpl.tight_layout()
        # mpl.savefig("figure_rff_gpr_sparse_data.svg")

    # Re-sampling from the predicted mean and covariance to verify the mean and covariance.
    with utils.Timer("Re-sampling: "):
        ys_samples = np.random.multivariate_normal(pred, pcov, size=100)

    # Plot re-sampled data.
    with utils.Timer("Drawing figure for re-sampling: "):
        mpl.figure(figsize=(6, 3.5))
        mpl.title("Re-sampling from the predicted mean and covariance")
        mpl.xlabel("X")
        mpl.ylabel("Y")
        for ys_sample in ys_samples:
            mpl.plot(Xs_test, ys_sample, "-")
        mpl.grid()
        mpl.tight_layout()
        # mpl.savefig("figure_rff_gpr_resampling.svg")

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
