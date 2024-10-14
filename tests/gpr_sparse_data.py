"""
Overview:
    An example of Gaussian Process Regression with Random Fourier Feature.
    As a comparison with the noemal GPR, this script has a capability
    to run the normal GPR under the same condition with RFF GPR.

Usage:
    gpr_sparse_data.py kernel [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    gpr_sparse_data.py cpu [--rtype <str>] [--kdim <int>] [--std_kernel <float>] [--std_error <float>]
                           [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    gpr_sparse_data.py gpu [--rtype <str>] [--kdim <int>] [--std_kernel <float>] [--std_error <float>]
                           [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    gpr_sparse_data.py (-h | --help)

Options:
    kernel                Run normal Gaussian Process.
    cpu                   Run Gaussian process regression on CPU.
    gpu                   Run Gaussian process regression on GPU.
    --rtype <str>         Type of random matrix (rff/orf/qrf).           [default: rff]
    --kdim <int>          Hyper parameter of RFF SVM (dimention of RFF). [default: 32]
    --std_kernel <float>  Hyper parameter of RFF SVM (stdev of RFF).     [default: 5.0]
    --std_error <float>   Hyper parameter of RFF SVM (stdev of error).   [default: 1.0]
    --n_train <int>       Number of training samples.                    [default: 1000]
    --n_test <int>        Number of test samples.                        [default: 101]
    --seed <int>          Random seed.                                   [default: 111]
    -h, --help            Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import numpy as np
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
    else:
        if   args["--rtype"] == "rff": gpr = rfflearn.RFFGPR(args["--kdim"], args["--std_kernel"], std_error=args["--std_error"])
        elif args["--rtype"] == "orf": gpr = rfflearn.ORFGPR(args["--kdim"], args["--std_kernel"], std_error=args["--std_error"])
        elif args["--rtype"] == "qrf": gpr = rfflearn.QRFGPR(args["--kdim"], args["--std_kernel"], std_error=args["--std_error"])

    # Create training and test data.
    Xs_train = np.random.randn(args["--n_train"], 1)
    ys_train = np.sin(Xs_train**2)
    Xs_test  = np.linspace(-4, 4, args["--n_test"]).reshape((args["--n_test"], 1))
    ys_test  = np.sin(Xs_test**2)

    # Train SVM with orthogonal random features.
    gpr.fit(Xs_train, ys_train)

    # Conduct prediction for the test data.
    pred, pstd, pcov = gpr.predict(Xs_test, return_std=True, return_cov=True)

    # Compute and print score.
    print("R2 score:", gpr.score(Xs_test, ys_test))


if __name__ == "__main__":

    # Append path to rfflearn directory.
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Import rfflearn.
    if args["gpu"]: import rfflearn.gpu as rfflearn
    else          : import rfflearn.cpu as rfflearn

    # Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    # Run main procedure.
    main(args)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
