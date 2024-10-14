"""
Overview:
    An example of support vector regression with random Fourier feature.

Usage:
    gpr_sparse_data.py cpu [--rtype <str>] [--kdim <int>] [--stdev <float>]
                           [--n_test <int>] [--n_train <int>] [--no_pred_std] [--seed <int>]
    gpr_sparse_data.py (-h | --help)

Options:
    cpu              Run Gaussian process regression on CPU.
    --rtype <str>    Type of random matrix (rff/orf/qrf).           [default: rff]
    --kdim <int>     Hyper parameter of RFF SVM (dimention of RFF). [default: 32]
    --stdev <float>  Hyper parameter of RFF SVM (stdev of RFF).     [default: 5.0]
    --n_train <int>  Number of training samples.                    [default: 1000]
    --n_test <int>   Number of test samples.                        [default: 101]
    --seed <int>     Random seed.                                   [default: 111]
    -h, --help       Show this message.
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
    if   args["--rtype"] == "rff": svr = rfflearn.RFFSVR(args["--kdim"], args["--stdev"])
    elif args["--rtype"] == "orf": svr = rfflearn.ORFSVR(args["--kdim"], args["--stdev"])
    elif args["--rtype"] == "qrf": svr = rfflearn.QRFSVR(args["--kdim"], args["--stdev"])

    # Create training and test data.
    Xs_train = np.random.randn(args["--n_train"], 1)
    ys_train = np.sin(Xs_train**1).flatten()
    Xs_test  = np.linspace(-4, 4, args["--n_test"]).reshape((args["--n_test"], 1))
    ys_test  = np.sin(Xs_test**2).flatten()

    # Train SVM with orthogonal random features.
    svr.fit(Xs_train, ys_train)

    # Conduct prediction for the test data.
    pred = svr.predict(Xs_test)

    # Compute and print score.
    print("R2 score:", svr.score(Xs_test, ys_test))


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
