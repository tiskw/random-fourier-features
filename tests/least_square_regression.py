"""
Overview:
    Train Random Fourier Feature least square regression.

Usage:
    least_square_regression.py [--rtype <str>] [--kdim <int>] [--stdev <float>]
                               [--n_train <int>] [--n_test <int>] [--seed <int>]
    least_square_regression.py (-h | --help)

Options:
    --rtype <str>     Random matrix type (rff ot orf).   [default: rff]
    --kdim <int>      Dimention of RFF/ORF.              [default: 8]
    --stdev <float>   Standard deviation of RFF/ORF.     [default: 0.5]
    --n_train <int>   Number of training data points.    [default: 21]
    --n_test <int>    Number of test data points.        [default: 101]
    --seed <int>      Random seed.                       [default: 111]
    -h, --help        Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import numpy as np


def main(args):
    """
    Main procedure
    """
    # Fix seed for random fourier feature calclation
    rfflearn.seed(111)

    # Create classifier instance
    if   args["--rtype"] == "rff": reg = rfflearn.RFFRegressor(dim_kernel=args["--kdim"], std_kernel=args["--stdev"])
    elif args["--rtype"] == "orf": reg = rfflearn.ORFRegressor(dim_kernel=args["--kdim"], std_kernel=args["--stdev"])
    elif args["--rtype"] == "qrf": reg = rfflearn.QRFRegressor(dim_kernel=args["--kdim"], std_kernel=args["--stdev"])
    else                         : raise RuntimeError("Error: 'random_type' must be 'rff' or 'orf'.")

    # Prepare training data
    Xs_train = np.linspace(0, 3, args["--n_train"]).reshape((args["--n_train"], 1))
    ys_train = np.sin(Xs_train**2)
    Xs_test  = np.linspace(0, 3, args["--n_test"]).reshape((args["--n_test"], 1))
    ys_test  = np.sin(Xs_test**2)

    # Train regression with random fourier features
    reg.fit(Xs_train, ys_train)

    # Conduct prediction for the test data
    predict = reg.predict(Xs_test)
    print("predict.shape =", predict.shape)


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
