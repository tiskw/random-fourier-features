"""
Overview:
    Train Gaussian process classifire with RFF/OFR.

Usage:
    gpc_for_mnist.py cpu [--data <str>] [--pcadim <int>] [--rtype <str>] [--kdim <int>]
                         [--std_kernel <float>] [--std_error <float>] [--seed <int>]
    gpc_for_mnist.py gpu [--data <str>] [--pcadim <int>] [--rtype <str>] [--kdim <int>]
                         [--std_kernel <float>] [--std_error <float>] [--seed <int>]
    gpc_for_mnist.py (-h | --help)

Options:
    --data <str>          Directory path to the MNIST dataset.              [default: ./scikit_learn_data]
    --pcadim <int>        Output dimention of Principal Component Analysis. [default: 64]
    --rtype <str>         Type of random matrix (rff/orf/qrf).              [default: rff]
    --kdim <int>          Hyper parameter of RFF SVM (dimention of RFF).    [default: 32]
    --std_kernel <float>  Hyper parameter of RFF SVM (stdev of RFF).        [default: 0.1]
    --std_error <float>   Hyper parameter of RFF SVM (stdev of error).      [default: 0.5]
    --seed <int>          Random seed.                                      [default: 111]
    --cpus <int>          Number of available CPUs.                         [default: -1]
    --n_trials <int>      Number of trials in hyper parameter tuning.       [default: 3]
    -h, --help            Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import numpy as np
import sklearn.datasets


def main(args):
    """
    Main procedure.
    """
    # Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    # Fix seed for random fourier feature calclation.
    rfflearn.seed(args["--seed"])

    # Create classifier instance.
    if   args["--rtype"] == "rff": gpc = rfflearn.RFFGPC(dim_kernel=args["--kdim"], std_kernel=args["--std_kernel"], std_error=args["--std_error"])
    elif args["--rtype"] == "orf": gpc = rfflearn.ORFGPC(dim_kernel=args["--kdim"], std_kernel=args["--std_kernel"], std_error=args["--std_error"])
    elif args["--rtype"] == "qrf": gpc = rfflearn.QRFGPC(dim_kernel=args["--kdim"], std_kernel=args["--std_kernel"], std_error=args["--std_error"])
    else                         : raise RuntimeError("Error: 'random_type' must be 'rff', 'orf' or 'qrf'.")

    # Load MNIST.
    Xs, ys = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, data_home=args["--data"])

    # Split to training and test data.
    Xs_train, Xs_test, ys_train, ys_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=10000, shuffle=False)

    # Convert the input data range from [0, 255] to [0, 1].
    Xs_train = Xs_train.astype(np.float64) / 255.0
    Xs_test  = Xs_test.astype(np.float64)  / 255.0

    # Convert the string label to integer.
    ys_train = ys_train.astype(np.int32)
    ys_test  = ys_test.astype(np.int32)

    # Create matrix for principal component analysis.
    _, V = np.linalg.eig(Xs_train.T @ Xs_train)
    T = np.real(V[:, :args["--pcadim"]])

    # Train SVM with orthogonal random features.
    gpc.fit(Xs_train @ T, ys_train)

    # Predict.
    ps_test, _, _ = gpc.predict(Xs_test @ T, return_std=True, return_cov=True)

    # Calculate score for test data.
    score = 100 * gpc.score(Xs_test @ T, ys_test)
    print(f"Score = {score:.2f} [%]")


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
