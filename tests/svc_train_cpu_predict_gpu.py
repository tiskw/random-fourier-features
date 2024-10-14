"""
Overview:
    Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST
    dataset. As a comparison with Kernel SVM, this script has a capability to run a kernel
    SVM as the same condition with RFF SVM.

Usage:
    svc_for_mnist.py [--data <str>] [--pcadim <int>] [--rtype <str>] [--kdim <int>]
                     [--stdev <float>] [--seed <int>] [--cpus <int>]
    svc_for_mnist.py (-h | --help)

Options:
    --data <str>      Directory path to the MNIST dataset.                [default: ./scikit_learn_data]
    --pcadim <int>    Output dimention of Principal Component Analysis.   [default: 64]
    --rtype <str>     Type of random matrix (rff/orf/qrf).                [default: rff]
    --kdim <int>      Hyper parameter of RFF/ORF SVM (dimention of RFF).  [default: 32]
    --stdev <float>   Hyper parameter of RFF/ORF SVM (stdev of RFF).      [default: 0.05]
    --seed <int>      Random seed.                                        [default: 111]
    --cpus <int>      Number of available CPUs.                           [default: -1]
    -h, --help        Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import numpy as np
import sklearn.datasets
import sklearn.svm


def main(args):
    """
    Main procedure.
    """
    # Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    # Fix seed for random fourier feature calclation.
    rfflearn.gpu.seed(args["--seed"])

    # Create classifier instance.
    if   args["--rtype"] == "rff": svc = rfflearn.cpu.RFFSVC(dim_kernel=args["--kdim"], std_kernel=args["--stdev"], n_jobs=args["--cpus"])
    elif args["--rtype"] == "orf": svc = rfflearn.cpu.ORFSVC(dim_kernel=args["--kdim"], std_kernel=args["--stdev"], n_jobs=args["--cpus"])
    elif args["--rtype"] == "qrf": svc = rfflearn.cpu.QRFSVC(dim_kernel=args["--kdim"], std_kernel=args["--stdev"], n_jobs=args["--cpus"])
    else                         : exit("Error: First argument must be 'kernel', 'rff' or 'orf'.")

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

    # Train SVM on CPU.
    svc.fit(Xs_train @ T, ys_train)

    # Calculate score for test data on CPU.
    score = 100 * svc.score(Xs_test @ T, ys_test)
    print(f"CPU: Score = {score:.2f} [%]")

    # Create SVC instance on GPU.
    if   args["--rtype"] == "rff": svc = rfflearn.gpu.RFFSVC(svc_cpu=svc)
    elif args["--rtype"] == "orf": svc = rfflearn.gpu.ORFSVC(svc_cpu=svc)
    elif args["--rtype"] == "qrf": svc = rfflearn.gpu.QRFSVC(svc_cpu=svc)

    # Calculate score for test data on GPU.
    score = 100 * svc.score(Xs_test @ T, ys_test)
    print(f"GPU: Score = {score:.2f} [%]")


if __name__ == "__main__":

    # Append path to rfflearn directory.
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Import rfflearn.
    import rfflearn.cpu
    import rfflearn.gpu

    # Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    # Run main procedure.
    main(args)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
