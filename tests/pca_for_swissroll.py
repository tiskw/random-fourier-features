"""
Overview:
  Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    pca_for_swissroll.py linear [--samples <int>] [--seed <int>]
    pca_for_swissroll.py kernel [--samples <int>] [--kernel <str>] [--gamma <float>] [--seed <int>]
    pca_for_swissroll.py cpu [--rtype <str>] [--samples <int>] [--kdim <int>] [--stdev <float>] [--seed <int>]
    pca_for_swissroll.py gpu [--rtype <str>] [--samples <int>] [--kdim <int>] [--stdev <float>] [--seed <int>]
    pca_for_swissroll.py (-h | --help)

Options:
    linear           Run linese PCA.
    kernel           Run kernel PCA.
    cpu              Run RFF PCA on CPU.
    gpu              Run RFF PCA on GPU.
    --rtype <str>    Type of random matrix (rff/orf/qrf).                [default: rff]
    --samples <int>  Number of swiss roll data points.                   [default: 10000]
    --kernel <str>   Hyper parameter of kernel SVM (type of kernel).     [default: rbf]
    --gamma <float>  Hyper parameter of kernel SVM (softness of kernel). [default: 0.003]
    --kdim <int>     Hyper parameter of RFF SVM (dimention of RFF).      [default: 1024]
    --stdev <float>  Hyper parameter of RFF SVM (stdev of RFF).          [default: 0.06]
    --seed <int>     Random seed.                                        [default: 111]
    -h, --help       Show this message.
"""

# Import standard libraries.
import pathlib
import sys

# Import 3rd-party packages.
import docopt
import sklearn.datasets
import sklearn.decomposition


def main(args):
    """
    Main procedure.
    """
    # Print all arguments for debuging purpouse.
    print("Program starts: args =", args)

    # Fix seed for random fourier feature calclation.
    if args["cpu"] or args["gpu"]:
        rfflearn.seed(args["--seed"])

    # Create swiss roll data.
    X, color = sklearn.datasets.make_swiss_roll(args["--samples"], random_state=args["--seed"])

    # Create PCA class instance.
    if   args["linear"]          : pca = sklearn.decomposition.PCA(n_components=2)
    elif args["kernel"]          : pca = sklearn.decomposition.KernelPCA(n_components=2, kernel=args["--kernel"], gamma=args["--gamma"])
    elif args["--rtype"] == "rff": pca = rfflearn.RFFPCA(n_components=2, dim_kernel=args["--kdim"], std_kernel=args["--stdev"])
    elif args["--rtype"] == "orf": pca = rfflearn.ORFPCA(n_components=2, dim_kernel=args["--kdim"], std_kernel=args["--stdev"])
    elif args["--rtype"] == "qrf": pca = rfflearn.QRFPCA(n_components=2, dim_kernel=args["--kdim"], std_kernel=args["--stdev"])
    else                         : raise NotImplementedError("No PCA type selected.")

    # Run PCA (fit and transform seperately).
    pca.fit(X)
    Z = pca.transform(X)
    _ = pca.inverse_transform(Z)

    # Run PCA (fit and transform at once).
    Z = pca.fit_transform(X)

    # Print results.
    print(X.shape, "->", Z.shape)


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
