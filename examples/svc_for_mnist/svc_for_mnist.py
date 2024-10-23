"""
Train and evaluate SVC with RFF/ORF/QRF.
"""

# Import standard libraries.
import argparse

# Import 3rd-party packages.
import numpy as np
import sklearn.datasets
import sklearn.svm

# Import rfflearn.
import rfflearn.cpu as rfflearn
# import rfflearn.gpu as rfflearn

# Dictionary of model classes.
model_classes = {
    "rff": rfflearn.RFFSVC,
    "orf": rfflearn.ORFSVC,
    "qrf": rfflearn.QRFSVC,
}


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtype", metavar="STR", type=str, default="rff",
                        help="type of random matrix (rff/orf/qrf)")
    parser.add_argument("--kdim", metavar="INT", type=int, default=128,
                        help="dimension of random kernel matrix")
    parser.add_argument("--kstd", metavar="FLOAT", type=float, default=0.05,
                        help="standard deviation of random kernel matrix")
    parser.add_argument("--pcadim", metavar="INT", type=int, default=128,
                        help="output dimention of principal component analysis")
    return parser.parse_args()


def load_mnist():
    """
    Load MNIST dataset.
    """
    # Load MNIST.
    Xs, ys = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, data_home="./scikit_learn_data")

    # Split to training and test data.
    Xs_train, Xs_test, ys_train, ys_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=10000, shuffle=False)

    # Standardize the data (convert the input data range from [0, 255] to [0, 1]).
    Xs_train = Xs_train.astype(np.float64) / 255.0
    Xs_test  = Xs_test.astype(np.float64)  / 255.0

    # Convert the string label to integer label.
    ys_train = ys_train.astype(np.int32)
    ys_test  = ys_test.astype(np.int32)

    return (Xs_train, ys_train, Xs_test, ys_test)


def compute_pca_matrix(Xs, pcadim):
    """
    Create matrix for principal component analysis.
    """
    _, V = np.linalg.eig(Xs.T @ Xs)
    T = np.real(V[:, :pcadim])
    return T


def main(args):
    """
    Main procedure.
    """
    print("args =", args)

    # Load MNIST dataset.
    Xs_train, ys_train, Xs_test, ys_test = load_mnist()

    # Fix seed for random fourier feature calclation.
    rfflearn.seed(111)

    # Create classifier instance.
    svc = model_classes[args.rtype](dim_kernel=args.kdim, std_kernel=args.kstd)

    # Create matrix for principal component analysis.
    T = compute_pca_matrix(Xs_train, args.pcadim)

    # Train SVM with batch random fourier features
    svc.fit(Xs_train @ T, ys_train)

    # Calculate score for test data
    score = 100 * svc.score(Xs_test @ T, ys_test)
    print("Score = %.2f [%%]" % score)


if __name__ == "__main__":
    main(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
