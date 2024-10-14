"""
Python module of support vector regression with random matrix for CPU.
"""

# Declare published functions and variables.
__all__ = ["RFFSVR", "ORFSVR", "QRFSVR"]

# Import 3rd-party packages.
import sklearn.svm
import sklearn.multiclass

# Import custom modules.
from .rfflearn_cpu_common import Base


class SVR(Base):
    """
    Support vector regression with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type, dim_kernel=128, std_kernel=0.1, W=None, b=None, **args):
        """
        Constractor. Save hyper parameters as member variables and create LinearSVR instance.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W          (np.ndarray): Random matrix for input `X` (generated automatically if None).
            b          (np.ndarray): Random bias for input `X` (generated automatically if None).
            args       (dict)      : Extra arguments. This will be passed to scikit-learn's
                                     LinearSVR class constructor.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.svr = sklearn.svm.LinearSVR(**args)

    def fit(self, X, y, **args):
        """
        Trains the SVR model according to the given data.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to scikit-learn's
                               `fit` function.

        Returns:
            (rfflearn.cpu.SVR): Fitted estimator.
        """
        self.set_weight(X.shape[1])
        self.svr.fit(self.conv(X), y, **args)
        return self

    def predict(self, X, **args):
        """
        Performs regression on the given data.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            args (dict)      : Extra arguments. This arguments will be passed to scikit-learn's
                               `predict_log_proba` function.

        Returns:
            (np.ndarray): The predicted values.
        """
        self.set_weight(X.shape[1])
        return self.svr.predict(self.conv(X), **args)

    def score(self, X, y, **args):
        """
        Returns the R2 score (coefficient of determination) of the prediction.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to scikit-learn's
                               `score` function.

        Returns:
            (float): The R2 score of the prediction.
        """
        self.set_weight(X.shape[1])
        return self.svr.score(self.conv(X), y, **args)


####################################################################################################
# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.
####################################################################################################


class RFFSVR(SVR):
    """
    Support vector machine with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFSVR(SVR):
    """
    Support vector machine with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


class QRFSVR(SVR):
    """
    Support vector machine with QRF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
