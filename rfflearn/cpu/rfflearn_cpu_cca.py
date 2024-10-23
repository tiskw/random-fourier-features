"""
Python module of canonical correlation analysis with random matrix for CPU.
"""

# Declare published functions and variables.
__all__ = ["RFFCCA", "ORFCCA", "QRFCCA"]

# Import 3rd-party packages.
import numpy as np
import sklearn.cross_decomposition

# Import custom modules.
from .rfflearn_cpu_common import Base


class CCA:
    """
    Canonival Correlation Analysis with random matrix (RFF/ORF)
    """
    def __init__(self, rand_type: str, dim_kernel: int = 128, std_kernel: float = 0.1,
                 W1: np.ndarray = None, b1: np.ndarray = None,
                 W2: np.ndarray = None, b2: np.ndarray = None, **args: dict):
        """
        Constructor of CCA class.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W1         (np.ndarray): Random matrix for input X (generated automatically if None).
            b1         (np.ndarray): Random bias for input X (generated automatically if None).
            W2         (np.ndarray): Random matrix for output Y (generated automatically if None).
            b2         (np.ndarray): Random bias for output Y (generated automatically if None).
            args       (dict)      : Extra arguments. This dictionaly will be unpacked and passed
                                     to CCA class constructor of scikit-learn.
        """
        self.fx1 = Base(rand_type, dim_kernel, std_kernel, W1, b1)
        self.fx2 = Base(rand_type, dim_kernel, std_kernel, W2, b2)
        self.cca = sklearn.cross_decomposition.CCA(**args)

    def fit(self, X: np.ndarray, Y: np.ndarray, **args: dict):
        """
        Extracts feature vectors and trains CCA.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            Y    (np.ndarray): Output matrix with shape (n_samples, n_features_output).
            args (dict)      : Extra arguments. This dictionaly will be unpacked and passed
                               to the fit function of CCA class of scikit-learn.

        Returns:
            (rfflearn.cpu.CCA): Fitted estimator.
        """
        self.fx1.set_weight(X.shape[1])
        self.fx2.set_weight(Y.shape[1])
        self.cca.fit(self.fx1.conv(X), self.fx2.conv(Y), **args)
        return self

    def predict(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
        """
        Returns prediction results.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            copy (bool)      : Make a copy of X if True, otherwise perform in-place operation.

        Returns:
            Y (np.ndarray): Output matrix with shape (n_samples, n_features_output).
        """
        return self.cca.predict(self.fx1.conv(X), copy=copy)

    def score(self, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Returns evaluation score (the coefficient of determination R2).

        Args:
            X             (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            Y             (np.ndarray): Output matrix with shape (n_samples, n_features_output).
            sample_weight (np.ndarray): Sample weight of the evaluation with shape (n_samples,).

        Returns:
            (float): Returns the R2 score (coefficient of determination) of the prediction.
        """
        return self.cca.score(self.fx1.conv(X), self.fx2.conv(Y), sample_weight)

    def transform(self, X: np.ndarray, Y: np.ndarray = None, copy: bool = True) -> np.ndarray:
        """
        Applies the dimension reduction using the trained CCA.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            Y    (np.ndarray): Output matrix with shape (n_samples, n_features_output).
            copy (bool)      : Make a copy of X if True, otherwise perform in-place operation.

        Returns:
            (np.array or tuple): Applies the dimension reduction. Returns X_transformed if Y is
                                 not given, otherwise returns (x_transformed, y_transformed).
        """
        X_in = self.fx1.conv(X)
        Y_in = None if (Y is None) else self.fx2.conv(Y)
        return self.cca.transform(X_in, Y_in, copy=copy)


####################################################################################################
# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.
####################################################################################################


class RFFCCA(CCA):
    """
    Canonical correlation analysis with RFF.
    """
    def __init__(self, *pargs: list, **kwargs: dict):
        super().__init__("rff", *pargs, **kwargs)


class ORFCCA(CCA):
    """
    Canonical correlation analysis with ORF.
    """
    def __init__(self, *pargs: list, **kwargs: dict):
        super().__init__("orf", *pargs, **kwargs)


class QRFCCA(CCA):
    """
    Canonical correlation analysis with ORF.
    """
    def __init__(self, *pargs: list, **kwargs: dict):
        super().__init__("qrf", *pargs, **kwargs)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
