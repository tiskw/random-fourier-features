"""
Python module of canonical correlation analysis with random matrix for CPU.
"""

import sklearn.cross_decomposition

from .rfflearn_cpu_common import Base


class CCA:
    """
    Canonival Correlation Analysis with random matrix (RFF/ORF)
    """
    def __init__(self, rand_type, dim_kernel=128, std_kernel=0.1, W1=None, b1=None, W2=None, b2=None, **args):
        """
        Constructor of CCA class.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W1         (np.ndarray): Random matrix for the input X. If None then generated automatically.
            b1         (np.ndarray): Random bias for the input X. If None then generated automatically.
            W2         (np.ndarray): Random matrix for the output Y. If None then generated automatically.
            b2         (np.ndarray): Random bias for the output Y. If None then generated automatically.
            args       (dict)      : Extra arguments. This dictionaly will be unpacked and passed to CCA class constructor of scikit-learn.
        """
        self.fx1 = Base(rand_type, dim_kernel, std_kernel, W1, b1)
        self.fx2 = Base(rand_type, dim_kernel, std_kernel, W2, b2)
        self.cca = sklearn.cross_decomposition.CCA(**args)

    def fit(self, X, Y):
        """
        Extracts feature vectors and trains CCA.

        Args:
            X (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            Y (np.ndarray): Output matrix with shape (n_samples, n_features_output).

        Returns:
            (rfflearn.cpu.CCA): Fitted estimator.
        """
        self.fx1.set_weight(X.shape[1])
        self.fx2.set_weight(Y.shape[1])
        self.cca.fit(self.fx1.conv(X), self.fx2.conv(Y))
        return self

    def predict(self, X, copy=True):
        """
        Returns prediction results.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            copy (bool)      : Make a copy of X if True, otherwise perform in-place operation.

        Returns:
            Y (np.ndarray): Output matrix with shape (n_samples, n_features_output).
        """
        return self.cca.predict(self.fx1.conv(X), copy)

    def score(self, X, Y, sample_weight=None):
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

    def transform(self, X, Y=None, copy=True):
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
        return self.cca.transform(self.fx1.conv(X), None if Y is None else self.fx2.conv(Y), copy)


# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.


class RFFCCA(CCA):
    """
    Canonical correlation analysis with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFCCA(CCA):
    """
    Canonical correlation analysis with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
