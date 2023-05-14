"""
Python module of regression with random matrix for CPU.
"""

import sklearn

from .rfflearn_cpu_common import Base


class Regression(Base):
    """
    Regression with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type, dim_kernel=16, std_kernel=0.1, W=None, b=None, **args):
        """
        Constractor. Save hyper parameters as member variables and create LinearRegression instance.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W          (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray): Random bias for the input `X`. If None then generated automatically.
            args       (dict)      : Extra arguments. This arguments will be passed to the constructor of sklearn's LinearRegression model.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.reg = sklearn.linear_model.LinearRegression(**args)

    def fit(self, X, y, **args):
        """
        Trains the RFF regression model according to the given data.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to the sklearn's `fit` function.

        Returns:
            (rfflearn.cpu.Regression): Fitted estimator.
        """
        self.set_weight(X.shape[1])
        self.reg.fit(self.conv(X), y, **args)
        return self

    def predict(self, X, **args):
        """
        Performs prediction on the given data.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            args (dict)      : Extra arguments. This arguments will be passed to the sklearn's `predict` function.

        Returns:
            (np.ndarray): Predicted vector.
        """
        self.set_weight(X.shape[1])
        return self.reg.predict(self.conv(X), **args)

    def score(self, X, y, **args):
        """
        Returns the R2 score (coefficient of determination) of the prediction.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. This arguments will be passed to sklearn's `score` function.

        Returns:
            (float): R2 score of the prediction.
        """
        self.set_weight(X.shape[1])
        return self.reg.score(self.conv(X), y, **args)


# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.


class RFFRegression(Regression):
    """
    Regression with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFRegression(Regression):
    """
    Regression with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
