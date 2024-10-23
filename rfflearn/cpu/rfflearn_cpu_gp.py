"""
Python module of Gaussian process with random matrix for CPU.
"""

# Declare published functions and variables.
__all__ = ["RFFGPR", "ORFGPR", "QRFGPR", "RFFGPC", "ORFGPC", "QRFGPC"]

# Import 3rd-party packages.
import numpy as np
import sklearn.metrics

# Import custom modules.
from .rfflearn_cpu_common import Base


class GPR(Base):
    """
    Gaussian Process Regression with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type: str, dim_kernel: int = 128, std_kernel: float = 0.1,
                 std_error: float = 0.1, W: np.ndarray = None, b: np.ndarray = None,
                 a: np.ndarray = None, S: np.ndarray = None):
        """
        Constractor of the GPR class.
        Save hyperparameters as member variables.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            std_error  (float)     : Standard deviation of the measurement error.
            W          (np.ndarray): Random matrix for input `X` (generated automatically if None).
            b          (np.ndarray): Random bias for input `X` (generated automatically if None).
            a          (np.ndarray): Cache of the matrix `a` (generated automatically if None).
            S          (np.ndarray): Cache of the matrix `S` (generated automatically if None).
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.s_e = std_error
        self.a   = a
        self.S   = S

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the GPR model according to the given data. The interface of this function imitate
        the interface of the 'sklearn.gaussian_process.GaussianProcessRegressor.fit'.

        Args:
            X (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y (np.ndarray): Output vector with shape (n_samples,).

        Returns:
            (rfflearn.cpu.GPR): Fitted estimator.
        """
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        P = F @ F.T
        I = np.eye(self.kdim)
        M = np.linalg.inv(P + self.s_e**2 * I)
        self.a = (F @ y).T @ M
        self.S = I - P @ M
        return self

    def predict(self, X: np.ndarray, return_std: bool = False,
                return_cov: bool = False) -> np.ndarray:
        """
        Performs regression on the given data.

        Args:
            X          (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            return_std (bool)      : Returns standard deviation of the prediction if True.
            return_cov (bool)      : Returns covariance of the prediction if True.

        Returns:
            (np.ndarray, or tuple): Prediction, or tuple of prediction, standard deviation,
                                    and covariance of the prediction.
        Notes:
            The following is the psuedo code of returned values.
                if return_std and return_cov => return [pred, std, cov]
                if return_std                => return [pred, std]
                if                return_cov => return [pred, cov]
                else                         => return  pred
        """
        self.set_weight(X.shape[1])

        F = self.conv(X).T
        p = np.array(self.a @ F).T
        p = np.squeeze(p, axis = 1) if len(p.shape) > 1 and p.shape[1] == 1 else p

        # Compute standard deviation and covariance of the prediction if necessary.
        if return_std or return_cov:

            # Compute covariance of the prediction.
            cov = F.T @ self.S @ F
            std = np.sqrt(np.diag(cov))

            # Overwrite unnecessary values with None.
            std = std if return_std else None
            cov = cov if return_cov else None

        # Otherwise, initialize as None.
        else:
            std, cov = None, None

        # Returns values.
        res = [v for v in [p, std, cov] if v is not None]
        return res[0] if len(res) == 1 else res

    def score(self, X: np.ndarray, y: np.ndarray, **args: dict) -> float:
        """
        Returns R2 score (coefficient of determination) of the prediction.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output matrix with shape (n_samples, n_features_output).
            args (dict)      : Extra arguments. However, this arguments will be ignored.

        Returns:
            (float): R2 score of the prediction.
        """
        self.set_weight(X.shape[1])
        return sklearn.metrics.r2_score(y, self.predict(X), **args)


class GPC(GPR):
    """
    Gaussian Process Classification with random matrix (RFF/ORF).

    RFFGPC is essentially the same as RFFGPR, but some pre-processing and post-processing
    are necessary. The required processing is:

      - Assumed input label is a class index, but the input of the RFFGPR should be a one hot
        vector of the class index.
      - Output of the RFFGPR is log-prob, not predicted class indexes.

    The purpose of this RFFGPC class is only to do these pre/post-processing.
    """
    def fit(self, X: np.ndarray, y: np.ndarray, **args: dict):
        """
        Trains the GPC model according to the given data. The interface of this function imitate
        the interface of the 'sklearn.gaussian_process.GaussianProcessRegressor.fit'.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. However, this arguments will be ignored.
                               This argument exists only for keeping the same interface
                               with scikit-learn.

        Returns:
            (rfflearn.cpu.GPC): Fitted estimator.
        """
        y_onehot = np.eye(int(np.max(y) + 1))[y]
        return super().fit(X, y_onehot, **args)

    def predict(self, X: np.ndarray, return_std: bool = False,
                return_cov: bool = False) -> np.ndarray:
        """
        Performs classification on the given data.

        Args:
            X          (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            return_std (bool)      : Returns standard deviation of the prediction if True.
            return_cov (bool)      : Returns covariance of the prediction if True.

        Returns:
            (np.ndarray, or tuple): Prediction, or tuple of prediction, standard deviation,
                                    and covariance of the prediction.
        """
        # Run GPC prediction. Note that the returned value is one-hot vector.
        res = super().predict(X, return_std, return_cov)

        # If the prediction does not have standard deviation and covariance matrix,
        # convert one-hot vector to class index and return it.
        if (not return_std) or (not return_cov):
            return np.argmax(res, axis=1)

        # Convert one-hot vector to class index.
        res[0] = np.argmax(res[0], axis=1)
        return res

    def score(self, X: np.ndarray, y: np.ndarray, **args: dict) -> float:
        """
        Returns the mean accuracy on the given data and labels.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments. However, this arguments will be ignored.

        Returns:
            (float): Mean classification accuracy.
        """
        return np.mean(self.predict(X) == y)


####################################################################################################
# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.
####################################################################################################


class RFFGPR(GPR):
    """
    Gaussian process regression with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFGPR(GPR):
    """
    Gaussian process regression with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


class QRFGPR(GPR):
    """
    Gaussian process regression with QRF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


class RFFGPC(GPC):
    """
    Gaussian process classifier with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFGPC(GPC):
    """
    Gaussian process classifier with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


class QRFGPC(GPC):
    """
    Gaussian process classifier with QRF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
