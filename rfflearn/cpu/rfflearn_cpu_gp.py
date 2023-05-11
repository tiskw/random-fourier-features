"""
Python module of Gaussian process with random matrix for CPU.
"""

import numpy as np
import sklearn.metrics

from .rfflearn_cpu_common import Base


class GPR(Base):
    """
    Gaussian Process Regression with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type, dim_kernel=128, std_kernel=0.1, std_error=0.1, W=None, b=None, a=None, S=None):
        """
        Constractor of the GPR class.
        Save hyperparameters as member variables.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            std_error  (float)     : Standard deviation of the measurement error.
            W          (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray): Random bias for the input `X`. If None then generated automatically.
            a          (np.ndarray): Cache of the matrix `a`. If None then generated automatically.
            S          (np.ndarray): Cache of the matrix `S`. If None then generated automatically.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.s_e = std_error
        self.a   = a
        self.S   = S

    def fit(self, X, y, **args):
        """
        Run the training process. The interface of this function imitate the interface of
        the 'sklearn.gaussian_process.GaussianProcessRegressor.fit'.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments, however, this arguments will be ignored.

        Returns:
            (rfflearn.cpu.GPR): Myself.
        """
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        P = F @ F.T
        I = np.eye(self.dim)
        s = self.s_e**2
        M = I - np.linalg.solve((P + s * I), P)
        self.a = (y.T @ F.T) @ M / s
        self.S = I - P @ M / s
        return self

    def predict(self, X, return_std=False, return_cov=False):
        """
        Run prediction. The interface of this function imitate the interface of
        the 'sklearn.gaussian_process.GaussianProcessRegressor.predict'.
        If shape of the vector p is (*, 1), then reshape to (*, ).

        Args:
            X          (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            return_std (bool)      : Returns standard deviation of the prediction if True.
            return_cov (bool)      : Returns covariance of the prediction if True.

        Returns:
            (np.ndarray, or tuple): Prediction, or tuple of prediction, std, and cov.
        """
        self.set_weight(X.shape[1])

        F = self.conv(X).T
        p = np.array(self.a.dot(F)).T
        p = np.squeeze(p, axis = 1) if len(p.shape) > 1 and p.shape[1] == 1 else p

        if return_std and return_cov: return [p, self.std(F), self.cov(F)]
        elif return_std             : return [p, self.std(F)]
        elif return_cov             : return [p, self.cov(F)]
        else                        : return  p

    def std(self, F):
        """
        Returns standard deviation of prediction.

        Args:
            F (np.ndarray): Matrix F (= self.conv(X).T).

        Returns:
            (np.ndarray): Standard deviation of the prediction.
        """
        clip_flt = lambda x: max(0.0, float(x))
        pred_var = [clip_flt(F[:, n].T @ self.S @ F[:, n]) for n in range(F.shape[1])]
        return np.sqrt(np.array(pred_var))

    def cov(self, F):
        """
        Returns covariance of prediction.

        Args:
            F (np.ndarray): Matrix F (= self.conv(X).T).

        Returns:
            (np.ndarray): Covariance of the prediction.
        """
        return F.T @ self.S @ F

    def score(self, X, y, **args):
        """
        Return R2 score of the regression result.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments, however, this arguments will be ignored.

        Returns:
            (float): R2 score of the regression.
        """
        self.set_weight(X.shape[1])
        return sklearn.metrics.r2_score(y, self.predict(X))


class GPC(GPR):
    """
    Gaussian Process Classification with random matrix (RFF/ORF).

    RFFGPC is essentially the same as RFFGPR, but some pre-processing and post-processing are necessary.
    The required processings are:
      - Assumed input label is a vector of class indexes, but the input of
        the RFFGPR should be a one hot vector of the class indexes.
      - Output of the RFFGPR is log-prob, not predicted class indexes.
    The purpouse of this RFFGPC class is only to do these pre/post-processings.
    """
    def __init__(self, rand_type, dim_kernel=128, std_kernel=0.1, std_error=0.1, W=None, b=None, a=None, S=None):
        """
        Constractor. Save hyperparameters as member variables.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            std_error  (float)     : Standard deviation of the measurement error.
            W          (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray): Random bias for the input `X`. If None then generated automatically.
            a          (np.ndarray): Cache of the matrix `a`. If None then generated automatically.
            S          (np.ndarray): Cache of the matrix `S`. If None then generated automatically.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, std_error, W, b, a, S)

    def fit(self, X, y):
        """
        Run the training process. The interface of this function imitate the interface of
        the 'sklearn.gaussian_process.GaussianProcessRegressor.fit'.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments, however, this arguments will be ignored.

        Returns:
            (rfflearn.cpu.GPC): Myself.
        """
        y_onehot = np.eye(int(np.max(y) + 1))[y]
        return super().fit(X, y_onehot)

    def predict(self, X, return_std=False, return_cov=False):
        """
        Run prediction. The interface of this function imitate the interface of
        the 'sklearn.gaussian_process.GaussianProcessRegressor.predict'.
        If shape of the vector p is (*, 1), then reshape to (*, ).

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

        # Convert one-hot vector to class index.
        if return_std or return_cov: res[0] = np.argmax(res[0], axis = 1)
        else                       : res    = np.argmax(res,    axis = 1)

        return res

    def score(self, X, y, **args):
        """
        Returns classification accuracy.

        Args:
            X    (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            y    (np.ndarray): Output vector with shape (n_samples,).
            args (dict)      : Extra arguments, however, this arguments will be ignored.

        Returns:
            (float): Classification accuracy.
        """
        return np.mean(self.predict(X) == y)


# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.


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


##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
