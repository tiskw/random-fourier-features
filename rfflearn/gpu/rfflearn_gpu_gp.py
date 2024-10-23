"""
Python module of Gaussian process with random matrix for GPU.
"""

# Declare published functions and variables.
__all__ = ["RFFGPR", "ORFGPR", "QRFGPR", "RFFGPC", "ORFGPC", "QRFGPC"]

# Import 3rd-party packages.
import numpy as np
import sklearn.metrics
import torch

# Import custom modules.
from .rfflearn_gpu_common import Base, detect_device


class GPR(Base):
    """
    This class provides the RFF based Gaussian Process classification using GPU.
    This class has the following member functions. See the comments and code below for details.
      - fit(self, X_cpu, y_cpu)  : run training using training data X_cpu and y_cpu
      - predict(self, X_cpu)     : run inference (this function also be able to return variance)
      - score(self, X_cpu, y_cpu): run inference and return the overall accuracy
    """
    def __init__(self, rand_type: str, dim_kernel: int = 256, std_kernel: float = 0.1,
                 std_error: float = 1.0, W: np.ndarray = None, b: np.ndarray = None,
                 a: np.ndarray = None, S: np.ndarray = None):
        """
        Constractor of the GPR class.
        Save hyperparameters as member variables (on CPU).

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
        self.s_e    = std_error
        self.device = detect_device()
        self.a      = a
        self.S      = S

    def fit(self, X_cpu, y_cpu):
        """
        Training of Gaussian process using GPU.

        Args:
            X_cpu (np.ndarray): Training data with shape (n_samples, n_features_input).
            y_cpu (np.ndarray): Training label with shape = (n_samples, n_features_output).

        Retuens:
            (rfflearn.gpu.GPR): Myself.
        """
        # Generate random matrix.
        self.set_weight(X_cpu.shape[1])

        # Generate random matrix W and identity matrix I on CPU.
        I_cpu = np.eye(self.kdim)

        # Derive posterior distribution 1/3 (on CPU).
        s_cpu = self.s_e**2
        F_cpu = self.conv(X_cpu).T
        P_cpu = F_cpu @ F_cpu.T

        # Derive posterior distribution 2/3 (on GPU).
        Q_gpu = torch.tensor(P_cpu + s_cpu * I_cpu, device=self.device, dtype=torch.float64)
        M_cpu = torch.inverse(Q_gpu).cpu().numpy()

        # Derive posterior distribution 3/3 (on CPU).
        self.a = (F_cpu @ y_cpu).T @ M_cpu
        self.S = I_cpu - P_cpu @ M_cpu

        # Clean GPU memory.
        del Q_gpu
        if "cuda" in self.device:
            torch.cuda.empty_cache()

        return self

    def predict(self, X_cpu, return_std=False, return_cov=False):
        """
        Inference of Gaussian process using GPU.

        Args:
            X_cpu       (np.ndarray) : Inference data with shape (n_samples, n_features_input).
            return_std  (boolean)    : Returns standard deviation vector if True.
            return_cov  (boolean)    : Returns covariance matrix if True.

        Returns:
            (np.ndarray, or tuple): Prediction, or tuple of prediction, std, and cov.

        Notes:
            The following is the psuedo code of returned values.
                if return_std and return_cov => return [pred, std, cov]
                if return_std                => return [pred, std]
                if                return_cov => return [pred, cov]
                else                         => return  pred
        """
        # Move matrices to GPU.
        # NOTE: 64bit precision is necessary for training, however,
        #       32bit precision is enough for inference in most cases.
        X = torch.tensor(X_cpu,  device=self.device, dtype=torch.float32)
        W = torch.tensor(self.W, device=self.device, dtype=torch.float32)
        b = torch.tensor(self.b, device=self.device, dtype=torch.float32)
        a = torch.tensor(self.a, device=self.device, dtype=torch.float32)
        S = torch.tensor(self.S, device=self.device, dtype=torch.float32)

        # Calculate mean of the prediction distribution.
        F = torch.cos(torch.matmul(X, W) + b).t()
        p = torch.matmul(a, F).t()

        # Move prediction value to CPU.
        # If shape of y_cpu is (*, 1), then reshape to (*, ).
        p_cpu = p.cpu().numpy()
        if (len(p_cpu.shape) > 1) and (p_cpu.shape[1] == 1):
            p_cpu = np.squeeze(p_cpu, axis=1)

        # Compute standard deviation and covariance of the prediction if necessary.
        if return_std or return_cov:

            # Compute covariance of the prediction.
            cov_cpu = torch.matmul(torch.matmul(F.t(), S), F).cpu().numpy()
            std_cpu = np.sqrt(np.diag(cov_cpu))

            # Overwrite unnecessary values with None.
            std_cpu = std_cpu if return_std else None
            cov_cpu = cov_cpu if return_cov else None

        # Otherwise, initialize as None.
        else:
            std_cpu, cov_cpu = None, None

        # Clean GPU memory.
        del X, W, a, S, F, p
        if "cuda" in self.device:
            torch.cuda.empty_cache()

        # Returns values.
        res = [v for v in [p_cpu, std_cpu, cov_cpu] if v is not None]
        return res[0] if len(res) == 1 else res

    def score(self, X_cpu, y_cpu):
        """
        Calculate score of the given inference data.

        Args:
            X_cpu (np.ndarray): Test data with shape (n_samples, n_features_input).
            y_cpu (np.ndarray): Test label with shape (n_samples, n_features_out).

        Returns:
            (float): R2 score of regression.
        """
        return sklearn.metrics.r2_score(y_cpu, self.predict(X_cpu))


class GPC(GPR):
    """
    This class provides the RFF based Gaussian Process classification using GPU.
    This class has the following member functions. See the comments and code below for details.
      - fit(self, X_cpu, y_cpu)  : run training using training data X_cpu and y_cpu
      - predict(self, X_cpu)     : run inference (this function also be able to return variance)
      - score(self, X_cpu, y_cpu): run inference and return the overall accuracy
    """
    def fit(self, X_cpu, y_cpu):
        """
        Training of Gaussian process using GPU.

        Args:
            X_cpu (np.ndarray): Test data with shape (n_samples, n_features).
            y_cpu (np.ndarray): Test label with shape (n_samples,).

        Returns:
            (rfflearn.gpu.GPC): Myself.
        """
        y_onehot_cpu = np.eye(int(np.max(y_cpu) + 1))[y_cpu]
        return super().fit(X_cpu, y_onehot_cpu)

    def predict(self, X_cpu, return_std=False, return_cov=False):
        """
        Inference of Gaussian process using GPU.

        Args:
            X_cpu      (np.ndarray): Inference data with shape = [N, K])
            return_std (boolean)   : Return standard deviation vector if True.
            return_cov (boolean)   : Return covariance matrix if True.
        """
        # Run GPC prediction. Note that the returned value is one-hot vector.
        res = super().predict(X_cpu, return_std, return_cov)

        # If the prediction does not have standard deviation and covariance matrix,
        # convert one-hot vector to class index and return it.
        if (not return_std) or (not return_cov):
            return np.argmax(res, axis=1)

        # Convert one-hot vector to class index.
        res[0] = np.argmax(res[0], axis=1)
        return res

    def score(self, X_cpu, y_cpu):
        """
        Calculate score of the given inference data.

        Args:
            X_cpu (np.array): Test data with shape (n_samples, n_features).
            y_cpu (np.array): Test label with shape (n_samples,).

        Returns:
            (float): Classification accuracy.
        """
        return np.mean(self.predict(X_cpu) == y_cpu)


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
    Gaussian process regression with Quasi-RFF.
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
    Gaussian process classifier with Quasi-RRF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
