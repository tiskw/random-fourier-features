"""
Python module of principal component analysis with random matrix for CPU.
"""

import numpy as np
import torch

from .rfflearn_gpu_common import Base, detect_device


class LinearPCA:
    """
    Linear Principal Component Analysis on GPU.
    This class designed to have similar interface to sklearn.decomposition.PCA.
    """
    def __init__(self, n_components, nniter=50):
        """
        Constractor: Store necessary parameters (on CPU).

        Args:
            n_components (int): The number of components to be kept.
            n_iter       (int): The number of subspace iterations to conduct.
        """
        # Hyper parameters for principal component analysis.
        self.n_components_ = n_components
        self.n_iter = n_iter

        # Intermediate values in the computation of PCA.
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

        # Device to be used.
        self.dev = detect_device()

    def fit(self, X_cpu, y_cpu=None):
        """
        Training of principal component analysis.

        Args:
            X_cpu (np.ndarray): Training data with shape = (n_samples, n_features).
            y_cpu (np.ndarray): Ignored.

        Returns:
            (rfflearn.gpu.LinearPCA): Myself.
        """
        # Input matrix `X_cpu` sould have member variable `.shape` and
        # the length of the shape should be 2 (i.e. 2-dimensional matrix).
        if not (hasattr(X_cpu, "shape") and len(X_cpu.shape) == 2):
            raise RuntimeError("PCA.fit: input variable should be 2-dimensional matrix.")

        # Move variable to GPU.
        X = torch.from_numpy(X_cpu).to(self.dev)

        # Calculate PCA. For getting stable results, subtract average before hand.
        m = torch.mean(X, dim = 0)
        U, S, V = torch.pca_lowrank(X - m, self.n_components_, center=False, niter=self.n_iter)

        # Store the PCA results as NumPy array on CPU.
        self.mean_ = m.cpu().numpy()
        self.components_ = V.cpu().numpy().T
        self.explained_variance_ = np.square(S.cpu().numpy()) / (X_cpu.shape[0] - 1)

        return self

    def fit_transform(self, X_cpu, *pargs, **kwargs):
        """
        Train and apply the PCA trandform to `X_cpu`.

        Args:
            X_cpu  (np.ndarray): Input matrix with shape (n_samples, n_features).
            pargs  (tuple)     : Extra Positional arguments. This dictionaly will be unpacked and passed to `fit_transform` function of sklearn's PCA class.
            kwargs (dict)      : Extra keyword arguments. This dictionaly will be unpacked and passed to `fit_transform` function of sklearn's PCA class.

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_components).
        """
        return self.fit(X_cpu, *pargs, **kwargs).transform(X_cpu)

    def inverse_transform(self, Z_cpu):
        """
        Inverse of PCA transformation.

        Args:
            Z_cpu (np.ndarray): Output matrix with shape (n_samples, n_components).

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_features) that is inversely computed.
        """
        Z = torch.from_numpy(Z_cpu).to(self.dev)
        m = torch.from_numpy(self.mean_).to(self.dev)
        W = torch.from_numpy(self.components_).to(self.dev)
        X = torch.matmul(Z, W) + m
        return X.cpu().numpy()

    def transform(self, X_cpu):
        """
        Apply PCA transform.

        Args:
            X_cpu (np.ndarray): Input matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_components).
        """
        X = torch.from_numpy(X_cpu).to(self.dev)
        m = torch.from_numpy(self.mean_).to(self.dev)
        W = torch.from_numpy(self.components_.T).to(self.dev)
        Z = torch.matmul(X - m, W)
        return Z.cpu().numpy()


class PCA(Base):
    """
    Principal Component Analysis with random matrix (RFF/ORF).
    This class designed to have similar interface to sklearn.decomposition.PCA.
    """
    def __init__(self, rand_type, n_components=None, dim_kernel=128, std_kernel=0.1, W=None, b=None, **args):
        """
        Constractor. Save hyperparameters as member variables.

        Args:
            rand_type    (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            n_components (int)       : The number of components to be kept.
            dim_kernel   (int)       : Dimension of the random matrix.
            std_kernel   (float)     : Standard deviation of the random matrix.
            W            (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            args         (dict)      : Extra arguments. This dictionaly will be unpacked and passed to PCA class constructor of scikit-learn.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.pca = LinearPCA(n_components, **args)

    def fit(self, X, *pargs, **kwargs):
        """
        Training of principal component analysis.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features).
            pargs  (tuple)     : Extra Positional arguments. This dictionaly will be unpacked and passed to `fit` function of sklearn's PCA class.
            kwargs (dict)      : Extra keyword arguments. This dictionaly will be unpacked and passed to `fit` function of sklearn's PCA class.
        """
        self.set_weight(X.shape[1])
        self.pca.fit(self.conv(X), *pargs, **kwargs)
        return self

    def fit_transform(self, X, *pargs, **kwargs):
        """
        Train and apply the PCA trandform to `X`.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features).
            pargs  (tuple)     : Extra Positional arguments. This dictionaly will be unpacked and passed to `fit_transform` function of sklearn's PCA class.
            kwargs (dict)      : Extra keyword arguments. This dictionaly will be unpacked and passed to `fit_transform` function of sklearn's PCA class.

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_components).
        """
        self.set_weight(X.shape[1])
        return self.pca.fit_transform(self.conv(X), *pargs, **kwargs)

    def transform(self, X):
        """
        Apply PCA transform.

        Args:
            X (np.ndarray): Input matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_components).
        """
        self.set_weight(X.shape[1])
        return self.pca.transform(self.conv(X), *pargs, **kwargs)

    def inverse_transform(self, Z):
        """
        Inverse of PCA transformation.

        Args:
            Z (np.ndarray): Output matrix with shape (n_samples, n_components).

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_features) that is inversely computed.
        """
        self.set_weight(X.shape[1])
        U = self.pca.inverse_transform(Z).cpu().numpy()
        return np.arccos((U - b) @ np.linalg.inv(self.W))


# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.


class RFFPCA(PCA):
    """
    Principal component analysis with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFPCA(PCA):
    """
    Principal component analysis with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


class QRFPCA(PCA):
    """
    Principal component analysis with Quasi-RRF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
