"""
Python module of principal component analysis with random matrix for CPU.
"""

# Declare published functions and variables.
__all__ = ["RFFPCA", "ORFPCA", "QRFPCA"]

# Import 3rd-party packages.
import numpy as np
import sklearn.decomposition

# Import custom modules.
from .rfflearn_cpu_common import Base


class PCA(Base):
    """
    Principal Component Analysis with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type: str, n_components: int = None, dim_kernel: int = 128,
                 std_kernel: float = 0.1, W: np.ndarray = None, b: np.ndarray = None, **args: dict):
        """
        Constractor of PCA class. Save hyperparameters as member variables.

        Args:
            rand_type    (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            n_components (int)       : The number of components to be kept.
            dim_kernel   (int)       : Dimension of the random matrix.
            std_kernel   (float)     : Standard deviation of the random matrix.
            W            (np.ndarray): Random matrix for input X (generated automatically if None).
            b            (np.ndarray): Random bias for input X (generated automatically if None).
            args         (dict)      : Extra arguments. This dictionaly will be unpacked and passed
                                       to PCA class constructor of scikit-learn.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.pca = sklearn.decomposition.PCA(n_components, **args)

    def fit(self, X, *pargs, **kwargs):
        """
        Trains the PCA model. This function is a wrapper of sklearn.decomposition.PCA.fit.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to fit function
                                 of sklearn's PCA model instance.
            kwargs (dict)      : Extra keywork arguments. This will be passed to fit function
                                 of sklearn's PCA model instance.

        Returns:
            (rfflearn.cpu.PCA): Myself.
        """
        self.set_weight(X.shape[1])
        self.pca.fit(self.conv(X), *pargs, **kwargs)
        return self

    def fit_transform(self, X, *pargs, **kwargs):
        """
        Wrapper function of sklearn.decomposition.PCA.fit_transform.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to fit_transform
                                 function of sklearn's PCA model instance.
            kwargs (dict)      : Extra keywork arguments. This will be passed to fit_transform
                                 function of sklearn's PCA model instance.

        Returns:
            (np.ndarray): Myself.
        """
        self.set_weight(X.shape[1])
        return self.pca.fit_transform(self.conv(X), *pargs, **kwargs)

    def transform(self, X, *pargs, **kwargs):
        """
        Wrapper function of sklearn.decomposition.PCA.transform.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to score
                                 function of sklearn's PCA model instance.
            kwargs (dict)      : Extra keywork arguments. This will be passed to score
                                 function of sklearn's PCA model instance.

        Returns:
            (np.ndarray): Transformed X.
        """
        self.set_weight(X.shape[1])
        return self.pca.transform(self.conv(X), *pargs, **kwargs)

    def inverse_transform(self, Z, *pargs, **kwargs):
        """
        Inverse of PCA transformation.

        Args:
            Z (np.ndarray): Output matrix with shape (n_samples, n_components).

        Returns:
            (np.ndarray): Input matrix with shape (n_samples, n_features) that is
                          inversely computed.
        """
        return self.pca.inverse_transform(Z, *pargs, **kwargs)


####################################################################################################
# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.
####################################################################################################


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
    Principal component analysis with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
