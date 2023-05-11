"""
Python module of principal component analysis with random matrix for CPU.
"""

import sklearn.decomposition

from .rfflearn_cpu_common import Base


class PCA(Base):
    """
    Principal Component Analysis with random matrix (RFF/ORF).
    """
    def __init__(self, rand_type, n_components=None, dim_kernel=128, std_kernel=0.1, W=None, b=None, **args):
        """
        Constractor of PCA class. Save hyperparameters as member variables.

        Args:
            rand_type    (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            n_components (int)       : The number of components to be kept.
            dim_kernel   (int)       : Dimension of the random matrix.
            std_kernel   (float)     : Standard deviation of the random matrix.
            W            (np.ndarray): Random matrix for the input X. If None then generated automatically.
            b            (np.ndarray): Random bias for the input X. If None then generated automatically.
            args         (dict)      : Extra arguments. This dictionaly will be unpacked and passed to PCA class constructor of scikit-learn.
        """
        super().__init__(rand_type, dim_kernel, std_kernel, W, b)
        self.pca = sklearn.decomposition.PCA(n_components, **args)

    def get_covariance(self):
        """
        Wrapper function of sklearn.decomposition.PCA.get_covariance.
        """
        return self.pca.get_covariance()

    def get_precision(self):
        """
        Wrapper function of sklearn.decomposition.PCA.get_precision.
        """
        return self.pca.get_precision()

    def fit(self, X, *pargs, **kwargs):
        """
        Train the PCA model. This function is a wrapper of sklearn.decomposition.PCA.fit.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to fit function of sklearn's PCA model.
            kwargs (dict)      : Extra keywork arguments. This will be passed to fit function of sklearn's PCA model.

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
            pargs  (tuple)     : Extra positional arguments. This will be passed to fit_transform function of sklearn's PCA model.
            kwargs (dict)      : Extra keywork arguments. This will be passed to fit_transform function of sklearn's PCA model.

        Returns:
            (np.ndarray): Myself.
        """
        self.set_weight(X.shape[1])
        return self.pca.fit_transform(self.conv(X), *pargs, **kwargs)

    def score(self, X, *pargs, **kwargs):
        """
        Wrapper function of sklearn.decomposition.PCA.score.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to score function of sklearn's PCA model.
            kwargs (dict)      : Extra keywork arguments. This will be passed to score function of sklearn's PCA model.

        Returns:
            (float): The average log-likelihood of all samples.
        """
        self.set_weight(X.shape[1])
        return self.pca.score(self.conv(X), *pargs, **kwargs)

    def score_samples(self, X, *pargs, **kwargs):
        """
        Wrapper function of sklearn.decomposition.PCA.score_samples.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to score function of sklearn's PCA model.
            kwargs (dict)      : Extra keywork arguments. This will be passed to score function of sklearn's PCA model.

        Returns:
            (np.ndarray): The log-likelihood of each sample with shape (n_samples,).
        """
        self.set_weight(X.shape[1])
        return self.pca.score_samples(self.conv(X), *pargs, **kwargs)

    def transform(self, X, *pargs, **kwargs):
        """
        Wrapper function of sklearn.decomposition.PCA.transform.

        Args:
            X      (np.ndarray): Input matrix with shape (n_samples, n_features_input).
            pargs  (tuple)     : Extra positional arguments. This will be passed to score function of sklearn's PCA model.
            kwargs (dict)      : Extra keywork arguments. This will be passed to score function of sklearn's PCA model.

        Returns:
            (np.ndarray): Transformed X.
        """
        self.set_weight(X.shape[1])
        return self.pca.transform(self.conv(X), *pargs, **kwargs)


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


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
