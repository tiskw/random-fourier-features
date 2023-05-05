#!/usr/bin/env python3
#
# Python module of principal component analysis with random matrix for CPU.
##################################################### SOURCE START #####################################################

import sklearn.decomposition

from .rfflearn_cpu_common import Base

### Principal Component Analysis with random matrix (RFF/ORF).
class PCA(Base):

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, rand_mat_type, n_components = None, dim_kernel = 128, std_kernel = 0.1, W = None, b = None, **args):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W, b)
        self.pca = sklearn.decomposition.PCA(n_components, **args)

    ### Wrapper function of sklearn.decomposition.PCA.get_covariance.
    def get_covariance(self):
        return self.pca.get_covariance()

    ### Wrapper function of sklearn.decomposition.PCA.get_precision.
    def get_precision(self):
        return self.pca.get_precision()

    ### Wrapper function of sklearn.decomposition.PCA.fit.
    def fit(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        self.pca.fit(self.conv(X), *pargs, **kwargs)
        return self

    ### Wrapper function of sklearn.decomposition.PCA.fit_transform.
    def fit_transform(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        return self.pca.fit_transform(self.conv(X), *pargs, **kwargs)

    ### Wrapper function of sklearn.decomposition.PCA.score.
    def score(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        return self.pca.score(self.conv(X), *pargs, **kwargs)

    ### Wrapper function of sklearn.decomposition.PCA.score_samples.
    def score_samples(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        return self.pca.score_samples(self.conv(X), *pargs, **kwargs)

    ### Wrapper function of sklearn.decomposition.PCA.transform.
    def transform(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        return self.pca.transform(self.conv(X), *pargs, **kwargs)

### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.

### Principal component analysis with RFF.
class RFFPCA(PCA):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Principal component analysis with ORF.
class ORFPCA(PCA):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
