#!/usr/bin/env python3
#
# Python module of principal component analysis with random matrix for CPU.
##################################################### SOURCE START #####################################################

import numpy as np
import torch

from .rfflearn_gpu_common import Base, detect_device

### Linear Principal Component Analysis on GPU.
### This class designed to have similar interface to sklearn.decomposition.PCA.
class LinearPCA:

    ### Constractor: Store necessary parameters (on CPU).
    def __init__(self, n_components, niter = 50):

        ### Hyper parameters for principal component analysis.
        self.n_components_ = n_components
        self.niter = niter

        ### Intermediate values in the computation of PCA.
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

        ### Device to be used.
        self.dev = detect_device()

    ### Training of principal component analysis.
    ###   - X_cpu (np.array, shape = [N, K]): training data
    ###   - y_cpu (none)                    : ignored
    ###   - transform (bool)                : if True, return transformed input
    ### where N is the number of training data, K is dimension of the input data.
    def fit(self, X_cpu, y_cpu = None, transform = False):

        ### Input matrix `X_cpu` sould have member variable `.shape` and
        ### the length of the shape should be 2 (i.e. 2-dimensional matrix).
        if not (hasattr(X_cpu, "shape") and len(X_cpu.shape) == 2):
            raise RuntimeError("PCA.fit: input variable should be 2-dimensional matrix.")

        ### Move variable to GPU.
        X = torch.from_numpy(X_cpu).to(self.dev)

        ### Calculate PCA. For getting stable results, subtract average before hand.
        m = torch.mean(X, dim = 0)
        U, S, V = torch.pca_lowrank(X - m, self.n_components_, center = False, niter = self.niter)

        ### Store the PCA results as NumPy array on CPU.
        self.mean_ = m.cpu().numpy()
        self.components_ = V.cpu().numpy().T
        self.explained_variance_ = np.square(S.cpu().numpy()) / (X_cpu.shape[0] - 1)

        ### Return transform results of `X_cpu` if `transform` is True.
        if transform: return torch.matmul(X - m, V).cpu().numpy()
        else        : return self

    ### Train and apply the PCA trandform to `X_cpu`.
    def fit_transform(self, X_cpu, *pargs, **kwargs):
        return self.fit(X_cpu, transform = True, *pargs, **kwargs)

    ### Inverse of PCA transformation.
    def inverse_transform(self, Z_cpu):
        Z = torch.from_numpy(Z_cpu).to(self.dev)
        m = torch.from_numpy(self.mean_).to(self.dev)
        W = torch.from_numpy(self.components_).to(self.dev)
        X = torch.matmul(Z, W) + m
        return X.cpu().numpy()

    ### Apply PCA transform.
    def transform(self, X_cpu):
        X = torch.from_numpy(X_cpu).to(self.dev)
        m = torch.from_numpy(self.mean_).to(self.dev)
        W = torch.from_numpy(self.components_.T).to(self.dev)
        Z = torch.matmul(X - m, W)
        return Z.cpu().numpy()

### Principal Component Analysis with random matrix (RFF/ORF).
### This class designed to have similar interface to sklearn.decomposition.PCA.
class PCA(Base):

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, rand_mat_type, n_components = None, dim_kernel = 128, std_kernel = 0.1, W = None, **args):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.pca = LinearPCA(n_components, **args)

    ### Wrapper function of sklearn.decomposition.PCA.fit.
    def fit(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        self.pca.fit(self.conv(X), *pargs, **kwargs)
        return self

    ### Wrapper function of sklearn.decomposition.PCA.fit_transform.
    def fit_transform(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        return self.pca.fit_transform(self.conv(X), *pargs, **kwargs)

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

### Principal component analysis with Quasi-RRF.
class QRFPCA(PCA):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
