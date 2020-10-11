#!/usr/bin/env python3
#
# Python module of canonical correlation analysis with random matrix for CPU.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
##################################################### SOURCE START #####################################################


import sklearn.cross_decomposition
from .rfflearn_cpu_common import Base


### Canonival Correlation Analysis with random matrix (RFF/ORF)
class CCA(Base):

    def __init__(self, rand_mat_type, dim_kernel = 128, std_kernel = 0.1, W = None, **args):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.cca = sklearn.cross_decomposition.CCA(**args)

    ### Run training, that is, extract feature vectors and train CCA.
    def fit(self, X, Y):
        self.set_weight((X.shape[1], Y.shape[1]))
        self.cca.fit(self.conv(X, 0), self.conv(Y, 1))
        return self

    ### Return prediction results.
    def predict(self, X, copy = True):
        return self.cca.predict(self.conv(X, 0), copy)

    ### Return evaluation score.
    def score(self, X, Y, sample_weight = None):
        return self.cca.score(self.conv(X, 0), self.conv(Y, 1), sample_weight)

    ### Return transformed results.
    def transform(self, X, Y = None, copy = True):
        return self.cca.transform(self.conv(X, 0), None if Y is None else self.conv(Y, 1), copy)


### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.


### Canonical correlation analysis with RFF.
class RFFCCA(CCA):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


### Canonical correlation analysis with ORF.
class ORFCCA(CCA):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
