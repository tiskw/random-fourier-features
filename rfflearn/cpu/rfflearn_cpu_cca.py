#!/usr/bin/env python3
#
# Python module of canonical correlation analysis with random matrix for CPU.
##################################################### SOURCE START #####################################################

import sklearn.cross_decomposition

from .rfflearn_cpu_common import Base

### Canonival Correlation Analysis with random matrix (RFF/ORF)
class CCA:

    def __init__(self, rand_mat_type, dim_kernel = 128, std_kernel = 0.1, W1 = None, W2 = None, **args):
        self.fx1 = Base(rand_mat_type, dim_kernel, std_kernel, W1)
        self.fx2 = Base(rand_mat_type, dim_kernel, std_kernel, W2)
        self.cca = sklearn.cross_decomposition.CCA(**args)

    ### Run training, that is, extract feature vectors and train CCA.
    def fit(self, X, Y):
        self.fx1.set_weight(X.shape[1])
        self.fx2.set_weight(Y.shape[1])
        self.cca.fit(self.fx1.conv(X), self.fx2.conv(Y))
        return self

    ### Return prediction results.
    def predict(self, X, copy = True):
        return self.cca.predict(self.fx1.conv(X), copy)

    ### Return evaluation score.
    def score(self, X, Y, sample_weight = None):
        return self.cca.score(self.fx1.conv(X), self.fx2.conv(Y), sample_weight)

    ### Return transformed results.
    def transform(self, X, Y = None, copy = True):
        return self.cca.transform(self.fx1.conv(X), None if Y is None else self.fx2.conv(Y), copy)

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
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
