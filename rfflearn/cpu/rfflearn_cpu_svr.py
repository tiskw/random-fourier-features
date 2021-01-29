#!/usr/bin/env python3
#
# Python module of support vector regression with random matrix for CPU.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 29, 2021
##################################################### SOURCE START #####################################################

import sklearn.svm
import sklearn.multiclass
from .rfflearn_cpu_common import Base

### Support vector regression with random matrix (RFF/ORF).
class SVR(Base):

    ### Constractor. Save hyper parameters as member variables and create LinearSVR instance.
    def __init__(self, rand_mat_type, dim_kernel = 128, std_kernel = 0.1, W = None, **args):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.svr = sklearn.svm.LinearSVR(**args)

    ### Run training, that is, extract feature vectors and train SVR.
    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        self.svr.fit(self.conv(X), y, **args)
        return self

    ### Return prediction results.
    def predict(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svr.predict(self.conv(X), **args)

    ### Return evaluation score.
    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.svr.score(self.conv(X), y, **args)

### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.

### Support vector machine with RFF.
class RFFSVR(SVR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Support vector machine with ORF.
class ORFSVR(SVR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
