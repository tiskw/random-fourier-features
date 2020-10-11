#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
##################################################### SOURCE START #####################################################


import sklearn
from .rfflearn_cpu_common import Base


### Regression with random matrix (RFF/ORF).
class Regression(Base):

    ### Constractor. Save hyper parameters as member variables and create LinearRegression instance.
    def __init__(self, rand_mat_type, dim_kernel = 16, std_kernel = 0.1, W = None, **args):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.reg = sklearn.linear_model.LinearRegression(**args)

    ### Run training, that is, extract feature vectors and train linear regressor.
    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        self.reg.fit(self.conv(X), y, **args)
        return self

    ### Return prediction results.
    def predict(self, X, **args):
        self.set_weight(X.shape[1])
        return self.reg.predict(self.conv(X), **args)

    ### Return evaluation score.
    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.reg.score(self.conv(X), y, **args)


### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.


### Regression with RFF.
class RFFRegression(Regression):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


### Regression with ORF.
class ORFRegression(Regression):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
