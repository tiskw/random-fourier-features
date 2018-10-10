# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct 10, 2018
#################################### SOURCE START ###################################

import numpy as np
import scipy.stats
import sklearn.svm

__version__ = "0.1"

def seed(seed):
# {{{

    np.random.seed(seed)

# }}}

class rff:
# {{{

    class SVC:

        def __init__(self, dim_output, std = 0.1, **args):
            self.dim = dim_output
            self.std = std
            self.svm = sklearn.svm.LinearSVC(**args)

        def conv(self, X):
            ts = np.dot(X, self.W)
            cs = np.cos(ts)
            ss = np.sin(ts)
            return np.bmat([cs, ss])

        def fit(self, X, y, **args):
            self.W = self.std * np.random.randn(X.shape[1], self.dim)
            return self.svm.fit(self.conv(X), y, **args)

        def predict(self, Xs, **args):
            return self.svm.predict(self.conv(X), **args)

        def score(self, X, y, **args):
            return self.svm.score(self.conv(X), y, **args)

# }}}

class orf:
# {{{

    class SVC:

        def __init__(self, dim_output, std = 0.1, **args):
            self.dim = dim_output
            self.std = std
            self.svm = sklearn.svm.LinearSVC(**args)

        def conv(self, X):
            ts = np.dot(X, self.W)
            cs = np.cos(ts)
            ss = np.sin(ts)
            return np.bmat([cs, ss])

        def fit(self, X, y, **args):
            self.G = np.random.randn(X.shape[1], self.dim)
            self.S = np.diag(scipy.stats.chi.rvs(X.shape[1], size = X.shape[1]))
            self.Q = np.linalg.qr(self.G)[0]
            self.W = self.std * np.dot(self.S, self.Q)
            return self.svm.fit(self.conv(X), y, **args)

        def predict(self, X, **args):
            return self.svm.predict(self.conv(X), **args)

        def score(self, X, y, **args):
            return self.svm.score(self.conv(X), y, **args)

# }}}


#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
