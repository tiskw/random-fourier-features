# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct  6, 2018
#################################### SOURCE START ###################################

import numpy as np
import sklearn.svm

__version__ = "0.1"

def seed(seed):
# {{{

    np.random.seed(seed)

# }}}

def features(Xs, Ws):
# {{{

    ts = np.dot(Xs, Ws)
    cs = np.cos(ts)
    ss = np.sin(ts)
    return np.bmat([cs, ss])

# }}}

class SVC:
# {{{

    def __init__(self, dim_output, std = 0.1, **args):
        self.dim = dim_output
        self.std = std
        self.svm = sklearn.svm.LinearSVC(**args)

    def conv(self, Xs):
        return features(Xs, self.Ws)

    def fit(self, Xs, ys, **args):
        self.Ws = self.std * np.random.randn(Xs.shape[1], self.dim)
        return self.svm.fit(self.conv(Xs), ys, **args)

    def predict(self, Xs, **args):
        return self.svm.predict(self.conv(Xs), **args)

    def score(self, Xs, ys, **args):
        return self.svm.score(self.conv(Xs), ys, **args)

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
