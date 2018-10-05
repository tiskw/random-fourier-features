# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct  6, 2018
#################################### SOURCE START ###################################

import numpy as np
import sklearn.svm

__version__ = "0.0.0"

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

    def __init__(self, dim_output):
        self.dim = dim_output
        self.svm = sklearn.svm.SVC(kernel = "linear")

    def conv(self, Xs):
        return features(Xs, self.Ws)

    def fit(self, Xs, ys):
        self.Ws = np.random.randn(Xs.shape[1], self.dim)
        return self.svm.fit(self.conv(Xs), ys)

    def predict(self, Xs):
        return self.svm.predict(self.conv(Xs))

    def score(self, Xs, ys):
        return self.svm.score(self.conv(Xs), ys)

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
