#!/usr/bin/env python3
#
# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct 20, 2018
#################################### SOURCE START ###################################

import numpy as np
import scipy.stats
import sklearn.svm
import sklearn.multiclass

__version__ = "1.0"

def seed(seed):
# {{{

    np.random.seed(seed)

# }}}

def conv_RFF(X, W):
# {{{

    ts = np.dot(X, W)
    cs = np.cos(ts)
    ss = np.sin(ts)
    return np.bmat([cs, ss])

# }}}

def select_classifier(mode, svm):
# {{{

    if   mode == "ovo": classifier = sklearn.multiclass.OneVsOneClassifier
    elif mode == "ovr": classifier = sklearn.multiclass.OneVsRestClassifier
    else              : classifier = sklearn.multiclass.OneVsRestClassifier
    return classifier(svm, n_jobs = -1)

# }}}

class RFFRegression:
# {{{

    def __init__(self, dim_output = 1024, std = 0.1, W = None, **args):
        self.dim = dim_output
        self.std = std
        self.reg = sklearn.linear_model.LinearRegression(**args)
        self.W   = W

    def set_weight(self, length):
        if self.W is None:
            self.W = self.std * np.random.randn(length, self.dim)

    def conv(self, X):
        return conv_RFF(X, self.W)

    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        self.reg.fit(self.conv(X), y, **args)
        return self

    def predict(self, X, **args):
        self.set_weight(X.shape[1])
        return self.reg.predict(self.conv(X), **args)

    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.reg.score(self.conv(X), y, **args)

# }}}

class RFFSVC:
# {{{

    def __init__(self, dim_output = 1024, std = 0.1, W = None, multi_mode = "ovr", **args):
        self.dim = dim_output
        self.std = std
        self.W   = W
        self.svm = select_classifier(multi_mode, sklearn.svm.LinearSVC(**args))

    def set_weight(self, length):
        if self.W is None:
            self.W = self.std * np.random.randn(length, self.dim)

    def conv(self, X):
        return conv_RFF(X, self.W)

    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        self.svm.fit(self.conv(X), y, **args)
        return self

    def predict(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svm.predict(self.conv(X), **args)

    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.svm.score(self.conv(X), y, **args)

# }}}

class RFFBatchSVC:
# {{{

    def __init__(self, dim, std, num_epochs = 10, num_batches = 10, alpha = 0.05):
        self.coef    = None
        self.icpt    = None
        self.W       = None
        self.dim     = dim
        self.std     = std
        self.n_epoch = num_epochs
        self.n_batch = num_batches
        self.alpha   = alpha

    def shuffle(self, X, y):
        data_all = np.bmat([X, y.reshape((y.size, 1))])
        np.random.shuffle(X)
        return (data_all[:, :-1], np.ravel(data_all[:, -1]))

    def train_batch(self, X, y, test):

        ### Create classifier instance
        svc = RFFSVC(self.dim, self.std, self.W, **self.args)

        ### Train SVM w/ random fourier features
        svc.fit(X, y)

        ### Update coefficients of linear SVM
        if self.coef is None: self.coef = svc.svm.coef_
        else                : self.coef = self.alpha * svc.svm.coef_ + (1 - self.alpha) * self.coef

        ### Update intercept of linear SVM
        if self.icpt is None: self.icpt = svc.svm.intercept_
        else                : self.icpt = self.alpha * svc.svm.intercept_ + (1 - self.alpha) * self.icpt

        ### Keep random matrices of RFF/ORF
        if self.W is None: self.W = svc.W

    def fit(self, X, y, test = None, **args):

        self.args = args

        ### Calculate batch size
        batch_size = X.shape[0] // self.n_batch

        ### Start training
        for epoch in range(self.n_epoch):
            X, y = self.shuffle(X, y)
            for batch in range(self.n_batch):
                index_bgn = batch_size * (batch + 0)
                index_end = batch_size * (batch + 1)
                self.train_batch(X[index_bgn:index_end, :], y[index_bgn:index_end], test)
                if test is not None:
                    print("Epoch = %d, Batch = %d, Accuracy = %.2f [%%]" % (epoch, batch, 100.0 * self.score(test[0], test[1])))

        return self

    def predict(self, X):
        num = X.shape[0]
        svc = RFFSVC(self.dim, self.std, self.W, **self.args)
        return np.argmax(np.dot(svc.conv(X), self.coef.T) + np.tile(self.icpt.T, (num, 1)), axis = 1)

    def score(self, X, y):
        pred  = self.predict(X)
        return np.mean([(1 if pred[n, 0] == y[n] else 0) for n in range(X.shape[0])])

# }}}

class ORFSVC:
# {{{

    def __init__(self, dim_output, std = 0.1, multi_mode = "ovr", **args):
        self.dim = dim_output
        self.std = std
        self.svm = select_classifier(multi_mode, sklearn.svm.LinearSVC(**args))

    def conv(self, X):
        return conv_RFF(X, self.W)

    def fit(self, X, y, **args):
        self.G = np.random.randn(X.shape[1], self.dim)
        self.S = np.diag(scipy.stats.chi.rvs(X.shape[1], size = X.shape[1]))
        self.Q = np.linalg.qr(self.G)[0]
        self.W = self.std * np.dot(self.S, self.Q)
        self.svm.fit(self.conv(X), y, **args)
        return self

    def predict(self, X, **args):
        return self.svm.predict(self.conv(X), **args)

    def score(self, X, y, **args):
        return self.svm.score(self.conv(X), y, **args)

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
