#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Nov 16, 2019
#################################### SOURCE START ###################################

import numpy as np
import scipy.stats
import sklearn.svm
import sklearn.multiclass

__version__ = "1.1"

def seed(seed):
# {{{

    np.random.seed(seed)

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
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

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
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

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

    def train_batch(self, X, y, test, **args):

        ### Create classifier instance
        svc = RFFSVC(self.dim, self.std, self.W, **args)

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

        ### Calculate batch size
        batch_size = X.shape[0] // self.n_batch

        ### Start training
        for epoch in range(self.n_epoch):
            X, y = self.shuffle(X, y)
            for batch in range(self.n_batch):
                index_bgn = batch_size * (batch + 0)
                index_end = batch_size * (batch + 1)
                self.train_batch(X[index_bgn:index_end, :], y[index_bgn:index_end], test, **args)
                if test is not None:
                    print("Epoch = %d, Batch = %d, Accuracy = %.2f [%%]" % (epoch, batch, 100.0 * self.score(test[0], test[1], **args)))

        return self

    def predict(self, X, **args):
        svc = RFFSVC(self.dim, self.std, self.W, **args)
        return np.argmax(np.dot(svc.conv(X), self.coef.T) + self.icpt.flatten(), axis = 1)

    def score(self, X, y, **args):
        pred  = self.predict(X)
        return np.mean([(1 if pred[n, 0] == y[n] else 0) for n in range(X.shape[0])])

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
