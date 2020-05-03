#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 19, 2020
#################################### SOURCE START ###################################

import multiprocessing
import numpy as np
import scipy.stats
import sklearn.svm
import sklearn.multiclass

def seed(seed):
# {{{

    np.random.seed(seed)

# }}}

def select_classifier(mode, svm, n_jobs):
# {{{

    if   mode == "ovo": classifier = sklearn.multiclass.OneVsOneClassifier
    elif mode == "ovr": classifier = sklearn.multiclass.OneVsRestClassifier
    else              : classifier = sklearn.multiclass.OneVsRestClassifier
    return classifier(svm, n_jobs = n_jobs)

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

    def __init__(self, dim_output = 1024, std = 0.1, W = None, multi_mode = "ovr", n_jobs = -1, **args):
        self.dim = dim_output
        self.std = std
        self.W   = W
        self.svm = select_classifier(multi_mode, sklearn.svm.LinearSVC(**args), n_jobs)

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

    def predict_proba(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svm.predict_proba(self.conv(X), **args)

    def predict_log_proba(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svm.predict_log_proba(self.conv(X), **args)

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

class ORFSVC(RFFSVC):
# {{{

    def set_weight(self, length):
        if self.W is None:
            for _ in range(self.dim // length + 1):
                s = scipy.stats.chi.rvs(df = length, size = (length, ))
                Q = np.linalg.qr(np.random.randn(length, length))[0]
                W = self.std * np.dot(np.diag(s), Q)
                if self.W is None: self.W = W
                else             : self.W = np.concatenate([self.W, W], axis = 1)
        self.W = self.W[:length, :self.dim]

# }}}

class RFFGaussianProcessRegression:
# {{{

    def __init__(self, dim_output = 16, std_kernel = 1.0, std_error = 0.1, W = None):
        self.dim = dim_output
        self.s_k = std_kernel
        self.s_e = std_error
        self.W   = W

    def set_weight(self, length):
        if self.W is None:
            self.W = self.s_k * np.random.randn(length, self.dim)

    def conv(self, X):
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        P = F @ F.T
        I = np.eye(2 * self.dim)
        s = self.s_e**2
        M = I - np.linalg.solve((P + s * I), P)
        self.a = (y.T @ F.T) @ M / s
        self.S = P @ M / s
        return self

    def predict(self, X, return_std = False, return_cov = False):
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        p = np.array(self.a.dot(F)).T
        if       return_std and     return_cov: return (p, self.std(F), self.conv(F))
        elif     return_std and not return_cov: return (p, self.std(F))
        elif not return_std and     return_cov: return (p, self.cov(F))
        elif not return_std and not return_cov: return p

    def std(self, F):
        clip_flt = lambda x: max(0.0, float(x))
        pred_var = [clip_flt(F[:, n].T @ (np.eye(2 * self.dim) - self.S) @ F[:, n]) for n in range(F.shape[1])]
        return np.sqrt(np.array(pred_var))

    def cov(self, F):
        return F @ self.S @ F

    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.reg.score(self.conv(X), y, **args)

# }}}

class RFFGaussianProcessClassifier(RFFGaussianProcessRegression):
# {{{

    def __init__(self, dim_output = 16, std_kernel = 5, std_error = 0.3, W = None):
        self.dim = dim_output
        self.s_k = std_kernel
        self.s_e = std_error
        self.W   = W

    def fit(self, Xs, ys):
        ys_onehot = np.eye(int(np.max(ys) + 1))[ys]
        return super().fit(Xs, ys_onehot)

    def predict(self, Xs):
        ps_onehot = super().predict(Xs)
        return np.argmax(ps_onehot, axis = 1)

    def score(self, Xs, ys):
        return np.mean(self.predict(Xs) == ys)

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
