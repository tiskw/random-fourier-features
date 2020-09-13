#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 05, 2020
##################################################### SOURCE START #####################################################

import multiprocessing
import numpy as np
import scipy.stats
import sklearn.decomposition
import sklearn.svm
import sklearn.multiclass

### Fix random seed used in this script.
def seed(seed):
# {{{

    ### Now it is enough to fix the random seed of Numpy.
    np.random.seed(seed)

# }}}

### Regression using Random Fourier Features.
class RFFRegression:
# {{{

    ### Constractor. Save hyper parameters as member variables and create LinearRegression instance.
    ### NOTE: If 'W' is None then the appropriate matrix will be set just before the training.
    def __init__(self, dim_kernel = 1024, std = 0.1, W = None, **args):
        self.dim = dim_kernel
        self.std = std
        self.reg = sklearn.linear_model.LinearRegression(**args)
        self.W   = W

    ### Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).
    def set_weight(self, length):
        if self.W is None:
            self.W = self.std * np.random.randn(length, self.dim)

    ### Apply random matrix to the given input vectors 'X' and create feature vectors.
    def conv(self, X):
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

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

# }}}

### Support Vector Classification using Random Fourier Features.
class RFFSVC:
# {{{

    ### Constractor. Save hyper parameters as member variables and create LinearSVC instance.
    ### The LinearSVC instance is always wrappered by multiclass classifier.
    def __init__(self, dim_kernel = 1024, std = 0.1, W = None, multi_mode = "ovr", n_jobs = -1, **args):
        self.dim = dim_kernel
        self.std = std
        self.W   = W
        self.svm = self.set_classifier(sklearn.svm.LinearSVC(**args), multi_mode, n_jobs)

    ### Select multiclass classifire. Now this function can handle one-vs-one and one-vs-others.
    def set_classifier(self, svm, mode, n_jobs):
        if   mode == "ovo": classifier = sklearn.multiclass.OneVsOneClassifier
        elif mode == "ovr": classifier = sklearn.multiclass.OneVsRestClassifier
        else              : classifier = sklearn.multiclass.OneVsRestClassifier
        return classifier(svm, n_jobs = n_jobs)

    ### Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).
    def set_weight(self, length):
        if self.W is None:
            self.W = self.std * np.random.randn(length, self.dim)

    ### Apply random matrix to the given input vectors 'X' and create feature vectors.
    def conv(self, X):
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

    ### Run training, that is, extract feature vectors and train SVC.
    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        self.svm.fit(self.conv(X), y, **args)
        return self

    ### Return predicted probability for each target classes.
    def predict_proba(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svm.predict_proba(self.conv(X), **args)

    ### Return predicted log-probability for each target classes.
    def predict_log_proba(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svm.predict_log_proba(self.conv(X), **args)

    ### Return prediction results.
    def predict(self, X, **args):
        self.set_weight(X.shape[1])
        return self.svm.predict(self.conv(X), **args)

    ### Return evaluation score.
    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.svm.score(self.conv(X), y, **args)

# }}}

### Support Vector Classification using Random Fourier Features with batch training.
class RFFBatchSVC:
# {{{

    ### Constractor. Save hyper parameters as member variables and create LinearSVC instance.
    ### The LinearSVC instance is always wrappered by multiclass classifier.
    def __init__(self, dim_kernel, std, num_epochs = 10, num_batches = 10, alpha = 0.05):
        self.coef    = None
        self.icpt    = None
        self.W       = None
        self.dim     = dim_kernel
        self.std     = std
        self.n_epoch = num_epochs
        self.n_batch = num_batches
        self.alpha   = alpha

    ### Shuffle the order of the training data.
    def shuffle(self, X, y):
        data_all = np.bmat([X, y.reshape((y.size, 1))])
        np.random.shuffle(X)
        return (data_all[:, :-1], np.ravel(data_all[:, -1]))

    ### Train only one batch. This function will be called from the 'fit' function.
    def train_batch(self, X, y, test, **args):

        ### Create classifier instance
        svc = RFFSVC(self.dim, self.std, self.W, **args)

        ### Train SVM with random fourier features
        svc.fit(X, y)

        ### Update coefficients of linear SVM
        if self.coef is None: self.coef = svc.svm.coef_
        else                : self.coef = self.alpha * svc.svm.coef_ + (1 - self.alpha) * self.coef

        ### Update intercept of linear SVM
        if self.icpt is None: self.icpt = svc.svm.intercept_
        else                : self.icpt = self.alpha * svc.svm.intercept_ + (1 - self.alpha) * self.icpt

        ### Keep random matrices of RFF/ORF
        if self.W is None: self.W = svc.W

    ### Run training.
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

    ### Return prediction results.
    def predict(self, X, **args):
        svc = RFFSVC(self.dim, self.std, self.W, **args)
        return np.argmax(np.dot(svc.conv(X), self.coef.T) + self.icpt.flatten(), axis = 1)

    ### Return score.
    def score(self, X, y, **args):
        pred  = self.predict(X)
        return np.mean([(1 if pred[n, 0] == y[n] else 0) for n in range(X.shape[0])])

# }}}

### Support Vector Classification using Orthogonal Random Features with batch training.
class ORFSVC(RFFSVC):
# {{{

    ### The difference between RFF and ORF is just the way to generate the random matrix 'W'.
    ### Therefore it is wnough to overwrite the function 'set_weight' in order to realize the
    ### ORF classification.
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

### Gaussian Process Regression using Random Fourier Features.
class RFFGPR:
# {{{

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, dim_kernel = 16, std_kernel = 1.0, std_error = 0.1, W = None):
        self.dim = dim_kernel
        self.s_k = std_kernel
        self.s_e = std_error
        self.W   = W

    ### Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).
    def set_weight(self, length):
        if self.W is None:
            self.W = self.s_k * np.random.randn(length, self.dim)

    ### Apply random matrix to the given input vectors 'X' and create feature vectors.
    def conv(self, X):
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

    ### Run training. The interface of this function imitate the interface of
    ### the 'sklearn.gaussian_process.GaussianProcessRegressor.fit'.
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

    ### Run prediction. The interface of this function imitate the interface of
    ### the 'sklearn.gaussian_process.GaussianProcessRegressor.predict'.
    def predict(self, X, return_std = False, return_cov = False):
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        p = np.array(self.a.dot(F)).T
        if       return_std and     return_cov: return (p, self.std(F), self.conv(F))
        elif     return_std and not return_cov: return (p, self.std(F))
        elif not return_std and     return_cov: return (p, self.cov(F))
        elif not return_std and not return_cov: return p

    ### Return predicted standard deviation.
    def std(self, F):
        clip_flt = lambda x: max(0.0, float(x))
        pred_var = [clip_flt(F[:, n].T @ (np.eye(2 * self.dim) - self.S) @ F[:, n]) for n in range(F.shape[1])]
        return np.sqrt(np.array(pred_var))

    ### Return predicted covariance.
    def cov(self, F):
        return F @ self.S @ F

    ### Return score.
    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return self.reg.score(self.conv(X), y, **args)

# }}}

### Gaussian Process Classification using Random Fourier Features.
class RFFGPC(RFFGPR):
# {{{

    ### RFFGPC is essentially the same as RFFGPR, but some pre-processing and post-processing are necessary.
    ### The required processings are:
    ###   - Assumed input label is a vector of class indexes, but the input of
    ###     the RFFGPR should be a one hot vector of the class indexes.
    ###   - Output of the RFFGPR is log-prob, not predicted class indexes.
    ### The purpouse of this RFFGPC class is only to do these pre/post-processings.

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, dim_kernel = 16, std_kernel = 5, std_error = 0.3, W = None):
        self.dim = dim_kernel
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

### Principal Component Analysis using Random Fourier Features.
class RFFPCA:
# {{{

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, n_components = None, dim_kernel = 16, std_kernel = 0.1, W = None):
        self.dim = dim_kernel
        self.std = std_kernel
        self.pca = sklearn.decomposition.PCA(n_components)
        self.W   = W

    ### Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).
    def set_weight(self, length):
        if self.W is None:
            self.W = self.std * np.random.randn(length, self.dim)

    ### Apply random matrix to the given input vectors 'X' and create feature vectors.
    def conv(self, X):
        ts = np.dot(X, self.W)
        cs = np.cos(ts)
        ss = np.sin(ts)
        return np.bmat([cs, ss])

    ### Run training, that is, extract feature vectors and train SVC.
    def fit(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        self.pca.fit(self.conv(X), y, **args)
        return self

    def fit_transform(self, X, *pargs, **kwargs):
        self.set_weight(X.shape[1])
        return self.pca.fit_transform(self.conv(X), *pargs, **kwargs)

# }}}

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
