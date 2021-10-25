#!/usr/bin/env python3
#
# Python module of support vector classification with random matrix for CPU.
##################################################### SOURCE START #####################################################

import numpy as np
import sklearn.svm
import sklearn.multiclass

from .rfflearn_cpu_common import Base

### Support vector classification with random matrix (RFF/ORF).
class SVC(Base):

    ### Constractor. Save hyper parameters as member variables and create LinearSVC instance.
    ### The LinearSVC instance is always wrappered by multiclass classifier.
    def __init__(self, rand_mat_type, dim_kernel = 128, std_kernel = 0.1, W = None, multi_mode = "ovr", n_jobs = -1, **args):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.svm = self.set_classifier(sklearn.svm.LinearSVC(**args), multi_mode, n_jobs)

    ### Select multiclass classifire. Now this function can handle one-vs-one and one-vs-others.
    def set_classifier(self, svm, mode, n_jobs):
        if   mode == "ovo": classifier = sklearn.multiclass.OneVsOneClassifier
        elif mode == "ovr": classifier = sklearn.multiclass.OneVsRestClassifier
        else              : classifier = sklearn.multiclass.OneVsRestClassifier
        return classifier(svm, n_jobs = n_jobs)

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

### Batch training extention of the support vector classification.
class BatchSVC:

    ### Constractor. Save hyper parameters as member variables and create LinearSVC instance.
    ### The LinearSVC instance is always wrappered by multiclass classifier.
    def __init__(self, rand_mat_type, dim_kernel, std_kernel, num_epochs = 10, num_batches = 10, alpha = 0.05):
        self.rtype   = rand_mat_type
        self.coef    = None
        self.icpt    = None
        self.W       = None
        self.dim     = dim_kernel
        self.std     = std_kernel
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
        if   self.rtype == "rff": svc = RFFSVC(self.dim, self.std, self.W, **args)
        elif self.rtype == "orf": svc = ORFSVC(self.dim, self.std, self.W, **args)
        else                    : raise RuntimeError("BatchSVC: 'rand_mat_type' must be 'rff' or 'orf'.")

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

### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.

### Support vector machine with RFF.
class RFFSVC(SVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Support vector machine with ORF.
class ORFSVC(SVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

### Support vector machine with QRF.
class QRFSVC(SVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

### Support vector machine with RFF.
class RFFBatchSVC(BatchSVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Support vector machine with ORF.
class ORFBatchSVC(BatchSVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

### Support vector machine with QRF.
class QRFBatchSVC(BatchSVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
