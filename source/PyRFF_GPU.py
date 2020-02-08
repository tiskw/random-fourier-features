#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 08, 2020
#################################### SOURCE START ###################################

import numpy as np
import sklearn.svm
import sklearn.multiclass
import tensorflow as tf
import PyRFF as pyrff

__version__ = "1.2.0"

def seed(seed):
# {{{

    np.random.seed(seed)
    tf.set_random_seed(seed)

# }}}

class RFFSVC_GPU:
# {{{

    def __init__(self, rffsvc, M_pre = None, batch_size = 32):

        ### Only RFFSVC support GPU inference.
        if type(rffsvc) != pyrff.RFFSVC:
            raise TypeError("PyRFF.RFFSVC_GPU: Only PyRFF.RFFSVC supported.")

        ### TODO: One-versus-one classifier is not supported now.
        if rffsvc.svm.get_params()["estimator__multi_class"] != "ovr":
            raise TypeError("PyRFF.RFFSVC_GPU: Sorry, current implementation support only One-versus-the-rest classifier.")

        ### Create parameters on CPU at first.
        ###   - M: Preprocessing matrix (e.g. Principal Component Analysis).
        ###   - W: Random matrix of Random Fourier Features.
        ###        If PCA applied, combine it to the random matrix for high throughput.
        ###   - A: Coefficients of Linear SVC.
        ###   - b: Intercepts of Linear SVC.
        M_pre = M_pre if M_pre is not None else np.eye(rffsvc.W.shape[0])
        W_cpu = M_pre.dot(rffsvc.W)
        A_cpu = rffsvc.svm.coef_.T
        b_cpu = rffsvc.svm.intercept_.T

        ### Cureate psuedo input on CPU for GPU variable creation.
        x_cpu = np.zeros((batch_size, W_cpu.shape[0]), dtype = np.float32)

        ### Create GPU variables.
        self.x_gpu = tf.Variable(x_cpu, dtype = tf.float32)
        self.W_gpu = tf.constant(W_cpu, dtype = tf.float32)
        self.A_gpu = tf.constant(A_cpu, dtype = tf.float32)
        self.b_gpu = tf.constant(b_cpu, dtype = tf.float32)

        ### Run the GPU model for creating the graph (because we are in the eager-mode here).
        _ = self.predict_proba_batch(x_cpu)

        ### Save several variables.
        self.batch_size  = batch_size
        self.input_shape = x_cpu.shape

    ### Function for running the Tensorflow model of RFF for one batch.
    @tf.function
    def predict_proba_batch(self, x_cpu):
        self.x_gpu.assign(x_cpu)
        z = tf.matmul(self.x_gpu, self.W_gpu)
        z = tf.concat([tf.cos(z), tf.sin(z)], 1)
        return tf.matmul(z, self.A_gpu) + self.b_gpu

    ### Run prediction and return probability (features).
    ### NOTE: Number of data (= X.shape[0]) must be a multiple of batch size.
    def predict_proba(self, X):

        ### Calculate size and number of batch.
        bs = self.batch_size
        bn = X.shape[0] // bs

        ### Batch number must be a multiple of batch size because the Tensorflow Graph already built.
        if X.shape[0] % bs != 0:
            raise ValueError("PyRFF_GPU: Number of input data must be a multiple of batch size (= %d)" % bs)

        ### Run prediction for each batch, concatenate them and return.
        Xs = [self.predict_proba_batch(X[bs*n:bs*(n+1), :]).numpy() for n in range(bn)]
        return np.concatenate(Xs)

    ### Run prediction and return log-probability.
    ### NOTE: Number of data (= X.shape[0]) must be a multiple of batch size.
    def predict_log_proba(self, X, **args):
        return np.log(self.predict_proba(X))

    ### Run prediction and return class label.
    ### NOTE: Number of data (= X.shape[0]) must be a multiple of batch size.
    def predict(self, X, **args):
        return np.argmax(self.predict_proba(X), 1)

    ### Run prediction and return the accuracy of the prediction.
    ### NOTE: Number of data (= X.shape[0]) must be a multiple of batch size.
    def score(self, X, y, **args):
        return np.mean(y == self.predict(X))

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
