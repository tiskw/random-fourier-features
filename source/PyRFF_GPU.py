#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 08, 2020
######################################### SOURCE START ########################################

import numpy as np
import sklearn.svm
import sklearn.multiclass
import tensorflow as tf
import PyRFF as pyrff

def seed(seed):
# {{{

    np.random.seed(seed)
    tf.set_random_seed(seed)

# }}}

### This class provides the RFFSVC classification using GPU.
### However, this class support only inference now.
### This class feeds a trained PyRFF.RFFSVC instance, read trained parameters from the class
### and run inference on GPU.
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

class RFFGaussianProcessClassifier_GPU:
# {{{

    def __init__(self, dim_output = 256, std_kernel = 0.1, std_error = 1.0, W = None):
        self.dim = dim_output
        self.s_k = std_kernel
        self.s_e = std_error
        self.W   = W

    ### Generate random metrix for RFF if not set.
    def set_weight(self, length):
        if self.W is None:
            self.W = self.s_k * np.random.randn(length, self.dim)

    ### Training of gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): training data
    ###   - y_cpu (np.array, shape = [N, C]): training label
    ### where N is the number of training data, K is dimention of the input data, and C is the number of classes.
    def fit(self, X_cpu, y_cpu):

        ### Generate random matrix.
        self.set_weight(X_cpu.shape[1])

        ### Convert the input label vector to the one hot format.
        y_oh = np.eye(int(np.max(y_cpu) + 1))[y_cpu]

        ### Generate random matrix W and identity matrix I on CPU.
        I_cpu = np.eye(2 * self.dim)

        ### Move matrices to GPU.
        W = tf.constant(self.W, dtype = tf.float64)
        I = tf.constant(I_cpu,  dtype = tf.float64)
        X = tf.constant(X_cpu,  dtype = tf.float64)
        y = tf.constant(y_oh,   dtype = tf.float64)

        Z   = tf.matmul(X, W)
        F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
        F   = tf.transpose(F_T)
        P   = tf.matmul(F, F_T)
        s   = (self.s_e)**2

        M = I - tf.linalg.solve((P + s * I), P)
        a = tf.matmul(tf.matmul(tf.transpose(y), F_T), M) / s
        S = tf.matmul(P, M) / s

        self.a = a.numpy()
        self.S = S.numpy()

    ### Inference of gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): inference data
    ###   - var   (boolean)                 : returns variance vector if true
    ###   - cov   (boolean)                 : returns covariance matrix if true
    ### where N is the number of training data and K is dimention of the input data.
    def predict(self, X_cpu, var = False, cov = False):

        ### Move matrix to GPU.
        X = tf.constant(X_cpu,  dtype = tf.float64)
        W = tf.constant(self.W, dtype = tf.float64)
        a = tf.constant(self.a, dtype = tf.float64)
        S = tf.constant(self.S, dtype = tf.float64)

        Z   = tf.matmul(X, W)
        F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
        F   = tf.transpose(F_T)

        p = tf.matmul(a, F)
        p = tf.transpose(p)
        V = tf.matmul(tf.matmul(F_T, S), F)

        ### Convert prediction one-hot vector to the class label.
        y = tf.argmax(p, axis = 1)

        if   var and cov: return (y.numpy(), np.diag(V.numpy()), V.numpy())
        elif var        : return (y.numpy(), np.diag(V.numpy()))
        elif cov        : return (y.numpy(), V.numpy())
        else            : return (y.numpy())

    ### calculate score of the given inference data.
    ###   - X_cpu (np.array, shape = [N, K]): test data
    ###   - y_cpu (np.array, shape = [N, C]): test label
    ### where N is the number of training data, K is dimention of the input data, and C is the number of classes.
    def score(self, X_cpu, y_cpu):
        return np.mean(self.predict(X_cpu) == y_cpu)

# }}}

######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
