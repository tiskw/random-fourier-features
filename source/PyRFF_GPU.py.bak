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

### This class provides the RFF based SVC classification using GPU.
### However, this class support only inference now.
### This class feeds a trained PyRFF.RFFSVC instance, read trained parameters from the class
### and run inference on GPU.
### NOTE: Number of data (= X_cpu.shape[0]) must be a multiple of batch size.
class RFFSVC:
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
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimention of the input data.
    @tf.function
    def predict_proba_batch(self, X_cpu):
        self.x_gpu.assign(X_cpu)
        z = tf.matmul(self.X_gpu, self.W_gpu)
        z = tf.concat([tf.cos(z), tf.sin(z)], 1)
        return tf.matmul(z, self.A_gpu) + self.b_gpu

    ### Run prediction and return probability (features).
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimention of the input data.
    def predict_proba(self, X_cpu):

        ### Calculate size and number of batch.
        bs = self.batch_size
        bn = X_cpu.shape[0] // bs

        ### Batch number must be a multiple of batch size because the Tensorflow Graph already built.
        if X_cpu.shape[0] % bs != 0:
            raise ValueError("PyRFF_GPU: Number of input data must be a multiple of batch size (= %d)" % bs)

        ### Run prediction for each batch, concatenate them and return.
        Xs = [self.predict_proba_batch(X_cpu[bs*n:bs*(n+1), :]).numpy() for n in range(bn)]
        return np.concatenate(Xs)

    ### Run prediction and return log-probability.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimention of the input data.
    def predict_log_proba(self, X_cpu, **args):
        return np.log(self.predict_proba(X_cpu))

    ### Run prediction and return class label.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimention of the input data.
    def predict(self, X_cpu, **args):
        return np.argmax(self.predict_proba(X_cpu), 1)

    ### Run prediction and return the accuracy of the prediction.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ###   - y_cpu (np.array, shape = [N, C]): training label,
    ### where N is the number of training data, K is dimention of the input data, and C is the number of classes.
    def score(self, X_cpu, y_cpu, **args):
        return np.mean(y_cpu == self.predict(X_cpu))

# }}}

### This class provides the RFF based Gaussian Process classification using GPU.
### This class has the following member functions. See the comments and code below for details.
###   - fit(self, X_cpu, y_cpu)  : run training using training data X_cpu and y_cpu
###   - predict(self, X_cpu)     : run inference (this function also be able to return variance)
###   - score(self, X_cpu, y_cpu): run inference and return the overall accuracy
class RFFGPC:
# {{{

    def __init__(self, dim_output = 256, std_kernel = 0.1, std_error = 1.0, scale = 1.0, W = None):
        self.dim = dim_output
        self.s_k = std_kernel
        self.s_e = std_error
        self.sca = scale
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

        ### Derive posterior distribution.
        s   = (self.s_e)**2
        c   = np.sqrt(self.sca)
        Z   = tf.matmul(X, W)
        F_T = c * tf.concat([tf.cos(Z), tf.sin(Z)], 1)
        F   = tf.transpose(F_T)
        P   = tf.matmul(F, F_T)
        M   = I - tf.linalg.solve((P + s * I), P)
        a   = tf.matmul(tf.matmul(tf.transpose(y), F_T), M) / s
        S   = tf.matmul(P, M) / s

        ### Save parameters for posterior distribution as np.array.
        self.a = a.numpy()
        self.S = S.numpy()

    ### Inference of gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): inference data
    ###   - var   (boolean, scalar)         : returns variance vector if true
    ###   - cov   (boolean, scalar)         : returns covariance matrix if true
    ### where N is the number of training data and K is dimention of the input data.
    def predict(self, X_cpu, var = False, cov = False):

        ### Move matrix to GPU.
        X = tf.constant(X_cpu,  dtype = tf.float64)
        W = tf.constant(self.W, dtype = tf.float64)
        a = tf.constant(self.a, dtype = tf.float64)
        S = tf.constant(self.S, dtype = tf.float64)

        ### Calculate mean of the prediction distribution.
        Z   = tf.matmul(X, W)
        F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
        F   = tf.transpose(F_T)
        p   = tf.matmul(a, F)
        p   = tf.transpose(p)

        ### Convert prediction vector (one-hot vector) to the class label.
        y_cpu = tf.argmax(p, axis = 1).numpy()

        ### Return prediction vector only if both variance and covariance are not required.
        if not var and not cov:
            return y_cpu

        ### Calculate covariance matrix and variance vector.
        V_cpu = tf.matmul(tf.matmul(F_T, S), F).numpy()
        v_cpu = np.diag(V)

        if var and cov: return (y_cpu, v_cpu, V_cpu)
        elif var      : return (y_cpu, v_cpu)
        elif cov      : return (y_cpu, V_cpu)
        else          : raise RuntimeError("Unexpected if branch.")

    ### calculate score of the given inference data.
    ###   - X_cpu (np.array, shape = [N, K]): test data
    ###   - y_cpu (np.array, shape = [N, C]): test label
    ### where N is the number of training data, K is dimention of the input data, and C is the number of classes.
    def score(self, X_cpu, y_cpu):
        return np.mean(self.predict(X_cpu) == y_cpu)

# }}}

### This class provides the function to estimate hyper parameter of the RFF based GP.
### In this class, kernel is assumed to be given by:
###     kernel(x1, x2) = scale * exp( -|x1 - x2|^2 / (2 * s_k^2) ),
### where s_k and scale are scalar variables.
### This class will estimate s_k, scale and s_e where s_e is a standard deviation of the measurement.
### NOTE: Optimization is conducted under log scale of each hyper parameters,
###       because all parameters should be positive values.
class GPKernelParameterEstimator:
# {{{

    ### This function conduct the optimization of the hyper parameters, s_k, s_e and scale.
    ###   - epoch_max (int, scalar)         : number of epochs
    ###   - batch_size (int, scalar)        : batch size (number of data in one batch)
    ###   - lr (float, scalar)              : learning rate
    ###   - ini_s_k (float, scalar)         : initial value of the s_k (if None, randomly initialized)
    ###   - ini_s_e (float, scalar)         : initial value of the s_e (if None, randomly initialized)
    ###   - ini_scale (float, scalar)       : initial value of the scale (if None, randomly initialized)
    ###   - reg (float, scalar)             : coefficient of the L2 regulalization term
    def __init__(self, epoch_max, batch_size, lr, ini_s_k = None, ini_s_e = None, ini_scale = None, reg = 1.0):
        self.e_max  = epoch_max
        self.b_size = batch_size
        self.lr     = lr
        self.s_k    = ini_s_k
        self.s_e    = ini_s_e
        self.sca    = ini_scale
        self.reg    = reg
        self.eps    = 1.0E-12

    ### This function conduct the optimization of the hyper parameters, s_k, s_e and scale.
    ###   - X_cpu (np.array, shape = [N, K]): training data
    ###   - y_cpu (np.array, shape = [N, C]): training label
    ###   - OTHERS                          : same as constractor and will be overwrite if given.
    ### where N is the number of training data, K is dimention of the input data, and C is the number of classes.
    def fit(self, X_cpu, y_cpu, epoch_max = None, batch_size = None, lr = None):

        ### Generate data batch for training.
        def batch(X_all, y_all, batch_size):
            num_dat = X_all.shape[0]
            indexes = np.random.choice(range(num_dat), num_dat)
            for n in range(0, num_dat, batch_size):
                Is = indexes[n:n+batch_size]
                Xs = tf.constant(X_all[Is, :], tf.float64)
                ys = tf.constant(y_all[Is],    tf.int32)
                yield (Xs, ys)

        @tf.function
        def train_step(Xs, ys, trainable_variables, opt):

            N = Xs.shape[0]
            I = tf.eye(N, dtype = tf.float64)

            with tf.GradientTape() as tape:

                tape.watch(trainable_variables)

                ### Unpack tuple of trainable variables.
                s_k_ln, s_e_ln, sca_ln = trainable_variables

                ### Restore original hyper parameter.
                s_k = tf.exp(s_k_ln)
                s_e = tf.exp(s_e_ln)
                sca = tf.exp(sca_ln)

                ### Calculate covariance matrix of the Gaussian Process model.
                X = tf.stack([Xs] * N)
                D = tf.reduce_sum(X - tf.transpose(X, perm = [1, 0, 2]), axis = -1)**2
                V = sca * tf.exp(- D / (2 * s_k**2)) + s_e * I

                ### Calculate one hot vector.
                y = tf.one_hot(ys, n_classes, dtype = tf.float64)

                ### Calculate loss function.
                loss = tf.math.log(tf.linalg.det(V) + self.eps)
                for c in range(n_classes):
                    yc = y[:, c:c+1]
                    loss += tf.matmul(yc, tf.matmul(tf.linalg.pinv(V), yc), transpose_a = True) / n_classes
                loss += (s_k**2 + s_e**2 + sca**2) / (2 * self.reg**2)

            ### Calculate gradient and apply.
            grads = tape.gradient(loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))

            return loss

        ### Update hyper parameters if given.
        self.e_max  = epoch_max  if epoch_max  else self.e_max
        self.b_size = batch_size if batch_size else self.b_size
        self.lr     = lr         if lr         else self.lr

        ### Raise error if initial values is not set.
        if not all([self.s_k, self.s_e, self.sca]):
            print("Warning: At least one of the optimization variable has no initial value.")
            print("         The variable will be initialized by random value.")

        ### Predict number of classes.
        n_classes = np.max(y_cpu) + 1

        ### Initial value of the hyper parameters (after applying natural logarithm).
        ini_s_k_ln = np.log(self.s_k if self.s_k else np.random.uniform(0.1, 1.0))
        ini_s_e_ln = np.log(self.s_e if self.s_e else np.random.uniform(0.1, 1.0))
        ini_sca_ln = np.log(self.sca if self.sca else np.random.uniform(0.1, 1.0))

        ### Create variables.
        s_k_ln = tf.Variable(ini_s_k_ln, dtype=tf.float64)
        s_e_ln = tf.Variable(ini_s_e_ln, dtype=tf.float64)
        sca_ln = tf.Variable(ini_sca_ln, dtype=tf.float64)
        trainable_variables = (s_k_ln, s_e_ln, sca_ln)

        ### Creator optimizer instance.
        opt = tf.keras.optimizers.SGD(self.lr, momentum=0.9)

        ### Variable for calculating the moving average of the loss values.
        loss_ave = 0.0

        for epoch in range(self.e_max):

            for step, (Xs, ys) in enumerate(batch(X_cpu, y_cpu, self.b_size)):

                ### Run training and update loss moving average.
                loss_val = train_step(Xs, ys, trainable_variables, opt)
                loss_ave = 0.95 * loss_ave + 0.05 * float(loss_val.numpy())

                ### Store intermediate results.
                self.s_k = float(np.exp(s_k_ln.numpy()))
                self.s_e = float(np.exp(s_e_ln.numpy()))
                self.sca = float(np.exp(sca_ln.numpy()))

                ### Print current hyper parameter.
                if step % 10 == 0:
                    format_str = "loss = %.3e, s_k = %.3e, s_e = %.3e, scale = %.3e (epoch %d step %d)"
                    print(format_str % (loss_ave, self.s_k, self.s_e, self.sca, epoch, step))

        return (self.s_k, self.s_e, self.sca)

# }}}

######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
