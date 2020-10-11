#!/usr/bin/env python3
#
# Python module of Gaussian process with random matrix for GPU.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
######################################### SOURCE START ########################################


import numpy as np
import sklearn.metrics
import tensorflow as tf
from .rfflearn_gpu_common import Base


### This class provides the RFF based Gaussian Process classification using GPU.
### This class has the following member functions. See the comments and code below for details.
###   - fit(self, X_cpu, y_cpu)  : run training using training data X_cpu and y_cpu
###   - predict(self, X_cpu)     : run inference (this function also be able to return variance)
###   - score(self, X_cpu, y_cpu): run inference and return the overall accuracy
class GPR(Base):

    def __init__(self, rand_mat_type, dim_kernel = 256, std_kernel = 0.1, std_error = 1.0, scale = 1.0, W = None):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.s_e = std_error

    ### Training of Gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): training data
    ###   - y_cpu (np.array, shape = [N, C]): training label
    ### where N is the number of training data, K is dimension of the input data, and C is dimention of output data.
    def fit(self, X_cpu, y_cpu):

        ### Generate random matrix.
        self.set_weight(X_cpu.shape[1])

        ### Generate random matrix W and identity matrix I on CPU.
        I_cpu = np.eye(2 * self.dim)

        ### Derive posterior distribution 1/3 (on CPU).
        s = (self.s_e)**2
        F = self.conv(X_cpu).T
        P = F @ F.T

        ### Derive posterior distribution 2/3 (on GPU).
        P1 = tf.constant(P + s * I_cpu, dtype = tf.float64)
        M  = I_cpu - tf.linalg.solve(P1, P).numpy()

        ### Derive posterior distribution 3/3 (on CPU).
        self.a = ((y_cpu.T @ F.T) @ M) / s
        self.S = P @ M / s

    ### Inference of Gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): inference data
    ###   - var   (boolean, scalar)         : returns variance vector if true
    ###   - cov   (boolean, scalar)         : returns covariance matrix if true
    ### where N is the number of training data and K is dimension of the input data.
    def predict(self, X_cpu, return_var = False, return_cov = False):

        ### Move matrix to GPU.
        X = tf.constant(X_cpu,  dtype = tf.float64)
        W = tf.constant(self.W, dtype = tf.float64)
        a = tf.constant(self.a, dtype = tf.float64)
        S = tf.constant(self.S, dtype = tf.float64)

        ### Calculate mean of the prediction distribution.
        Z   = tf.matmul(X, W)
        F_T = tf.concat([tf.cos(Z), tf.sin(Z)], 1)
        F   = tf.transpose(F_T)
        p   = tf.transpose(tf.matmul(a, F))

        ### Move prediction value to CPU.
        y_cpu = p.numpy()

        ### Calculate covariance matrix and variance vector.
        if return_var or return_cov:
            V_cpu = tf.matmul(tf.matmul(F_T, S), F).numpy()
            v_cpu = np.diag(V)

        if return_var and return_cov: return (y_cpu, v_cpu, V_cpu)
        elif return_var             : return (y_cpu, v_cpu)
        elif return_cov             : return (y_cpu, V_cpu)
        else                        : return  y_cpu

    ### calculate score of the given inference data.
    ###   - X_cpu (np.array, shape = [N, K]): test data
    ###   - y_cpu (np.array, shape = [N, C]): test label
    ### where N is the number of training data, K is dimension of the input data, and C is dimention of output data.
    def score(self, X_cpu, y_cpu):
        return sklearn.metrics.r2_score(y_cpu, self.predict(X_cpu))


### This class provides the RFF based Gaussian Process classification using GPU.
### This class has the following member functions. See the comments and code below for details.
###   - fit(self, X_cpu, y_cpu)  : run training using training data X_cpu and y_cpu
###   - predict(self, X_cpu)     : run inference (this function also be able to return variance)
###   - score(self, X_cpu, y_cpu): run inference and return the overall accuracy
class GPC(GPR):

    ### Training of Gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): test data
    ###   - y_cpu (np.array, shape = [N,  ]): test label
    ### where N is the number of training data, K is dimension of the input data.
    def fit(self, X_cpu, y_cpu):
        y_onehot_cpu = np.eye(int(np.max(y_cpu) + 1))[y_cpu]
        return super().fit(X_cpu, y_onehot_cpu)

    ### Inference of Gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): inference data
    ###   - var   (boolean, scalar)         : returns variance vector if true
    ###   - cov   (boolean, scalar)         : returns covariance matrix if true
    ### where N is the number of training data and K is dimension of the input data.
    def predict(self, X_cpu, return_var = False, return_cov = False):

        ### Run GPC prediction. Note that the returned value is a one-hot vector.
        res = super().predict(X_cpu, return_var, return_cov)

        ### Convert one-hot vector to class index.
        if not return_var and not return_cov: res    = np.argmax(res,    axis = 1)
        else                                : res[0] = np.argmax(res[0], axis = 1)

        return res

    ### calculate score of the given inference data.
    ###   - X_cpu (np.array, shape = [N, K]): test data
    ###   - y_cpu (np.array, shape = [N,  ]): test label
    ### where N is the number of training data, K is dimension of the input data.
    def score(self, X_cpu, y_cpu):
        return np.mean(self.predict(X_cpu) == y_cpu)


### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.


### Gaussian process regression with RFF.
class RFFGPR(GPR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


### Gaussian process regression with ORF.
class ORFGPR(GPR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


### Gaussian process classifier with RFF.
class RFFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


### Gaussian process classifier with ORF.
class ORFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


### This class provides the function to estimate hyper parameter of the RFF based GP.
### In this class, kernel is assumed to be given by:
###     kernel(x1, x2) = scale * exp( - |x1 - x2|^2 / (2 * s_k^2) ),
### where s_k and scale are scalar variables.
### This class will estimate s_k, scale and s_e where s_e is a standard deviation of the measurement.
### NOTE: Optimization is conducted under log scale of each hyper parameters,
###       because all parameters should be positive values.
class GPKernelParameterEstimator:

    ### This function conduct the optimization of the hyper parameters, s_k, s_e and scale.
    ###   - epoch_max (int, scalar)         : number of epochs
    ###   - batch_size (int, scalar)        : batch size (number of data in one batch)
    ###   - lr (float, scalar)              : learning rate
    ###   - ini_s_k (float, scalar)         : initial value of the s_k (if None, randomly initialized)
    ###   - ini_s_e (float, scalar)         : initial value of the s_e (if None, randomly initialized)
    ###   - ini_scale (float, scalar)       : initial value of the scale (if None, randomly initialized)
    ###   - reg (float, scalar)             : coefficient of the L2 regularization term
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
    ###   - OTHERS                          : same as constructor and will be overwrite if given.
    ### where N is the number of training data, K is dimension of the input data, and C is the number of classes.
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


######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
