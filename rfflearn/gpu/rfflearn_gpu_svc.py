#!/usr/bin/env python3
#
# Python module of support vector classification with random matrix for CPU.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
######################################### SOURCE START ########################################

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from .rfflearn_gpu_common import Base

### This class provides the RFF based SVC classification using GPU.
### NOTE: Number of data (= X_cpu.shape[0]) must be a multiple of batch size.
class SVC(Base):

    ### Create parameters on CPU, then move the parameters to GPU.
    ### There are two ways to initialize these parameters:
    ###   (1) from scratch: generate parameters from scratch,
    ###   (2) from rffsvc: copy parameters from RFFSVC (CPU) class instance.
    ### The member variable 'self.initialized' indicate that the parameters are well initialized or not.
    ### If the parameters are initialized by one of the ways other than (1), 'self.initialized' is set to True.
    ### And if 'self.initialized' is still False when just before the training/inference,
    ### then the parameters are initialized by the way (1).
    def __init__(self, rand_mat_type, svc = None, M_pre = None, dim_kernel = 128, std_kernel = 0.1, W = None, batch_size = 5000, dtype = 'float32', *pargs, **kwargs):

        ### Save important variables.
        self.dim_kernel  = dim_kernel
        self.std         = std_kernel
        self.batch_size  = batch_size
        self.dtype       = dtype
        self.initialized = False
        self.W           = W

        ### Inisialize variables.
        if svc: self.init_from_RFFSVC_cpu(svc, M_pre)
        else  : super().__init__(rand_mat_type, dim_kernel, std_kernel, W)

    ### Constractor: initialize parameters from scratch.
    def init_from_scratch(self, dim_input, dim_kernel, dim_output, std):

        ### Generate random matrix.
        self.set_weight(dim_input)

        ### Create parameters on CPU.
        ###   - W: Random matrix of Random Fourier Features. (shape = [dim_input,  dim_kernel]).
        ###   - A: Coefficients of Linear SVC.               (shape = [dim_kernel, dim_output]).
        ###   - b: Intercepts of Linear SVC.                 (shape = [1,          dim_output]).
        self.W_cpu = self.W
        self.A_cpu = np.random.randn(2 * dim_kernel, dim_output)
        self.b_cpu = np.random.randn(1,              dim_output)

        ### Create GPU variables and build GPU model.
        self.build()

        ### Mark as initialized.
        self.initialized = True

    ### Copy parameters from the given rffsvc and create instance.
    def init_from_RFFSVC_cpu(self, svc_cpu, M_pre):

        ### Only RFFSVC support GPU inference.
        if not hasattr(svc_cpu, "W") or not hasattr(svc_cpu, "svm") or not hasattr(svc_cpu.svm, "coef_") or not hasattr(svc_cpu.svm, "intercept_"):
            raise TypeError("rfflearn.gpu.SVC: Only rfflearn.cpu.SVC supported.")

        ### TODO: One-versus-one classifier is not supported now.
        if svc_cpu.svm.get_params()["estimator__multi_class"] != "ovr":
            raise TypeError("rfflearn.gpu.SVC: Sorry, current implementation support only One-versus-the-rest classifier.")

        ### Copy parameters from rffsvc on CPU.
        ###   - W: Random matrix of Random Fourier Features.
        ###        If PCA applied, combine it to the random matrix for high throughput.
        ###   - A: Coefficients of Linear SVC.
        ###   - b: Intercepts of Linear SVC.
        self.W_cpu = M_pre.dot(svc_cpu.W) if M_pre is not None else svc_cpu.W
        self.A_cpu = svc_cpu.svm.coef_.T
        self.b_cpu = svc_cpu.svm.intercept_.T

        ### Create GPU variables and build GPU model.
        self.build()

        ### Mark as initialized.
        self.initialized = True

    ### Create GPU variables if all CPU variables are ready.
    def build(self):

        ### Run build procedure if and only if all variables is available.
        if all(v is not None for v in [self.W_cpu, self.A_cpu, self.b_cpu]):

            ### Create pseudo input on CPU for GPU variable creation.
            self.X_cpu = np.zeros((self.batch_size, self.W_cpu.shape[0]), dtype = self.dtype)
            self.y_cpu = np.zeros((self.batch_size, self.A_cpu.shape[1]), dtype = self.dtype)

            ### Create GPU variables.
            self.X_gpu = tf.Variable(self.X_cpu, dtype = self.dtype)
            self.y_gpu = tf.Variable(self.y_cpu, dtype = self.dtype)
            self.W_gpu = tf.constant(self.W_cpu, dtype = self.dtype)
            self.A_gpu = tf.Variable(self.A_cpu, dtype = self.dtype)
            self.b_gpu = tf.Variable(self.b_cpu, dtype = self.dtype)

            ### Run a inference for building the model (because we are in the eager-mode here).
            _ = self.predict_proba_batch(self.X_cpu)

    ### Train the model for one batch.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimension of the input data.
    @tf.function
    def fit_batch(self, X_gpu, y_gpu, opt):

        ### Calclate loss function under the GradientTape.
        with tf.GradientTape() as tape:
            tape.watch([self.A_gpu, self.b_gpu])
            z = tf.matmul(X_gpu, self.W_gpu)
            z = tf.concat([tf.cos(z), tf.sin(z)], 1)
            p = tf.matmul(z, self.A_gpu) + self.b_gpu
            v = tf.math.reduce_mean(tf.math.maximum(tf.cast(0, self.dtype), 1 - y_gpu * p))

        ### Derive gradient for all variables and apply gradient decent optimizer.
        dv_dA, dv_db = tape.gradient(v, [self.A_gpu, self.b_gpu])
        opt.apply_gradients(zip([dv_dA, dv_db], [self.A_gpu, self.b_gpu]))

        ### Register loss value to the metric.
        self.losses(v)

    def fit(self, X_cpu, y_cpu, epoch_max = 1000, opt = "RAdam", learning_rate = 0.1, weight_decay = 1.0E-8, quiet = False):

        ### Get optimizer.
        if   opt == "SGD"    : opt = tf.keras.optimizers.SGD(learning_rate)
        elif opt == "RMSProp": opt = tf.keras.optimizers.RMSprop(learning_rate)
        elif opt == "Adam"   : opt = tf.keras.optimizers.Adam(learning_rate)
        elif opt == "AdamW"  : opt = tfa.optimizers.AdamW(learning_rate, weight_decay = weight_decay)
        elif opt == "RAdam"  : opt = tfa.optimizers.RectifiedAdam(learning_rate, weight_decay = weight_decay)

        ### Get hyper parameters.
        dim_input  = X_cpu.shape[1]
        dim_output = np.max(y_cpu) + 1
        num_data   = X_cpu.shape[0]

        ### Convert the label to +1/-1 format.
        X_cpu = X_cpu.astype(self.dtype)
        y_cpu = (2 * np.identity(dim_output)[y_cpu] - 1).astype(self.dtype)

        ### Create random matrix of RFF and linear SVM parameters on GPU.
        if not self.initialized:
            self.init_from_scratch(dim_input, self.dim_kernel, dim_output, self.std)

        ### Create dataset instance for training.
        data_train = tf.data.Dataset.from_tensor_slices((X_cpu, y_cpu)).shuffle(num_data).batch(self.batch_size)

        ### Loss matric.
        self.losses = tf.keras.metrics.Mean(name='train_loss')

        for epoch in range(epoch_max):

            ### Learning rate decay.
            ### TODO: Modify to be changable by user.
            if   epoch == 200: opt.learning_rate = 0.01
            elif epoch == 700: opt.learning_rate = 0.001

            ### Train one epoch.
            for Xs_batch, ys_batch in data_train:
                self.fit_batch(Xs_batch, ys_batch, opt)

            ### Exit training loop if loss value is close to machine epsilone.
            if self.losses.result().numpy() < 1.0E-10:
                break

            ### Print training log.
            if not quiet and epoch % 10 == 0:
                print(F"Epoch {epoch:>4}: Train loss = {self.losses.result().numpy():.4e}")
            self.losses.reset_states()

        return self

    ### Function for running the Tensorflow model of RFF for one batch.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimension of the input data.
    @tf.function
    def predict_proba_batch(self, X_cpu):
        self.X_gpu.assign(X_cpu)
        z = tf.matmul(self.X_gpu, self.W_gpu)
        z = tf.concat([tf.cos(z), tf.sin(z)], 1)
        return tf.matmul(z, self.A_gpu) + self.b_gpu

    ### Run prediction and return probability (features).
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimension of the input data.
    def predict_proba(self, X_cpu):

        ### Calculate size and number of batch.
        bs = self.batch_size
        bn = X_cpu.shape[0] // bs

        ### Batch number must be a multiple of batch size because the Tensorflow Graph already built.
        if X_cpu.shape[0] % bs != 0:
            raise ValueError("rfflearn.gpu.SVC: Number of input data must be a multiple of batch size")

        ### Run prediction for each batch, concatenate them and return.
        Xs = [self.predict_proba_batch(X_cpu[bs*n:bs*(n+1), :].astype(self.dtype)).numpy() for n in range(bn)]
        return np.concatenate(Xs)

    ### Run prediction and return log-probability.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimension of the input data.
    def predict_log_proba(self, X_cpu, **args):
        return np.log(self.predict_proba(X_cpu))

    ### Run prediction and return class label.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ### where N is the number of training data, K is dimension of the input data.
    def predict(self, X_cpu, **args):
        return np.argmax(self.predict_proba(X_cpu), 1)

    ### Run prediction and return the accuracy of the prediction.
    ###   - X_cpu (np.array, shape = [N, K]): training data,
    ###   - y_cpu (np.array, shape = [N, C]): training label,
    ### where N is the number of training data, K is dimension of the input data, and C is the number of classes.
    def score(self, X_cpu, y_cpu, **args):
        return np.mean(y_cpu == self.predict(X_cpu))


### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.


### Gaussian process regression with RFF.
class RFFSVC(SVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


### Gaussian process regression with ORF.
class ORFSVC(SVC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
