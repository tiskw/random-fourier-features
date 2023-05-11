"""
Python module of support vector classification with random matrix for CPU.
"""

import numpy as np
import torch

from .rfflearn_gpu_common import Base


class SVC(Base):
    """
    This class provides the RFF based SVC classification using GPU.

    Notes:
        Number of data (= X_cpu.shape[0]) must be a multiple of batch size.
    """
    def __init__(self, rand_type, svc=None, M_pre=None, dim_kernel=128, std_kernel=0.1,
                 W=None, b=None, batch_size=200, dtype="float32"):
        """
        Create parameters on CPU, then move the parameters to GPU.
        There are two ways to initialize these parameters:
          (1) from scratch: generate parameters from scratch,
          (2) from RFFSVC: copy parameters from RFFSVC (CPU) class instance.
        The member variable 'self.initialized' indicate that the parameters are well initialized
        or not. If the parameters are initialized by one of the ways other than (1),
        'self.initialized' is set to True. And if 'self.initialized' is still False when just
        before the training/inference, then the parameters are initialized by the way (1).

        Args:
            rand_type  (str)                : Type of random matrix ("rff", "orf", "qrf", etc).
            svc        (rfflearn.cpu.RFFSVC): RFFSVC instance for initialization.
            M_pre      (np.ndarray)         : Matrix to be merged to the random matrix `W`.
            dim_kernel (int)                : Dimension of the random matrix.
            std_kernel (float)              : Standard deviation of the random matrix.
            W          (np.ndarray)         : Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray)         : Random bias for the input `X`. If None then generated automatically.
            batch_size (int)                : Size of one batch.
            dtype      (str)                : Data type used in the training and inference.
        """
        # Save important variables.
        self.dim_kernel  = dim_kernel
        self.std         = std_kernel
        self.batch_size  = batch_size
        self.dtype       = dtype
        self.initialized = False
        self.W           = W
        self.b           = b

        # Automatically detect device.
        # This module assumes that GPU is available, but works if not available.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Inisialize variables.
        if svc: self.init_from_RFFSVC_cpu(svc, M_pre)
        else  : super().__init__(rand_type, dim_kernel, std_kernel, W, b)

    def init_from_scratch(self, dim_input, dim_kernel, dim_output, std):
        """
        Constractor: initialize parameters from scratch.

        Args:
            dim_input  (int)  : Input dimension of SVC.
            dim_kernel (int)  : Dimension of the random matrix.
            dim_output (int)  : Output dimension of SVC.
            std        (float): Standard deviation of the random matrix.
        """
        # Generate random matrix.
        self.set_weight(dim_input)

        # Create parameters on CPU.
        #   - W: Random matrix of Random Fourier Features. (shape = [dim_input,  dim_kernel]).
        #   - A: Coefficients of Linear SVC.               (shape = [dim_kernel, dim_output]).
        #   - b: Intercepts of Linear SVC.                 (shape = [1,          dim_output]).
        self.W_cpu = self.W
        self.b_cpu = self.b
        self.A_cpu = 0.01 * np.random.randn(dim_kernel, dim_output)
        self.k_cpu = 0.01 * np.random.randn(1,          dim_output)

        # Create GPU variables and build GPU model.
        self.build()

        # Mark as initialized.
        self.initialized = True

    def init_from_RFFSVC_cpu(self, svc_cpu, M_pre):
        """
        Copy parameters from the given rffsvc and create instance.

        Args:
            svc_svc (rfflearn.cpu.RFFSVC): RFFSVC instance for initialization.
            M_pre   (np.ndarray)         : Matrix to be merged to the random matrix `W`.
        """
        # Only RFFSVC support GPU inference.
        if not (hasattr(svc_cpu, "W") and hasattr(svc_cpu, "svm")):
            raise TypeError("rfflearn.gpu.SVC: Only rfflearn.cpu.SVC supported.")

        # TODO: One-versus-one classifier is not supported now.
        if svc_cpu.svm.get_params()["estimator__multi_class"] != "ovr":
            raise TypeError("rfflearn.gpu.SVC: Sorry, current implementation support only One-versus-the-rest classifier.")

        # If OneVsRestClassifier does not support `coef_` and `intercept_`, manualy create them.
        if not (hasattr(svc_cpu.svm, "coef_") and hasattr(svc_cpu.svm, "intercept_")):
            svc_cpu.svm.coef_      = np.array([estimator.coef_.flatten() for estimator in svc_cpu.svm.estimators_])
            svc_cpu.svm.intercept_ = np.array([estimator.intercept_      for estimator in svc_cpu.svm.estimators_])

        # Copy parameters from rffsvc on CPU.
        #   - W: Random matrix of Random Fourier Features.
        #        If PCA applied, combine it to the random matrix for high throughput.
        #   - A: Coefficients of Linear SVC.
        #   - b: Intercepts of Linear SVC.
        self.W_cpu = M_pre.dot(svc_cpu.W) if M_pre is not None else svc_cpu.W
        self.b_cpu = svc_cpu.b
        self.A_cpu = svc_cpu.svm.coef_.T
        self.k_cpu = svc_cpu.svm.intercept_.T

        # Create GPU variables and build GPU model.
        self.build()

        # Mark as initialized.
        self.initialized = True

    def build(self):
        """
        Create GPU variables if all CPU variables are ready.
        """
        # Run build procedure if and only if all variables is available.
        if all(v is not None for v in [self.W_cpu, self.b_cpu, self.A_cpu, self.k_cpu]):

            # Create GPU variables.
            self.W_gpu = torch.tensor(self.W_cpu, dtype = torch.float32, device = self.device, requires_grad = False)
            self.b_gpu = torch.tensor(self.b_cpu, dtype = torch.float32, device = self.device, requires_grad = False)
            self.A_gpu = torch.tensor(self.A_cpu, dtype = torch.float32, device = self.device, requires_grad = True)
            self.k_gpu = torch.tensor(self.k_cpu, dtype = torch.float32, device = self.device, requires_grad = True)

            self.params = [self.A_gpu, self.k_gpu]

    def fit_batch(self, X_gpu, y_gpu, opt):
        """
        Train the model for one batch.

        Args:
            X_gpu (np.ndarray)           : Input matrix of shape (n_samples, n_features).
            y_gpu (np.ndarray)           : Output matrix of shape (n_samples,).
            opt   (torch.optim.Optimizer): Optimizer for gradient descent.
        """
        # Calclate loss function under the GradientTape.
        z = torch.cos(torch.matmul(X_gpu, self.W_gpu) + self.b_gpu)
        p = torch.matmul(z, self.A_gpu) + self.k_gpu
        v = torch.mean(torch.sum(torch.max(torch.zeros_like(y_gpu), 1 - y_gpu * p), dim = 1))

        # Derive gradient for all variables.
        v.backward()

        with torch.no_grad():
            opt.step()

        return float(v.cpu())

    def fit(self, X_cpu, y_cpu, epoch_max=300, opt="sgd", learning_rate=1.0E-2, weight_decay=10.0, quiet=False):
        """
        Training of SVC.

        Args:
            X_cpu         (np.ndarray): Input matrix of shape (n_samples, n_features).
            y_cpu         (np.ndarray): Output matrix of shape (n_samples, n_features).
            epoch_max     (int)       : Maximum number of training epochs.
            opt           (str)       : Optimizer name.
            learning_rate (float)     : Lerning rate of the optimizer.
            weight_decay  (float)     : Weight decay rate.
            quiet         (bool)      : Suppress messages if True.

        Returns:
            (rfflearn.gpu.SVC): Myself.
        """
        # Get hyper parameters.
        dim_input  = X_cpu.shape[1]
        dim_output = np.max(y_cpu) + 1

        # Convert the label to +1/-1 format.
        X_cpu = X_cpu.astype(self.dtype)
        y_cpu = (2 * np.identity(dim_output)[y_cpu] - 1).astype(self.dtype)

        # Create random matrix of RFF and linear SVM parameters on GPU.
        if not self.initialized:
            self.init_from_scratch(dim_input, self.dim_kernel, dim_output, self.std)

        # Get optimizer.
        if   opt == "sgd"    : opt = torch.optim.SGD(self.params, learning_rate, weight_decay = weight_decay)
        elif opt == "rmsprop": opt = torch.optim.RMSprop(self.params, learning_rate)
        elif opt == "adam"   : opt = torch.optim.Adam(self.params, learning_rate)
        elif opt == "adamw"  : opt = torch.optim.AdamW(self.params, learning_rate, weight_decay = weight_decay, amsgrad = True)

        # Create dataset instance for training.
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_cpu), torch.tensor(y_cpu))
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

        # Variable to store loss values in one epoch.
        losses = []

        for epoch in range(epoch_max):

            # Train one epoch.
            for step, (Xs_batch, ys_batch) in enumerate(train_loader):
                loss = self.fit_batch(Xs_batch.to(self.device), ys_batch.to(self.device), opt)
                losses.append(loss)

            # Print training log.
            if not quiet and epoch % 10 == 0:
                print(f"Epoch {epoch:>4}: Train loss = {np.mean(losses):.4e}")

            # Clear loss values.            
            losses.clear()

        return self

    def predict_proba_batch(self, X_cpu):
        """
        Function for running the PyTorch model of RFF for one batch.

        Args:
            X_cpu (np.ndarray): Input matrix of shape (n_samples, n_features).

        Returns:
            (np.ndarray): Probability of prediction.
        """
        X = torch.tensor(X_cpu, device = self.device)
        z = torch.cos(torch.matmul(X, self.W_gpu) + self.b_gpu)
        return (torch.matmul(z, self.A_gpu) + self.k_gpu).detach().cpu().numpy()

    def predict_proba(self, X_cpu):
        """
        Function for running the PyTorch model of RFF for one batch.

        Args:
            X_cpu (np.ndarray): Input matrix of shape (n_samples, n_features).

        Returns:
            (np.ndarray): Probability of prediction.
        """
        # Calculate size and number of batch.
        bs = self.batch_size
        bn = X_cpu.shape[0] // bs

        # Batch number must be a multiple of batch size.
        if X_cpu.shape[0] % bs != 0:
            raise ValueError("rfflearn.gpu.SVC: Number of input data must be a multiple of batch size")

        # Run prediction for each batch, concatenate them and return.
        Xs = [self.predict_proba_batch(X_cpu[bs*n:bs*(n+1), :].astype(self.dtype)) for n in range(bn)]
        return np.concatenate(Xs)

    def predict_log_proba(self, X_cpu, **args):
        """
        Run prediction and return log-probability.

        Args:
            X_cpu (np.array): Input matrix of shape (n_samples, n_features).
            args  (dict)    : Extra arguments. This will be ignored.

        Returns:
            (np.ndarray): Log probability of prediction.
        """
        return np.log(self.predict_proba(X_cpu))

    def predict(self, X_cpu, **args):
        """
        Run prediction and return class label.

        Args:
            X_cpu (np.array): Input matrix of shape (n_samples, n_features).
            args  (dict)    : Extra arguments. This will be ignored.

        Returns:
            (np.ndarray): Predicted class labels.
        """
        return np.argmax(self.predict_proba(X_cpu), 1)

    def score(self, X_cpu, y_cpu, **args):
        """
        Run prediction and return the accuracy of the prediction.

        Args:
            X_cpu (np.array): Input matrix of shape (n_samples, n_features).
            y_cpu (np.array): Output matrix of shape (n_samples, n_features).

        Returns:
            (np.ndarray): Accuracy of the prediction.
        """
        return np.mean(y_cpu == self.predict(X_cpu))


# The above functions/classes are not visible from users of this library, becasue the usage of
# the function is a bit complicated. The following classes are simplified version of the above
# classes. The following classes are visible from users.


class RFFSVC(SVC):
    """
    Gaussian process regression with RFF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)


class ORFSVC(SVC):
    """
    Gaussian process regression with ORF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)


class QRFSVC(SVC):
    """
    Gaussian process regression with quasi-RRF.
    """
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
