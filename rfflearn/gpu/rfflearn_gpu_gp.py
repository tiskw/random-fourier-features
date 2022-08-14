#!/usr/bin/env python3
#
# Python module of Gaussian process with random matrix for GPU.
######################################### SOURCE START ########################################

import numpy as np
import sklearn.metrics
import torch

from .rfflearn_gpu_common import Base, detect_device

### This class provides the RFF based Gaussian Process classification using GPU.
### This class has the following member functions. See the comments and code below for details.
###   - fit(self, X_cpu, y_cpu)  : run training using training data X_cpu and y_cpu
###   - predict(self, X_cpu)     : run inference (this function also be able to return variance)
###   - score(self, X_cpu, y_cpu): run inference and return the overall accuracy
class GPR(Base):

    ### Constractor: Store necessary parameters (on CPU).
    def __init__(self, rand_mat_type, dim_kernel = 256, std_kernel = 0.1, std_error = 1.0, scale = 1.0, W = None, b = None, a = None, S = None):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W, b)
        self.s_e = std_error
        self.dev = detect_device()
        self.a   = a
        self.S   = S

    ### Training of Gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): training data
    ###   - y_cpu (np.array, shape = [N, C]): training label
    ### where N is the number of training data, K is dimension of the input data, and C is dimention of output data.
    def fit(self, X_cpu, y_cpu):

        ### Generate random matrix.
        self.set_weight(X_cpu.shape[1])

        ### Generate random matrix W and identity matrix I on CPU.
        I_cpu = np.eye(self.dim)

        ### Derive posterior distribution 1/3 (on CPU).
        s_cpu = self.s_e**2
        F_cpu = self.conv(X_cpu).T
        P_cpu = F_cpu @ F_cpu.T

        ### Derive posterior distribution 2/3 (on GPU).
        Q_gpu = torch.tensor(P_cpu,                 device = self.dev, dtype = torch.float64)
        R_gpu = torch.tensor(P_cpu + s_cpu * I_cpu, device = self.dev, dtype = torch.float64)
        M_cpu = I_cpu - torch.linalg.solve(R_gpu, Q_gpu).cpu().numpy()

        ### Derive posterior distribution 3/3 (on CPU).
        self.a = (y_cpu.T @ F_cpu.T) @ M_cpu / s_cpu
        self.S = I_cpu - P_cpu @ M_cpu / s_cpu

        ### Clean GPU memory.
        del Q_gpu, R_gpu
        if "cuda" in self.dev: torch.cuda.empty_cache()

        return self

    ### Inference of Gaussian process using GPU.
    ###   - X_cpu (np.array, shape = [N, K]): inference data
    ###   - var   (boolean, scalar)         : returns variance vector if true
    ###   - cov   (boolean, scalar)         : returns covariance matrix if true
    ### where N is the number of training data and K is dimension of the input data.
    def predict(self, X_cpu, return_std = False, return_cov = False):

        ### Move matrix to GPU.
        X = torch.tensor(X_cpu,  device = self.dev, dtype = torch.float64)
        W = torch.tensor(self.W, device = self.dev, dtype = torch.float64)
        b = torch.tensor(self.b, device = self.dev, dtype = torch.float64)
        a = torch.tensor(self.a, device = self.dev, dtype = torch.float64)
        S = torch.tensor(self.S, device = self.dev, dtype = torch.float64)

        ### Calculate mean of the prediction distribution.
        F = torch.cos(torch.matmul(X, W) + b).t()
        p = torch.matmul(a, F).t()

        ### Move prediction value to CPU.
        ### If shape of y_cpu is (*, 1), then reshape to (*, ).
        y_cpu = p.cpu().numpy()
        y_cpu = np.squeeze(y_cpu, axis = 1) if len(y_cpu.shape) > 1 and y_cpu.shape[1] == 1 else y_cpu

        ### Calculate covariance matrix and variance vector.
        if return_std or return_cov:
            V_cpu = torch.matmul(torch.matmul(F.t(), S), F).cpu().numpy()
            s_cpu = np.sqrt(np.diag(V_cpu))

        ### Clean GPU memory.
        del X, W, a, S, F, p
        if "cuda" in self.dev: torch.cuda.empty_cache()

        if return_std and return_cov: return [y_cpu, s_cpu, V_cpu]
        elif return_std             : return [y_cpu, s_cpu]
        elif return_cov             : return [y_cpu, V_cpu]
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
    ###   - X_cpu      (np.array, shape = [N, K]): inference data
    ###   - return_std (boolean, scalar)         : return standard deviation vector if true
    ###   - return_cov (boolean, scalar)         : return covariance matrix if true
    ### where N is the number of training data and K is dimension of the input data.
    def predict(self, X_cpu, return_std = False, return_cov = False):

        ### Run GPC prediction. Note that the returned value is one-hot vector.
        res = super().predict(X_cpu, return_std, return_cov)

        ### Convert one-hot vector to class index.
        if return_std or return_cov: res[0] = np.argmax(res[0], axis = 1)
        else                       : res    = np.argmax(res,    axis = 1)

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

### Gaussian process regression with Quasi-RFF.
class QRFGPR(GPR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

### Gaussian process classifier with RFF.
class RFFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Gaussian process classifier with ORF.
class ORFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

### Gaussian process classifier with Quasi-RRF.
class QRFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

######################################### SOURCE FINISH #######################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
