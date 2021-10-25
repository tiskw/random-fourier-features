#!/usr/bin/env python3
#
# Common functions/classes for the other classes.
# All classes except 'seed' function is not visible from users.
##################################################### SOURCE START #####################################################

import functools
import numpy as np
import scipy.stats

### Fix random seed used in this script.
def seed(seed):

    ### Now it is enough to fix the random seed of Numpy.
    np.random.seed(seed)

### Generate random matrix for random Fourier features.
def get_rff_matrix(dim_in, dim_out, std):

    return  std * np.random.randn(dim_in, dim_out)

### Generate random matrix for orthogonal random features.
def get_orf_matrix(dim_in, dim_out, std):

    ### Initialize matrix W.
    W = None

    for _ in range(dim_out // dim_in + 1):
        s = scipy.stats.chi.rvs(df = dim_in, size = (dim_in, ))
        Q = np.linalg.qr(np.random.randn(dim_in, dim_in))[0]
        V = std * np.dot(np.diag(s), Q)
        W = V if W is None else np.concatenate([W, V], axis = 1)

    ### Trim unnecessary part.
    return W[:dim_in, :dim_out]

### Generate random matrix for random Fourier features.
def get_qrf_matrix(dim_in, dim_out, std):

    ### Parameters for quasi random numbers generation.
    QUASI_MC_SKIP = 1000
    QUASI_MC_LEAP = 100

    ### Implementation of Box-Muller method for converting
    ### uniform random sequence to normal random sequence.
    def box_muller_method(xs, ys):
        zs1 = np.sqrt(-2 * np.log(xs)) * np.cos(2 * np.pi * ys)
        zs2 = np.sqrt(-2 * np.log(xs)) * np.sin(2 * np.pi * ys)
        return np.array([zs1, zs2])

    import torch

    ### Generate sobol sequence engine and throw away the first several values.
    sobol = torch.quasirandom.SobolEngine(dim_in, scramble = True)
    sobol.fast_forward(QUASI_MC_SKIP)

    ### Generate uniform random matrix.
    W = np.zeros((dim_in, dim_out))
    for index in range(dim_out):
        W[:, index] = sobol.draw(1).numpy()
        sobol.fast_forward(QUASI_MC_LEAP)

    ### Convert the uniform random matrix to normal random matrix.
    for index in range(0, dim_out, 2):
        W[:, index:index+2]  = box_muller_method(W[:, index], W[:, index+1]).T

    return std * W

### This function returns a function which generate RFF/ORF matrix.
### The usage of the returned value of this function are:
###     f(dim_input:int) -> np.array with shape (dim_input, dim_kernel)
def get_matrix_generator(rand_mat_type, std, dim_kernel):

    if   rand_mat_type == "rff": return functools.partial(get_rff_matrix, std = std, dim_out = dim_kernel)
    elif rand_mat_type == "orf": return functools.partial(get_orf_matrix, std = std, dim_out = dim_kernel)
    elif rand_mat_type == "qrf": return functools.partial(get_qrf_matrix, std = std, dim_out = dim_kernel)
    else                       : raise RuntimeError("matrix_generator: 'rand_mat_type' must be 'rff' or 'orf'.")

### Base class of the following RFF/ORF related classes.
class Base:

    ### Constractor. Create random matrix generator and random matrix instance.
    ### NOTE: If 'W' is None then the appropriate matrix will be set just before the training.
    def __init__(self, rand_mat_type, dim_kernel, std_kernel, W):
        self.dim = dim_kernel
        self.s_k = std_kernel
        self.mat = get_matrix_generator(rand_mat_type, std_kernel, dim_kernel)
        self.W   = W

    ### Apply random matrix to the given input vectors 'X' and create feature vectors.
    ### NOTE: This function can manipulate multiple random matrix. If argument 'index'
    ###       is given, then use self.W[index] as a random matrix, otherwise use self.W itself.
    ### NOTE: Computation of `ts` is equivarent with ts = X @ W, however, for reducing
    ###       memory consumption, split X to smaller matrices and concatenate after multiplication wit W.
    def conv(self, X, index = None, chunk_size = 1024):
        W  = self.W if index is None else self.W[index]
        ts = X @ W
        return np.bmat([np.cos(ts), np.sin(ts)])

    ### Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).
    ### NOTE: This function can manipulate multiple random matrix. If argument 'dim_in'
    ###       is a list/tuple of integers, then generate multiple random matrixes.
    def set_weight(self, dim_in):
        if   self.W is not None         : pass
        elif hasattr(dim_in, "__iter__"): self.W = tuple([self.mat(d) for d in dim_in])
        else                            : self.W = self.mat(dim_in)

##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
