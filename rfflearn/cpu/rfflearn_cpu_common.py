#!/usr/bin/env python3
#
# Python module of regression and support vector machine using random fourier features.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 05, 2020
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


### This function returns a function which generate RFF/ORF matrix.
### The usage of the returned value of this function are:
###     f(dim_input:int) -> np.array with shape (dim_input, dim_kernel)
def get_matrix_generator(rand_mat_type, std, dim_kernel):

    if   rand_mat_type == "rff": return functools.partial(get_rff_matrix, std = std, dim_out = dim_kernel)
    elif rand_mat_type == "orf": return functools.partial(get_orf_matrix, std = std, dim_out = dim_kernel)
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
    def conv(self, X, index = None):
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
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
