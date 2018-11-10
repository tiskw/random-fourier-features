# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Nov 10, 2018
#################################### SOURCE START ###################################

import ctypes
import numpy           as np
import numpy.ctypeslib as npct

array_1d_double = npct.ndpointer(dtype = np.float32, ndim = 1, flags = "CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype = np.float32, ndim = 2, flags = "CONTIGUOUS")

lib = npct.load_library("libFastInferenceRFF", ".")

lib.rff_fast_inference.restype  = ctypes.c_float
lib.rff_fast_inference.argtypes = [array_1d_double, array_2d_double, array_1d_double, ctypes.c_int, ctypes.c_int]

def rff_fast_inference(x, W, a):
    return lib.rff_fast_inference(x, W, a, W.shape[0], W.shape[1])

#################################### SOURCE FINISH ##################################
# Ganerated by grasp version 0.0
