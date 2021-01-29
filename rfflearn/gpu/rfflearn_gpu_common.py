#!/usr/bin/env python3
#
# Common functions/classes for the other classes.
# All classes except "seed" function is not visible from users.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 29, 2021
######################################### SOURCE START ########################################

import numpy as np
import torch

### Import "Base" class from CPU implementation.
### The role of the "Base" class is generation of random matrix.
### Under author's observation, generation of random matrix is not a heavy task,
### therefore GPU inplementation of the "Base" class is not necessary.
from ..cpu.rfflearn_cpu_common import Base

### Fix random seed used in this script.
def seed(seed):

    ### Need to fix the random seed of Numpy and PyTorch.
    np.random.seed(seed)
    torch.manual_seed(seed)

### Detect available devices and return an appropriate device string.
def detect_device():

    ### Return current GPU's device string if GPU is available.
    if torch.cuda.is_available():
        return "cuda:%d" % torch.cuda.current_device()

    ### Otherwise, return CPU device string.
    else: return "cpu"

######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
