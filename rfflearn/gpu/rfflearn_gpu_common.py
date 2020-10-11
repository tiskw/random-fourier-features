#!/usr/bin/env python3
#
# Common functions/classes for the other classes.
# All classes except "seed" function is not visible from users.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
######################################### SOURCE START ########################################


import numpy as np
import tensorflow as tf


### Import "Base" class from CPU implementation.
### The role of the "Base" class is generation of random matrix.
### Under author's observation, generation of random matrix is not a heavy task,
### therefore GPU inplementation of the "Base" class is not necessary.
from ..cpu.rfflearn_cpu_common import Base


### Fix random seed used in this script.
def seed(seed):

    ### Need to fix the random seed of Numpy and Tensorflow.
    np.random.seed(seed)
    tf.random.set_seed(seed)


######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
