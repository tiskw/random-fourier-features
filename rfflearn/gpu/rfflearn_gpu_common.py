"""
Common functions/classes for the other classes.
All classes except "seed" function is not visible from users.
"""

import numpy as np
import torch

# Import "Base" class from CPU implementation.
# The role of the "Base" class is generation of random matrix.
# Under author's observation, generation of random matrix is not a heavy task,
# therefore GPU inplementation of the "Base" class is not necessary.
from ..cpu.rfflearn_cpu_common import Base


def seed(seed):
    """
    Fix random seed used in this script.

    Args:
        seed (int): Random seed.
    """
    # Need to fix the random seed of Numpy and PyTorch.
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device():
    """
    Detect available devices and return an appropriate device string.
    """
    # Return current GPU's device string if GPU is available.
    # This implementation only support single GPU, and the last GPU will be
    # automatically selected.
    if torch.cuda.is_available():
        return "cuda:%d" % torch.cuda.current_device()

    # Otherwise, return CPU device string.
    else: return "cpu"


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
