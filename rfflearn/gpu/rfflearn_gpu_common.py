"""
Common functions/classes for the other classes.
All classes except "seed" function is not visible from users.
"""

# Declare published functions and variables.
__all__ = ["seed", "detect_device", "Base"]

# Import standard libraries.
import importlib
import pathlib
import sys

# Import 3rd-party packages.
import numpy as np
import torch

# Append the root directory of rfflearn to Python path.
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Import "Base" class from CPU implementation. The role of the "Base" class is generation of
# random matrix. Under author's observation, generation of random matrix is not a heavy task,
# therefore GPU inplementation of the "Base" class is not necessary.
rfflearn_cpu_common = importlib.import_module("cpu.rfflearn_cpu_common")
Base = rfflearn_cpu_common.Base


def seed(random_seed: int):
    """
    Fix random seed used in this script.

    Args:
        seed (int): Random seed.
    """
    # Fix the random seed of Numpy.
    np.random.seed(random_seed)

    # Fix the random seed of PyTorch.
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def detect_device():
    """
    Detect available devices and return an appropriate device string.
    """
    # Return current GPU's device string if GPU is available.
    # This implementation only support single GPU, and the last GPU will be
    # automatically selected.
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"

    # Otherwise, return CPU device string.
    return "cpu"


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
