"""
Common functions/classes for the other classes.
All classes except 'seed' function is not visible from users.
"""

# Declare published functions and variables.
__all__ = ["seed", "get_matrix_generator", "Base"]

# Import standard libraries.
import functools
import typing

# Import 3rd-party packages.
import numpy as np
import scipy.stats


####################################################################################################
# Random matrix generators
####################################################################################################


def get_rff_matrix(dim_in: int, dim_out: int, std: float) -> np.ndarray:
    """
    Generates random matrix of random Fourier features.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    return std * np.random.randn(dim_in, dim_out)


def get_orf_matrix(dim_in: int, dim_out: int, std: float) -> np.ndarray:
    """
    Generates random matrix of orthogonal random features.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    # Initialize matrix W.
    W = None

    for _ in range(dim_out // dim_in + 1):
        s = scipy.stats.chi.rvs(df=dim_in, size=(dim_in, ))
        Q = np.linalg.qr(np.random.randn(dim_in, dim_in))[0]
        V = std * np.dot(np.diag(s), Q)
        W = V if W is None else np.concatenate([W, V], axis=1)

    # Trim unnecessary part.
    return W[:dim_in, :dim_out]


def get_qrf_matrix(dim_in: int, dim_out: int, std: float) -> np.ndarray:
    """
    Generates random matrix for quasi-random Fourier features.

    Args:
        dim_in  (int)  : Input dimension of the quasi-random matrix.
        dim_out (int)  : Output dimension of the quasi-random matrix.
        std     (float): Standard deviation of the quasi-random matrix.

    Returns:
        (np.ndarray): Quasi-random matrix with shape (dim_out, dim_in).
    """
    # Parameters for quasi random numbers generation.
    QUASI_MC_SKIP = 1000
    QUASI_MC_LEAP = 100

    # Implementation of Box-Muller method for converting
    # uniform random sequence to normal random sequence.
    def box_muller_method(xs, ys):
        zs1 = np.sqrt(-2 * np.log(xs)) * np.cos(2 * np.pi * ys)
        zs2 = np.sqrt(-2 * np.log(xs)) * np.sin(2 * np.pi * ys)
        return np.array([zs1, zs2])

    # Generate sobol sequence engine and throw away the first several values.
    sampler = scipy.stats.qmc.Sobol(dim_in, scramble=True, seed=np.random.randint(0, np.iinfo(np.int32).max))
    sampler.fast_forward(QUASI_MC_SKIP)

    # Generate uniform random matrix.
    W = np.zeros((dim_in, dim_out))
    for index in range(dim_out):
        W[:, index] = sampler.random(1)
        sampler.fast_forward(QUASI_MC_LEAP)

    # Convert the uniform random matrix to normal random matrix.
    for index in range(0, dim_out, 2):
        W[:, index:index+2]  = box_muller_method(W[:, index], W[:, index+1]).T

    return std * W


def get_matrix_generator(rand_type: str, std: float, dim_kernel: int) -> typing.Callable:
    """
    This function returns a function which generate RFF/ORF matrix.
    The usage of the returned value of this function are:
        f(dim_input: int) -> np.array with shape (dim_input, dim_kernel)

    Args:
        rand_type  (str)  : Type of random matrix ("rff", "orf", "qrf", etc).
        std        (float): Standard deviation of the random matrix.
        dim_kernel (int)  : Dimension of the random matrix.

    Returns:
        (Callable): A function that returns a random matrix.
    """
    # Carete a dictionary of random functions.
    rand_funcs = {
        "rff": get_rff_matrix,
        "orf": get_orf_matrix,
        "qrf": get_qrf_matrix,
    }

    # Raise an error if the specified random type not found.
    if rand_type not in rand_funcs:
        raise RuntimeError(f"'rand_type' must be either of {rand_funcs.keys()}.")

    return functools.partial(rand_funcs[rand_type], std=std, dim_out=dim_kernel)


####################################################################################################
# Base class of RFF/ORF/QRF based ML algorithms
####################################################################################################


class Base:
    """
    Base class of the following RFF/ORF/QRF related classes.
    """
    def __init__(self, rand_type: str, dim_kernel: int, std_kernel: float,
                 W: np.ndarray, b: np.ndarray):
        """
        Constractor of the Base class.
        Create random matrix generator and random matrix instance.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W          (np.ndarray): Random matrix for input `X` (generated automatically if None).
            b          (np.ndarray): Random bias for input `X` (generated automatically if None).

        Notes:
            If `W` is None then the appropriate matrix will be set just before the training.
        """
        self.kdim = dim_kernel
        self.kstd = std_kernel
        self.kgen = get_matrix_generator(rand_type, std_kernel, dim_kernel)
        self.W    = W
        self.b    = b

    def conv(self, X: np.ndarray, index: int = None) -> np.ndarray:
        """
        Applies random matrix to the given input vectors `X` and create feature vectors.

        Args:
            X     (np.ndarray): Input matrix with shape (n_samples, n_features).
            index (int)       : Index of the random matrix. This value should be specified only
                                when multiple random matrices are used.

        Returns:
            (np.ndarray): Converted vector.

        Notes:
            This function can manipulate multiple random matrix. If argument 'index' is given,
            then use self.W[index] as a random matrix, otherwise use self.W itself.
            Also, computation of `ts` is equivarent with ts = X @ W, however, for reducing memory 
            consumption, split X to smaller matrices and concatenate after multiplication wit W.
        """
        W = self.W if index is None else self.W[index]
        b = self.b if index is None else self.b[index]
        return np.cos(X @ W + b)

    def set_weight(self, dim_in: int):
        """
        Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).

        Args:
            dim_in (int): Input dimension of the random matrix.

        Notes:
            This function can manipulate multiple random matrix. If argument 'dim_in' is
            a list/tuple of integers, then generate multiple random matrixes.
        """
        def get_W():
            """
            Generate matrix W.
            """
            if self.W is not None:
                return self.W

            if hasattr(dim_in, "__iter__"):
                return tuple(self.kgen(d) for d in dim_in)

            return self.kgen(dim_in)

        def get_b():
            """
            Generate vector b.
            """
            if self.b is not None:
                return self.b

            if hasattr(dim_in, "__iter__"):
                return tuple(np.random.uniform(0, 2*np.pi, size=self.W.shape[1]) for _ in dim_in)

            return np.random.uniform(0, 2*np.pi, size=self.kdim)

        self.W = get_W()
        self.b = get_b()


####################################################################################################
# Utility functions
####################################################################################################


def seed(random_seed: int):
    """
    Fix random seed used in this script.

    Args:
        random_seed (int): Random seed.
    """
    # Now it is enough to fix the random seed of Numpy.
    np.random.seed(random_seed)


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
