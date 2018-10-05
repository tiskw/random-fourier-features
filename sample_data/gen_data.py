#!/usr/bin/env python3
#
# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct  5, 2018
#################################### SOURCE START ###################################

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as mpl

def randn2d(center, stdev, num):
# {{{

    cs = np.tile(center, (num, 1))
    vs = stdev * np.random.randn(num, 2)
    return (cs + vs)

# }}}

if __name__ == "__main__":
# {{{

    if os.path.exists("xs.npy") or os.path.exists("ys.npy"):
        exit("Error: Output file (xs.npy or ys.npy) already exists. Remove them before call me.")

    if len(sys.argv) > 1: seed = int(sys.argv[1])
    else                : seed = 333

    random.seed(seed)
    np.random.seed(seed)

    num   = 1500
    stdev = 0.30

    xs1 = np.bmat([[randn2d((1, 1), stdev, num)], [randn2d((-1, -1), stdev, num)]])
    ys1 = np.ones((2 * num, 1))

    xs2 = np.bmat([[randn2d((1, -1), stdev, num)], [randn2d((-1, 1), stdev, num)]])
    ys2 = - np.ones((2 * num, 1))

    xs = np.bmat([[xs1], [xs2]])
    ys = np.bmat([[ys1], [ys2]])

    np.save("xs.npy", xs)
    np.save("ys.npy", ys)

    mpl.plot(xs1[:, 0], xs1[:, 1], ".")
    mpl.plot(xs2[:, 0], xs2[:, 1], ".")
    mpl.grid()
    mpl.show()

# }}}

#################################### SOURCE FINISH ##################################
# Ganerated by grasp version 0.0
