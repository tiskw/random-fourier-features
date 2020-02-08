#!/usr/bin/env python3
#
# This Python script generate a figure which shows a trade off relationship
# between inference time and test accuracy on MNIST dataset.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Nov 16, 2019
#################################### SOURCE START ###################################

import numpy             as np
import matplotlib.pyplot as mpl

### DIM_RFF, Learning time [sec], Inference time [us], Test accuracy on MNIST [%]
SCORES_RFF = np.array([[  64,  36.795353,   9.378099, 89.39],
                       [ 128,  54.435829,  13.771224, 93.64],
                       [ 256,  83.267031,  22.314167, 95.37],
                       [ 384, 108.231835,  30.490088, 95.99],
                       [ 512, 126.593045,  38.989401, 96.50],
                       [ 640, 138.779719,  47.662687, 96.61],
                       [ 768, 145.924327,  55.336690, 96.69],
                       [ 896, 159.675971,  64.729142, 96.93],
                       [1024, 226.717515,  96.054697, 97.48],
                       [1152, 202.715609, 125.541520, 97.19],
                       [1280, 224.689160, 136.949730, 97.29],
                       [1408, 247.876134, 156.303930, 97.35],
                       [1536, 271.724527, 198.049259, 97.24]])

fig = mpl.figure()
axs = fig.add_subplot(111)

axs.grid()
axs.set_title("Performance of SVM with RFF")
axs.set_xlabel("Inference time [us]")
axs.set_ylabel("Test accuracy on MNIST [%]")
axs.set_xlim(0, 225)
axs.set_ylim(88, 98)

for dim, _, x, y in SCORES_RFF:
    axs.plot(x, y, "o", label = "d=%d" % dim)

axs.legend()
fig.savefig("figure_rff_svc_for_mnist.png")

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
