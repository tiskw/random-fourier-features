# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct  5, 2018
#################################### SOURCE START ###################################

import numpy                 as np
import RandomFourierFeatures as rff
import matplotlib.pyplot     as mpl

if __name__ == "__main__":
# {{{

    rff.seed(333)

    Xs  = np.load("sample_data/xs.npy")
    ys  = np.load("sample_data/ys.npy")
    svc = rff.SVC(dim_output = 10)
    svc.fit(Xs, ys)

    positives = []
    negatives = []
    for x in np.linspace(-2, 2, 41):
        for y in np.linspace(-2, 2, 41):
            v = svc.predict([[x, y]])[0]
            if v > 0: positives.append((x, y))
            else    : negatives.append((x, y))

    print("Score = %.2f [%%]" % (100 * svc.score(Xs, ys)))

    positives = np.array(positives)
    negatives = np.array(negatives)

    mpl.figure(0)
    mpl.title("")
    mpl.plot(Xs[ys[:, 0] > 0, 0], Xs[ys[:, 0] > 0, 1], ".")
    mpl.plot(Xs[ys[:, 0] < 0, 0], Xs[ys[:, 0] < 0, 1], ".")
    mpl.xlim(-2.1, 2.1)
    mpl.ylim(-2.1, 2.1)
    mpl.grid()

    mpl.figure(1)
    mpl.plot(positives[:, 0], positives[:, 1], ".")
    mpl.plot(negatives[:, 0], negatives[:, 1], ".")
    mpl.xlim(-2.1, 2.1)
    mpl.ylim(-2.1, 2.1)
    mpl.grid()

    mpl.show()

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
