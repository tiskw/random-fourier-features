#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 24, 2021
##################################################### SOURCE START #####################################################

import matplotlib.pyplot as mpl

if __name__ == "__main__":

    epochs  = []
    batches = []
    accs    = []

    ### Parse output file
    for line in open("output_main_rff_batch.txt"):
        if line.startswith("Epoch = "):
            items = [s.strip() for s in line.split(",")]
            epoch = int(items[0][-1])
            batch = int(items[1][-1])
            acc   = float(items[2][-9:-4])
            epochs.append(epoch)
            batches.append(batch)
            accs.append(acc)

    ### Convert epoch and batch numbers to real value epochs
    epochs_float = [e + b / (max(batches) + 1.0) for e, b in zip(epochs, batches)]

    ### Plot a figure
    mpl.figure(0)
    mpl.title("Learning process of SVM with batch RFF")
    mpl.xlabel("Epochs [#]")
    mpl.ylabel("Accuracy for MNIST testing data")
    mpl.plot(epochs_float, accs, "-o")
    mpl.grid()
    mpl.tight_layout()
    mpl.show()

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
