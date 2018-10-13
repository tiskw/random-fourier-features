# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct 10, 2018
#################################### SOURCE START ###################################

import Utils as utils
import Timer as timer
import PyRFF as pyrff

if __name__ == "__main__":
# {{{

    ### Fix seed for random fourier feature calclation
    pyrff.seed(111)

    ### Create classifier instance
    svc = pyrff.rff.BatchSVC(dim = 1024, std = 0.1, num_epochs = 5, num_batches = 10, alpha = 0.05)

    ### Load training data
    with timer.Timer("Loading training data: "):
        Xs_train = utils.vectorise_MNIST_images("../data/MNIST_train_images.npy")
        ys_train = utils.vectorise_MNIST_labels("../data/MNIST_train_labels.npy")

    ### Load test data
    with timer.Timer("Loading test data: "):
        Xs_test = utils.vectorise_MNIST_images("../data/MNIST_test_images.npy")
        ys_test = utils.vectorise_MNIST_labels("../data/MNIST_test_labels.npy")

    ### Create matrix for principal component analysis
    with timer.Timer("Calculate PCA matrix: "):
        T = utils.mat_transform_pca(Xs_train, dim = 256)

    ### Train SVM w/ random fourier features
    with timer.Timer("Batch RFF SVM learning time: "):
        svc.fit(Xs_train.dot(T), ys_train, test = (Xs_test.dot(T), ys_test), max_iter = 1E6)

    ### Calculate score for test data
    with timer.Timer("RFF SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * svc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)

# }}}

#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
