#!/usr/bin/env python3
#
# Python script for downloading the MNIST dataset and convert them to
# .npy format which is easy to read, train and test for a Python script.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Nov 16, 2019
#################################### SOURCE START ###################################

import os
import subprocess
import gzip
import numpy as np

BYTE_ORDER = "big"


### Run 'wget' command if the target file does not exist.
def download_MNIST(filepath):

    if os.path.exists(filepath):
        print("File '%s' already exists. Skip it." % filepath)
        return

    url = "http://yann.lecun.com/exdb/mnist/%s" % filepath
    cmd = "wget %s ." % url
    subprocess.run(cmd, shell = True)


### Convert the MNIST image data to .npy format.
def convert_image_data(filepath_input, filepath_output):

    ### Skip the following procedure if the output file exists.
    if os.path.exists(filepath_output):
        print("File '%s' already exists. Skip it." % filepath_output)
        return

    print("Convert: %s -> %s" % (filepath_input, filepath_output))

    ### Unzip downloaded file.
    with gzip.open(filepath_input, "rb") as ifp:
        data = ifp.read()

    ### Parse header information.
    identifier = int.from_bytes(data[ 0: 4], BYTE_ORDER)
    num_images = int.from_bytes(data[ 4: 8], BYTE_ORDER)
    image_rows = int.from_bytes(data[ 8:12], BYTE_ORDER)
    image_cols = int.from_bytes(data[12:16], BYTE_ORDER)
    image_data = data[16:]

    if identifier != 2051:
        print("Input file '%s' does not seems to be MNIST image file." % filepath)

    images = np.zeros((num_images, image_rows, image_cols))

    for n in range(num_images):
        index_b = image_rows * image_cols * n
        index_e = image_rows * image_cols * (n + 1)
        image   = [int(b) for b in image_data[index_b:index_e]]
        images[n, :, :] = np.array(image).reshape((image_rows, image_cols))

    np.save(filepath_output, images)


### Convert the MNIST label data to .npy format.
def convert_label_data(filepath_input, filepath_output):

    ### Skip the following procedure if the output file exists.
    if os.path.exists(filepath_output):
        print("File '%s' already exists. Skip it." % filepath_output)
        return

    print("Convert: %s -> %s" % (filepath_input, filepath_output))

    ### Unzip downloaded file.
    with gzip.open(filepath_input, "rb") as ifp:
        data = ifp.read()

    ### Parse header information.
    identifier = int.from_bytes(data[ 0: 4], BYTE_ORDER)
    num_images = int.from_bytes(data[ 4: 8], BYTE_ORDER)

    if identifier != 2049:
        print("Input file '%s' does not seems to be MNIST image file." % filepath)

    labels = np.array([int(b) for b in data[8:]]).reshape((num_images, ))

    np.save(filepath_output, labels)


if __name__  == "__main__":

    ### Download the MNIST dataset.
    download_MNIST("train-images-idx3-ubyte.gz")
    download_MNIST("train-labels-idx1-ubyte.gz")
    download_MNIST("t10k-images-idx3-ubyte.gz")
    download_MNIST("t10k-labels-idx1-ubyte.gz")

    ### Convert to npy format.
    convert_image_data("train-images-idx3-ubyte.gz", "MNIST_train_images.npy")
    convert_label_data("train-labels-idx1-ubyte.gz", "MNIST_train_labels.npy")
    convert_image_data("t10k-images-idx3-ubyte.gz",  "MNIST_test_images.npy")
    convert_label_data("t10k-labels-idx1-ubyte.gz",  "MNIST_test_labels.npy")


#################################### SOURCE FINISH ##################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
# Ganerated by grasp version 0.0
