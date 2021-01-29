# Python script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 15, 2020
##################################################### SOURCE START #####################################################

import os
import pickle
import subprocess
import hashlib
import tarfile
import numpy as np

### Download a file from the given URL and check SHA256.
### This function returns True if and only if the SHA256 of the downloaded file is same as the ground truth.
def download_cifar10(filepath):

    url_origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    sha_256_gt = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"

    ### Calculate SHA256 of the given file and compare with ground truth.
    def has_correct_sha256sum(filepath, hash_sha256_gt):
        with open(filepath, "rb") as ifp:
            data = ifp.read()
        hash_sha256 = hashlib.sha256(data).hexdigest()
        return hash_sha256 == hash_sha256_gt

    command = "wget -c %s -O %s" % (url_origin, filepath)
    subprocess.run(command, shell = True)
    return has_correct_sha256sum(filepath, sha_256_gt)

### Read file and unpickle from Tar and TarInfo data.
def read_and_unpickle(tar, tarinfo):

    return pickle.loads(tar.extractfile(tarinfo).read(), encoding = "bytes")

### This function extract all images from one batch data of CIFAR10.
### Each batch data contains 10,000 images and labels.
def parse_cifar_images(batch):

    Xs_batch = np.zeros((10000, 32, 32, 3))
    Xs_batch[:,:,:, 0] = batch[b"data"][:, 2048:    ].reshape((10000, 32, 32))
    Xs_batch[:,:,:, 1] = batch[b"data"][:, 1024:2048].reshape((10000, 32, 32))
    Xs_batch[:,:,:, 2] = batch[b"data"][:,    0:1024].reshape((10000, 32, 32))
    return Xs_batch

### This function extract all labels from one batch data of CIFAR10.
### Each batch data contains 10,000 images and labels.
def parse_cifar_labels(batch):

    return np.array(batch[b"labels"], dtype = np.uint8)

### Load and return CIFAR10 dataset as numpy array which is compatible with OpenCV image
### that has shape (height, width, channel) and the channel order is Blue-Green-Red.
### Returned values of this function is ((Xs_train, ys_train), (Xs_valid, ys_valid)), where
###   - Xs_train: (50000, 32, 32, 3),   - ys_train: (50000, ),
###   - Xs_valid: (10000, 32, 32, 3),   - ys_valid: (10000, ).
### If archive of the CIFAR10 is not exists, this function will download the archive from official website automatically.
def create_dataset(filepath):

    ### Download the CIFAR10 archive from the official website.
    if not os.path.exists(filepath):

        success_dl = download_cifar10(filepath)

        ### Abort if failed to download.
        if not success_dl:
            print("Hash (SHA256) does not match. You should remove %s and try again." % filepath)
            exit("Error: cifar10.py: Download failed (hash does not match)")

    ### Extract data and label files from the archive. The archive has the following files:
    ###   * cifar-10-batches-py/readme.html   (Readme file)
    ###   * cifar-10-batches-py/batches.meta  (Binary header)
    ###   * cifar-10-batches-py/data_batch_1  (Pickled images and labels for training)
    ###   * cifar-10-batches-py/data_batch_2  (Pickled images and labels for training)
    ###   * cifar-10-batches-py/data_batch_3  (Pickled images and labels for training)
    ###   * cifar-10-batches-py/data_batch_4  (Pickled images and labels for training)
    ###   * cifar-10-batches-py/data_batch_5  (Pickled images and labels for training)
    ###   * cifar-10-batches-py/test_batch    (Pickled images and labels for testing)
    ### and the following code extract only data files, un-pickle and store them as dictionary format.
    with tarfile.open(filepath, "r") as tar:
        data = {tarinfo.name:read_and_unpickle(tar, tarinfo) for tarinfo in tar if tarinfo.isreg() and "_batch" in tarinfo.name}

    ### Create training data.
    Xs_train = np.concatenate([parse_cifar_images(data["cifar-10-batches-py/data_batch_%d" % n]) for n in range(1, 6)])
    ys_train = np.concatenate([parse_cifar_labels(data["cifar-10-batches-py/data_batch_%d" % n]) for n in range(1, 6)])

    ### Create testing data.
    Xs_valid = parse_cifar_images(data["cifar-10-batches-py/test_batch"])
    ys_valid = parse_cifar_labels(data["cifar-10-batches-py/test_batch"])

    np.save("CIFAR10_train_images.npy", Xs_train)
    np.save("CIFAR10_train_labels.npy", ys_train)
    np.save("CIFAR10_test_images.npy",  Xs_valid)
    np.save("CIFAR10_test_labels.npy",  ys_valid)

if __name__  == "__main__":

    ### Create file path of the local cache of the CIFAR10 archive.
    current_dir = os.path.dirname(__file__)
    filepath    = os.path.join(current_dir, "cifar-10-python.tar.gz")

    create_dataset(filepath)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
