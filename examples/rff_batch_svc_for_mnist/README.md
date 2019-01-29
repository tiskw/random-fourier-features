Random Fourier Features RFF Batch SVC for MNIST
====


This python script provide an example for RFF/ORF SVC with MNIST dataset.
Our module for random fourier features (PyRFFF.py) needs scikit-learn as a backend of SVM solver therefore you need to install scikit-learn.


## Preparation

If you don't have scikit-learn, please run the following as root to install it:

    # pip3 install scikit-learn

Also, you need to download and convert MNIST data before running sample code by the following command:

    $ cd ../../data
    $ python3 download_and_convert_mnist.py

Original MNIST data will be downloaded automatically and converted to .npy file.


## Usage

After generating MNIST .npy files, run sample script by the following command:

    $ python3 sample_rff_batch_svc_for_mnist.py


## Results of Support Vector Classification with Batch RFF

I implemented a code for RFF SVM with batch learning and evaluate its accuracy on MNIST dataset.
I've got the following results on my computing environmrnt (CPU: Intl Core i5 5250U, RAM: 4GB).
Learning time is even longer than kernel SVM.
However, accuracy is same as kernel SVM and, moreover, prediction time is much faster than kernel SVM.

| Method                         | Training time (sec) | Prediction time (us)| Score (%) |
| :----------------------------: | :-----------------: | :-----------------: | :-------: |
| SVM w/ batch RFF <br> d = 1024 | 2062.2 sec          | 108.6 us            | 96.4 %    |

<p align="center">
  <img src="figure_rff_batch_svc_for_mnist.png" width="480" height="320" alt="Accuracy for each epochs in SVM with batch RFF" />
</p>



