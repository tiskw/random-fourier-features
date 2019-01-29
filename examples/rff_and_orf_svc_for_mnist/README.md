Random Fourier Features: Sample RFF Batch SVC for MNIST
====


This python script provide an example for RFF Batch SVC with MNIST dataset.
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

    $ python3 sample_rff_batch_svc_for_mnist.py kernel   # Run kernel SVC
    $ python3 sample_rff_batch_svc_for_mnist.py rff      # Run SVC with RFF
    $ python3 sample_rff_batch_svc_for_mnist.py orf      # Run SVC with ORF


## Results of Support Vector Classification with RFF and ORF

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I've got the following results:

<center>
|Method                 |Training time (sec)|Prediction time (us)|Score (%)|
|:---------------------:|:-----------------:|:------------------:|:-------:|
|Kernel SVM             |214.2 sec          |5837.4 us           |96.3 %   |
|SVM w/ RFF <br> d = 128|136.6 sec          |21.3 us             |93.4 %   |
|SVM w/ ORF <br> d = 128|138.2 sec          |21.4 us             |93.4 %   |
</center>

Learning time on RFF and ORF is faster than kernel SVM,
moreover, prediction time of RFF and ORF is amazingly faster.
On the otherhand, accuracy of RFF and ORF is lower than kernel SVM.

