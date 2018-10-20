Random Fourier Features
====

Python module of random fourier features and orthogonal random features for regression and support vector machines [1, 2].
This python module needs scikit-learn as a backend of SVM solver.
Now this mpdule only has a module for classification (RandomFourierFeatures.SVC),
however I will provide other SVM functions soon.

## Requirement

- Python 3.6.6
- Numpy 1.13.3
- Scipy 0.19.1
- scikit-learn 0.19.1

## Usage

Before running sample code, you need to download and convert MNIST data by the following command:

    $ cd data
    $ python3 download_and_convert_mnist.py

Original MNIST data will be downloaded automatically and converted to .npy file.
After generating MNIST .npy files, run sample script by the following command:

    $ cd source
    $ sh run.sh

This script run both of kernel SVM and SVM with RFF for MNIST data.
Results of these methods are logged and log files are stored in etc/ directory.

Regression and support vector classification is implemented in source/PyRFF.py.
Usage of the classes written in source/PyRFF.py is quite close to the classes provided by Scikit-learn.
The following is a sample usage of RFF regression class:

    >>> import numpy as np
    >>> import PyRFF as pyrff
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = X**2
    >>> reg = pyrff.RFFRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.predict(np.array([[1.5]]))
    array([[ 2.25415897]])

## Results of Regression with RFF

The following figure shows a regression results for the function y = sin(x^2) with RFF where dimension of RFF is 16.

<p align="center">
  <img src="./etc/output_main_reg_rff_plot.png" width="480" height="320" alt="Regression results for function y = sin(x^2) with RFF" />
</p>

## Results of Support Vector Classification with RFF and ORF

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I got the following results:

|Method|Training time (sec)|Prediction time (us)|Score (%)|
|:---:|:---:|:---:|:---:|
|Kernel SVM|214.2 sec|5837.4 us|96.3 %|
|SVM w/ RFF <br> d = 128|136.6 sec|21.3 us|93.4 %|
|SVM w/ ORF <br> d = 128|138.2 sec|21.4 us|93.4 %|

Learning time on RFF and ORF is faster than kernel SVM, moreover, prediction time of RFF and ORF is amazingly faster.
On the otherhand, accuracy of RFF and ORF is lower than kernel SVM.

## Results of Support Vector Classification with Batch RFF

I implemented a code for RFF SVM with batch learning and evaluate its accuracy on MNIST dataset.
I've got the following results. Learning time is even longer than kernel SVM.
However, accuracy is same as kernel SVM and, moreover, prediction time is much faster than kernel SVM.

|Method|Training time (sec)|Prediction time (us)|Score (%)|
|:---:|:---:|:---:|:---:|
|SVM w/ batch RFF <br> d = 1024|2062.2 sec|108.6 us|96.4 %|

<p align="center">
  <img src="./etc/output_main_svm_rff_batch_plot.png" width="480" height="320" alt="Accuracy for each epochs in SVM with batch RFF" />
</p>

## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)

## Reference

[1] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", NIPS, 2007.
[PDF](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)

[2] F. X. Yu, A. T. Suresh, K. Choromanski, D. Holtmann-Rice and S. Kumar, "Orthogonal Random Features", NIPS, 2016.
[PDF](https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf)

## Author

Tetsuya Ishikawa (https://github.com/tiskw)

