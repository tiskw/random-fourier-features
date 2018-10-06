Random Fourier Feature
====

Python implementation of random fourier features for support vector machine.
Now it only has a module for classification (RandomFourierFeatures.SVC), however I will provide other SVM functions soon.
Also, I'm preparing for C/C++ implementation of random fourier features.

## Requirement

- Python 3.6.6
- Numpy 1.13.3
- scikit-learn 0.19.1

## Usage

Before running sample code, you need to download and convert MNIST data by the following command:

    $ cd data
    $ python3 convert_mnist_to_npy.py

Original MNIST data will be downloaded automatically and converted to .npy file.
After generating MNIST .npy files, run sample script by the following command:

    $ cd source
    $ ./run.sh

This script run both of kernel SVM and SVM with RFF for MNIST data.
Results of these methods are logged and log files are stored in etc/ directory.

## Results

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I got the following results:

|Method|Training time (sec)|Prediction time (us)|Score (%)|
|:---|:---|:---|
|Kernel SVM|709.5 sec|19556 us|94.8 %|
|SVM w/ RFF|102.2 sec|12.2 us|94.8 %|

## Licence

MIT Licence (https://opensource.org/licenses/mit-license.php)

## Reference

A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", NIPS, 2007.
[PDF](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)

## Author

Tetsuya Ishikawa (https://github.com/tiskw)

