Random Fourier Features: Sample RFF Regression
====


This python script provides an example for regression with random fourier features.
Our module for random fourier features (PyRFFF.py) needs scikit-learn as a backend of SVM solver therefore you need to install scikit-learn.


## Usage

If you don't have scikit-learn, please run the following as root to install it:

    # pip3 install scikit-learn

You can run the example code by the following command:

    $ python3 sample_rff_regression.py


## Results of Regression with RFF

The following figure shows a regression results for the function y = sin(x^2) with RFF where dimension of RFF is 16.

<p align="center">
  <img src="figure_rff_regression.png" width="480" height="320" alt="Regression results for function y = sin(x^2) with RFF" />
</p>

