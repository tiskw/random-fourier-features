# Least Square Regression using Random Fourier Features

This python script provides an example of regression with Random Fourier Features.
Our module for Random Fourier Features (PyRFFF.py) needs scikit-learn as a backend of SVM solver therefore you need to install scikit-learn.


## Usage

If you don't have scikit-learn, please run the following as root to install it:

```console
$ pip3 install scikit-learn
```

You can run the example code by the following command:

```console
$ python3 main_rff_regression.py
```

## Results of Regression with RFF

The following figure shows regression results for the function y = sin(x^2) with RFF where the dimension of RFF is 16.

<div align="center">
  <img src="./figure_rff_regression.png" width="600" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>

