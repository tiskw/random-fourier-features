# Canonical Correlation Analysis using Random Fourier Features

This python script provides an example of regression with Random Fourier Features.
Our module for Random Fourier Features (PyRFFF.py) needs scikit-learn as a backend of CCA solver therefore you need to install scikit-learn.


## Usage

If you don't have scikit-learn and matplotlib, please run the following as root to install it:

```console
$ pip3 install scikit-learn matplotlib
```

You can run the example code by the following command:

```console
$ python3 main_cca_for_artificial_data.py
```

## Results of Regression with RFF

The input data X and Y have shape (number_of_samples, dimention) = (500, 2),
and the data is composed of 2 parts, correlated and noise part.
As shown in the following figure, `X[:, 0]` and `Y[:, 0]` have string correlation,
but `X[:, 1]` and `Y[:, 1]` are completely independent.
The linear CCA was failed to find the correlation, but RFF CCA succeeded.

<div align="center">
  <img src="./figure_cca_for_artificial_data.png" width="840" height="640" alt="CCS results for artificial dataset" />
</div>
