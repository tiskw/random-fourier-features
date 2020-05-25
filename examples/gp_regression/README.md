# Gaussian Process Regression using Random Fourier Features

The Gaussian process can be accelerated by the RFF technique.
This python script provides an example of Gaussian process regression with random Fourier features.
Our module for Gaussian Process (RFFGPR) needs Numpy and Docopt.


## Usage

If you don't have docopt, please run one of the following commands as root to install it:

```console
$ pip3 install docopt         # using pip
$ apt install python3-docopt  # using apt
```

You can run the example code by the following command:

```console
$ python3 main_gp_regression.py kernel  # Kernel regression
$ python3 main_gp_regression.py rff     # RFF regression
```

## Results of Gaussian Process Regression with RFF

The following figure shows regression results for the function y = sin(x^2) with RFF where the dimension of RFF is 16.
RFF makes the training and inference speed much faster than the usual Gaussian process.
The following table is a summary of training and inference (100 test data points) speed
under my environment (Intel Core i7-8665U@1.90GHz, 4GB RAM).

| Number of trainig samples | Training/Inference Time of GP | Training/Inference Time of GP w/ RFF |
| :-----------------------: | :---------------------------: | :----------------------------------: |
|   1,000                   | 202 ms / 338 ms               | 1.48 ms / 1.28 ms                    |
|   5,000                   | 26.2 s / 42.9 s               | 6.14 ms / 1.25 ms                    |
|  10,000                   |  198 s / 328 s                | 11.2 ms / 1.36 ms                    |
|  50,000                   |    - s / - s                  | 70.5 ms / 1.38 ms                    |
| 100,000                   |    - s / - s                  |  142 ms / 1.70 ms                    |

<div align="center">
  <img src="./figure_gp_regression.png" width="600" height="480" alt="Regression results for function y = sin(x^2) using Gaussian process w/ RFF" />
</div>

