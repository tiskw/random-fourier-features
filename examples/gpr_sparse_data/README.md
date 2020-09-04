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
$ python3 main_gp_regression.py kernel  # Normal GP regression
$ python3 main_gp_regression.py rff     # RFF GP regression
```

## Results of Gaussian Process Regression with RFF

The following figure shows regression results for the function y = sin(x^2) with RFF where the dimension of RFF is 16.
RFF makes the training and inference speed much faster than the usual Gaussian process.
I would like to specially mention that the inference time of the RFF GPR is almost constant while the inference time of normal GPR grow rapidly.
The following table is a summary of training and inference (100 test data points) speed
under my environment (Intel Core i7-8665U@1.90GHz, 4GB RAM).

| Number of trainig samples | Number of test samples       | Training/Inference Time of GPR | Training/Inference Time of RFF GPR |
| :-----------------------: | :--------------------------: | :----------------------------: | :--------------------------------: |
|   1,000                   | 1 (average of 1,000 samples) | 1.50 s / 18.9 us               | 0.156 ms / 0.670 us                |
|   5,000                   | 1 (average of 1,000 samples) | 98.3 s / 105 us                |  6.14 ms / 0.921 us                |
|  10,000                   | 1 (average of 1,000 samples) |  468 s / 1.87 ms               |  11.3 ms / 0.700 us                |
|  50,000                   | 1 (average of 1,000 samples) |    - s / - s                   |  47.1 ms / 0.929 us                |
| 100,000                   | 1 (average of 1,000 samples) |    - s / - s                   |  93.5 ms / 0.852 us                |

<div align="center">
  <img src="./figure_gpr_sparse_data.png" width="600" height="480" alt="Regression results for function y = sin(x^2) using Gaussian process w/ RFF" />
</div>

