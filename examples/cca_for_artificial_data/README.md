# Canonical Correlation Analysis using Random Fourier Features

This python script provides examples of regression canonical correlation analysis with random Fourier features.


## Installation

See [this document](https://tiskw.gitbook.io/rfflearn/tutorial#setting-up) for more details.

### Install on your environment (easier, but pollute your development environment)

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install optuna                           # Required only for hyper parameter tuning
```

### Docker image (recommended)

```console
$ docker pull tiskw/tensorflow:2021-01-08
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
$ docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/tensorflow:2021-01-08 bash
$ cd examples/gpr_sparse_data/
```


## Usage

```console
$ python3 main_cca_for_artificial_data.py
```

### Results of canonical correlation analysis with RFF

The input data X and Y have shape (number_of_samples, dimention) = (500, 2),
and the data is composed of 2 parts, correlated and noise part.
As shown in the following figure, `X[:, 0]` and `Y[:, 0]` have strong correlation,
however, `X[:, 1]` and `Y[:, 1]` are completely independent.
The linear CCA was failed to find the correlation, but CCA with random Fourier features succeeded because of its nonlinearity.

<div align="center">
  <img src="./figure_cca_for_artificial_data.png" width="840" height="640" alt="CCA results for artificial dataset" />
</div>

