# Canonical Correlation Analysis using Random Fourier Features

This python script provides an example of regression with Random Fourier Features.
Our module for Random Fourier Features (PyRFFF.py) needs scikit-learn as a backend of CCA solver therefore you need to install scikit-learn.


## Installation

See [this document](https://tiskw.gitbook.io/rfflearn/) for more details.

### Docker image (recommended)

If you don't like to pollute your development environment, it is a good idea to run everything inside a Docker container.
Scripts in this directory are executable on [this docker image](https://hub.docker.com/repository/docker/tiskw/tensorflow).

```console
$ docker pull tiskw/tensorflow:2020-05-29    # Download docker image from DockerHub
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO # Move to the root directory of this repository
$ docker run --rm -it --runtime=nvidia -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/tensorflow:2020-01-18 bash
$ cd examples/gpc_for_mnist/                 # Move to this directory
```

If you don't need GPU support, the option `--runtime=nvidia` is not necessary.

### Install Python packages (alternative)

The training end validation script requires `docopt`, `numpy`, `scipy`, `scikit-learn` and, if you need GPU support, `tensorflow-gpu`.
If you don't have them, please run the following as root to install them:

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install tensorflow-gpu                   # Required only for GPU inference
```


## Usage

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
  <img src="./figure_cca_for_artificial_data.png" width="840" height="640" alt="CCA results for artificial dataset" />
</div>
