# Least Square Regression using Random Fourier Features

This directory provides an example of regression with Random Fourier Features.
The target function is y = sin(x^2), therefore linear regression cannot handle this function well.


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

```console
$ python3 main_rff_regression.py
```

## Results of Regression with RFF

The following figure shows regression results for the function y = sin(x^2) with RFF where the dimension of RFF is 16.

<div align="center">
  <img src="./figure_rff_regression.png" width="600" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>
