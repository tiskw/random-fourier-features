# Gaussian Process Regression using Random Fourier Features

This directory provides an example usage of the RFF Gaussian process regressor.

The training script in this directory supports both CPU/GPU training.
For the GPU training, you need to install Tensorflow 2.x (Tensorflow 1.x is not supported).


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
$ python3 main_gpr_sparse_data.py kernel  # Normal GP regression
$ python3 main_gpr_sparse_data.py rff     # RFF GP regression
$ python3 main_gpr_sparse_data.py orf     # ORF GP regression
```

## Results of Gaussian Process Regression with RFF/ORF

The following figure shows regression results for the function y = sin(x^2) with RFF where the dimension of RFF is 16.
RFF makes the training and inference speed much faster than the usual Gaussian process.
I would like to specially mention that the inference time of the RFF GPR is almost constant while the inference time of normal GPR grow rapidly.
The following table is a summary of training and inference (100 test data points) speed
under my environment (Intel Core i7-8665U@1.90GHz, 4GB RAM).

| Number of trainig samples | Number of test samples       | Training/Inference Time of GPR | Training/Inference Time of RFFGPR |
| :-----------------------: | :--------------------------: | :----------------------------: | :-------------------------------: |
|   1,000                   | 1 (average of 1,000 samples) | 1.50 s / 18.9 us               | 0.156 ms / 0.670 us               |
|   5,000                   | 1 (average of 1,000 samples) | 98.3 s / 105 us                |  6.14 ms / 0.921 us               |
|  10,000                   | 1 (average of 1,000 samples) |  468 s / 1.87 ms               |  11.3 ms / 0.700 us               |
|  50,000                   | 1 (average of 1,000 samples) |    - s / - s                   |  47.1 ms / 0.929 us               |
| 100,000                   | 1 (average of 1,000 samples) |    - s / - s                   |  93.5 ms / 0.852 us               |

<div align="center">
  <img src="./figure_gpr_sparse_data.png" width="600" height="480" alt="Regression results for function y = sin(x^2) using Gaussian process w/ RFF" />
</div>
