# Princaipal Component Analysis using Random Fourier Features

This python script provides an example of PCA with Random Fourier Features (RFF PCA).
Our module for Random Fourier Features (PyRFFF.py) needs scikit-learn as a backend of PCA solver therefore you need to install scikit-learn.


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

If you don't have scikit-learn or matplotlib, please run the following as root to install it:

```console
$ pip3 install scikit-learn matplorlib
```

You can run the example code by the following command:

```console
$ python3 main_pca_for_swissroll.py rff     # RFF PCA
$ python3 main_pca_for_swissroll.py kernel  # Kernel PCA
```


## Results of Regression with RFF

The following figure shows the input data (10,000 points of swiss roll) and results of RFF PCA.

<div align="center">
  <img src="./figure_pca_for_swissroll_3d.png" width="600" height="480" alt="3D plot of input data (10,000 points of swiss roll)" />
  <img src="./figure_pca_for_swissroll_rffpca.png" width="600" height="480" alt="2D plot of 1st/2nd PC obtained by RFF PCA" />
</div>


## Computation Time

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I've got the following results:

| Method                | Training time (sec) |
| :------------------:  | :-----------------: |
| Kernel PCA            | 6.01 sec            |
| RFF PCA <br> d = 1024 | 0.88 sec            |

