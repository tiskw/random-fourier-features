# Least Square Regression using Random Fourier Features

This directory provides an example of regression with Random Fourier Features.

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
$ python3 main_rff_regression_optuna.py
[I 2021-01-08 21:12:14,317] Trial 0 finished with value: 0.35751021523162807 and parameters: {'dim_kernel': 90, 'std_kernel': 1.234027988508813e-09}. Best is trial 0 with value: 0.35751021523162807．
...
[I 2021-01-08 21:12:24,807] Trial 299 finished with value: 0.7548302129658875 and parameters: {'dim_kernel': 41, 'std_kernel': 1.8019149197842532e-07}. Best is trial 218 with value: 0.9022873284215501.
Hyper parameter tuning: 10.542555 [s]
  - study.best_params: {'dim_kernel': 49, 'std_kernel': 2.0916950657792097e-07}
  - study.best_value: 0.9022873284215501
  - study.best_model: <rfflearn.cpu.rfflearn_cpu_regression.RFFRegression object at 0x7fae96be0b38>
Prediction with the best model: 0.001718 [s]
  - R2 score of the best model:  0.9022873284215501
Drawing figure: 0.176260 [s]
  - Saved to 'figure_rff_regression_optuna.png'
```

## Results of Regression with RFF

The following figure shows regression results for the function y = sin(x^2) with RFF where the dimension of RFF is 16.

<div align="center">
  <img src="./figure_rff_regression_optuna.png" width="600" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>
