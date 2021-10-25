# Automatic Hyper Parameter Tuning using Optuna

This directory provides an example of automatic hyper parameter tuning and visualization the behavior of the tuning process.
In this example, we use `RFFGPC` as a model and the Boston house-price dataset as a dataset.
The backend of the hyper parameter tuning is [Optuna](https://optuna.org/).


## Installation

See [this document](../..SETUP.md) for more details.

### Docker image (recommended)

```console
$ docker pull tiskw/pytorch:latest
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
$ docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
$ cd examples/optuna_and_shap_for_boston_housing/
```

### Install on your environment (easier, but pollute your development environment)

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install optuna                           # Required for hyper parameter tuning
```


## Usage

The `rfflearn` library contains easier interface to Optuna.
The sample script `main_optuna_for_boston_housing.py` provides sample usage of the interface.

```console
$ python3 main_optuna_for_boston_housing.py
```

The best parameters and best estimators are returned like the following:

```console
[I 2021-01-10 03:43:58,825] A new study created in memory with name: no-name-227d6170-3621-49ca-a3e1-12efd44950fb
[I 2021-01-10 03:43:58,841] Trial 0 finished with value: 0.6769491684645892 and parameters: {'dim_kernel': 65, 'std_kernel': 0.00017494157111930657}. Best is trial 0 with value: 0.6769491684645892.    
[I 2021-01-10 03:43:58,851] Trial 1 finished with value: -1.861664555317465 and parameters: {'dim_kernel': 71, 'std_kernel': 7.526405082226793e-06}. Best is trial 0 with value: 0.6769491684645892. 
...
[I 2021-01-10 03:44:05,485] Trial 299 finished with value: 0.6762523470264499 and parameters: {'dim_kernel': 48, 'std_kernel': 3.36900720312351e-06}. Best is trial 251 with value: 0.8925418926511931.
- study.best_params: {'dim_kernel': 51, 'std_kernel': 1.5584067988983678e-07}
- study.best_value: 0.8925418926511931
- study.best_model: <rfflearn.cpu.rfflearn_cpu_regression.RFFRegression object at 0x7f8f15980fd0>
- R2 score of the best model:  0.8925418926511931
```

### Visualization of the searching behavior

The foillowing command generate a gif animation which shows the behavior of the hyper parameter search.

```console
$ python3 main_optuna_for_boston_housing.py --visualize
```

<div align="center">
  <img src="./figures/hyper_parameter_search.gif" width="500" height="400" alt="Animation of hyper parameter search behavior" />
</div>

### Training on GPU

Open the script file, replace `rfflearn.cpu` as `rfflean.gpu` and run the script again.
