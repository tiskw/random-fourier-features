Automatic Hyper Parameter Tuning using Optuna
====================================================================================================

This directory provides an example of automatic hyperparameter tuning and visualization of the
tuning process. In this example, we use `RFFRegressor` as a model and the California housing
dataset as a data.


Installation
----------------------------------------------------------------------------------------------------

See [this document](../..SETUP.md) for more details.

### Docker image (recommended)

```console
docker pull tiskw/pytorch:latest
cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
cd examples/optuna_and_shap_for_california_housing/
```

### Install on your environment (easier, but pollute your development environment)

```console
pip3 install docopt numpy scipy scikit-learn  # Necessary packages
pip3 install optuna                           # Required for hyper parameter tuning
apt install apngasm                           # Required for tuning process visualization
```

The `apngasm` is a command to generate APNG animation.
This command is required only for specifying `--visualize` option that generates the APNG animation below.


Usage
----------------------------------------------------------------------------------------------------

The `rfflearn` module contains wrapper interfaces to [Optuna](https://optuna.org/).
The sample script `main_optuna_for_california_housing.py` provides sample usage of the interface.

```console
python3 main_optuna_for_california_housing.py
```

The best parameters and best estimators are returned like the following:

```console
[I 2023-05-05 12:53:56,844] A new study created in memory with name: no-name-21ef2418-55da-406e-82a3-688272499c8c
[I 2023-05-05 12:53:57,406] Trial 0 finished with value: -0.035425696107509674 and parameters: {'dim_kernel': 455, 'std_kernel': 9.82951337287064}. Best is trial 0 with value: -0.035425696107509674.
[I 2023-05-05 12:53:57,647] Trial 1 finished with value: 0.6490642072770361 and parameters: {'dim_kernel': 171, 'std_kernel': 1.1330193705188478}. Best is trial 1 with value: 0.6490642072770361.
...
[I 2023-05-05 12:57:40,541] Trial 499 finished with value: 0.7075758192249759 and parameters: {'dim_kernel': 277, 'std_kernel': 0.8911256847874699}. Best is trial 399 with value: 0.7405694965884491.
- study.best_params: {'dim_kernel': 421, 'std_kernel': 0.5757745831870693}
- study.best_value: 0.7405694965884491
- study.best_model: <rfflearn.cpu.rfflearn_cpu_regression.RFFRegression object at 0x7f86e152aec0>
- R2 score of the best model:  0.6875709095108584
```

### Visualization of the tuning process

The following command generates an APNG animation that visualizes the process of the hyperparameter search.

```console
python3 main_optuna_for_california_housing.py --visualize
```

<div align="center">
  <img src="./hyper_parameter_search.png" width="500" alt="Animation of hyper parameter search behavior" />
</div>

### Training on GPU

Open the script file, replace `rfflearn.cpu` as `rfflean.gpu` and run the script again.
