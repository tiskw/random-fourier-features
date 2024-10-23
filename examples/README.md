Example of rfflearn Module
====================================================================================================

This directory contains the example code for the `rfflearn.cpu` and `rfflearn.gpu` module.


Least square regression with random Fourier features
----------------------------------------------------------------------------------------------------

An example of regression with random Fourier features.
See [README.md](./least_square_regression/README.md) for more details.

<div align="center">
  <img src="./least_square_regression/figures/figure_least_square_regression.svg" width="640" alt="Regression results for function y = sin(x^2) with RFF" />
</div>


Gaussian process regression with random Fourier features
----------------------------------------------------------------------------------------------------

An example of Gaussian process regression with random Fourier features.
See [README.md](./gpr_sparse_data/README.md) for more details.

<div align="center">
  <img src="./gpr_sparse_data/figures/figure_rff_gpr_sparse_data.svg" width="640" alt="Regression results for function y = sin(x^2) with RFF" />
</div>


Gaussian process classification with random Fourier features
----------------------------------------------------------------------------------------------------

An example of the Gaussian process classification with random Fourier features.
See [README.md](./gpc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./gpc_for_mnist/figures/Inference_time_and_acc_on_MNIST_gpc.svg" width="640" alt="Inference Time vs Accuracy on MNIST" />
</div>


Support vector classification with random Fourier features
----------------------------------------------------------------------------------------------------

An example of support vector classification for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset
with random Fourier features. See [README.md](./svc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./svc_for_mnist/figures/Inference_time_and_acc_on_MNIST_svc.svg" width="640" alt="Inference Time vs Accuracy on MNIST" />
</div>


Canonical correlation analysis with random Fourier features
----------------------------------------------------------------------------------------------------

An example of canonical correlation analysis with random Fourier features.
See [README.md](./cca_for_artificial_data/README.md) for more details.

<div align="center">
  <img src="./cca_for_artificial_data/figures/figure_cca_for_artificial_data.png" width="840" alt="CCA results for artificial dataset" />
</div>


Principal component analysis with random Fourier features
----------------------------------------------------------------------------------------------------

An example of principal component analysis for the swiss roll dataset with random Fourier features.
See [README.md](./pca_for_swissroll/README.md) for more details.

<div align="center">
  <img src="./pca_for_swissroll/figures/figure_pca_for_swissroll.svg" width="640" alt="3D plot of input data (10,000 points of swiss roll)" />
</div>


Automatic hyperparameter tuning using Optuna
----------------------------------------------------------------------------------------------------

An example of automatic hyperparameter tuning functions that uses [Optuna](https://optuna.org/)
as a backend. See [README.md](./optuna_for_california_housing/README.md) for more details.

<div align="center">
  <img src="./optuna_for_california_housing/figures/hyperparameter_search.png" width="500" alt="Animation of hyper parameter search behavior" />
</div>


Feature importance of trained model and visualization of the importance
----------------------------------------------------------------------------------------------------

An example of the computation of feature importance that uses
[scikit-learn](https://scikit-learn.org/) and [Optuna](https://optuna.org/) as backends.
See [README.md](./feature_importances_for_california_housing/README.md) for more details.

<div align="center">
  <img src="./feature_importances_for_california_housing/figures/figure_california_housing_shap_importance.png" width="450" alt="Permutation importances of Boston housing dataset" />
  <img src="./feature_importances_for_california_housing/figures/figure_california_housing_permutation_importance.png" width="450" alt="SHAP importances of Boston housing dataset" />
</div>

