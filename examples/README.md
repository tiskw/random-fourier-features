# Example of RFFLearn module

This directory contains the example code for the `rfflearn.cpu` and `rfflearn.gpu` module.


## Least square regression with random Fourier features

An example of regression with random Fourier features.
See [README.md](./least_square_regression/README.md) for more details.

<div align="center">
  <img src="./least_square_regression/figure_least_square_regression.png" width="640" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>


## Gaussian process regression with random Fourier features

An example of Gaussian process regression with random Fourier features.
See [README.md](./gpr_sparse_data/README.md) for more details.

<div align="center">
  <img src="./gpr_sparse_data/figure_rff_gpr_sparse_data.png" width="640" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>


## Gaussian process classification with random Fourier features

An example of the Gaussian process classification with random Fourier features.
See [README.md](./gpc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./gpc_for_mnist/figures/figure_inference_time_and_accuracy_on_MNIST.png" width="671" height="351" alt="Inference Time vs Accuracy on MNIST" />
</div>


## Support vector classification with random Fourier features

An example of support vector classification for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset with random Fourier features.
See [README.md](./svc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./svc_for_mnist/figures/figure_Inference_Time_and_Accuracy_on_MNIST.png" width="671" height="351" alt="Inference Time vs Accuracy on MNIST" />
</div>


## Canonical correlation analysis with random Fourier features

An example of canonical correlation analysis with random Fourier features.
See [README.md](./cca_for_artificial_data/README.md) for more details.

<div align="center">
  <img src="./cca_for_artificial_data/figure_cca_for_artificial_data.png" width="840" height="640" alt="CCA results for artificial dataset" />
</div>


## Principal component analysis with random Fourier features

An example of principal component analysis for swiss roll dataset with random Fourier features.
See [README.md](./pca_for_swissroll/README.md) for more details.

<div align="center">
  <img src="./pca_for_swissroll/figure_pca_for_swissroll_3d.png" width="400" height="300" alt="3D plot of input data (10,000 points of swiss roll)" />
  <img src="./pca_for_swissroll/figure_pca_for_swissroll_rffpca.png" width="400" height="300" alt="2D plot of 1st/2nd PC obtained by RFF PCA" />
</div>


## Automatic hyper parameter tuning using Optuna

An example of automatic hyper parameter tuning functions that uses [Optuna](https://optuna.org/) as a backend.
See [README.md](./optuna_for_boston_housing/README.md) for more details.

<div align="center">
  <img src="./optuna_for_boston_housing/figures/hyper_parameter_search.gif" width="500" height="400" alt="Animation of hyper parameter search behavior" />
</div>


## Feature importance of trained model and visualization of the importance

An example of automatic hyper parameter tuning functions that uses [Optuna](https://optuna.org/) as a backend.
See [README.md](./feature_importances_for_boston_housing/README.md) for more details.

<div align="center">
  <img src="./feature_importances_for_boston_housing/figure_boston_housing_shap_importance.png" width="400" height="300" alt="Permutation importances of Boston housing dataset" />
  <img src="./feature_importances_for_boston_housing/figure_boston_housing_permutation_importance.png" width="400" height="300" alt="SHAP importances of Boston housing dataset" />
</div>


## Support vector classification with random Fourier features with batch learning

An example of support vector classification with batch learning for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset with random Fourier features.
See [README.md](./svc_for_mnist_batch/README.md) for more details.

However, you do not need to pay much attention to this example because
[non-batch learning approach](./svc_for_mnist/README.md)
(i.e. usual SVC training using all dataset) now shows higher performance than the batch learning approach.
