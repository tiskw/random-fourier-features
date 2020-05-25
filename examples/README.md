# Example of PyRFF module

This directory contains the example code for PyRFF module.

## Least Square Regression using Random Fourier Features

An example for regression using random fourier features.
See [README.md](./least_square_regression/README.md) for more details.

<div align="center">
  <img src="./least_square_regression/figure_rff_regression.png" width="640" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>

## Support Vector Classification using Random Fourier Features

An example of support vector classification for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using random fourier features.
See [README.md](./svc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./svc_for_mnist/figures/figure_Inference_Time_and_Accuracy_on_MNIST.png" width="600" height="371" alt="Inference Time vs Accuracy on MNIST" />
</div>

## Support Vector Classification using Random Fourier Features with Batch Learning

An example of support vector classification with batch learning for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using random fourier features.
See [README.md](./svc_for_mnist_batch/README.md) for more details.

However, you do not need to pay much attention to this example because
[non-batch learning approach](./svc_for_mnist/README.md)
(i.e. usual SVC training using all dataset) now shows higher performance than the batch learning approach.

## Gaussian Process Regression using Random Fourier Features

An example of Gaussian process regression using random fourier features.
See [README.md](./gpc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./gp_regression/figure_gp_regression.png" width="640" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</div>

## Gaussian Process Classification using Random Fourier Features

An example of Gaussian process classification using random fourier features.
See [README.md](./gpc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./gpc_for_mnist/figures/figure_inference_time_and_accuracy_on_MNIST.png" width="600" height="371" alt="Inference Time vs Accuracy on MNIST" />
</div>


