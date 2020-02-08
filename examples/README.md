# Example of PyRFF module

This directory contains the example code for PyRFF module.

## Regression using Random Fourier Features

An example for regression using Random Fourier Features.
See [README.md](./rff_regression/README.md) for more details.

<p align="center">
  <img src="./rff_regression/figure_rff_regression.png" width="600" height="480" alt="Regression results for function y = sin(x^2) with RFF" />
</p>

## Support Vector Classification using Random Fourier Features

An example of support vector classification for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using Random Fourier Features.
See [README.md](./rff_svc_for_mnist/README.md) for more details.

<div align="center">
  <img src="./rff_svc_for_mnist/figures/figure_Inference_Time_and_Accuracy_on_MNIST.png" width="600" height="371" alt="Inference Time vs Accuracy on MNIST" />
</div>

## Support Vector Classification using Random Fourier Features with Batch Learning

An example of support vector classification with batch learning for [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using Random Fourier Features.
See [README.md](./rff_batch_svc_for_mnist/README.md) for more details.

However, you do not need to pay much attention for this example because
[non-batch learning approach](./rff_svc_for_mnist/README.md)
(i.e. usual SVC training using all dataset) now shows more higher performance than the batch learning approach.

