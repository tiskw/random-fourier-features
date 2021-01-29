# Random Fourier Features

This repository provides Python module `rfflearn`
which is a library of random Fourier features [1, 2] for kernel method,
like support vector machine and Gaussian process model.
Features of this module are:

* interfaces of the module are quite close to the [scikit-learn](https://scikit-learn.org/),
* support vector classifier and Gaussian process regressor/classifier provides CPU/GPU training and inference,
* interface to [optuna](https://optuna.org/) for easier hyper parameter tuning,
* this repository provides [example code](./examples/README.md) that shows RFF is useful for actual machine learning tasks.

Now, this module supports the following methods:

| Method                          | CPU support                  | GPU support           |
| ------------------------------- | ---------------------------- | --------------------- |
| canonical correlation analysis  | `rfflearn.cpu.RFFCCA`        | -                     |
| Gaussian process regression     | `rfflearn.cpu.RFFGPR`        | `rfflearn.gpu.RFFGPR` |
| Gaussian process classification | `rfflearn.cpu.RFFGPC`        | `rfflearn.gpu.RFFGPC` |
| principal component analysis    | `rfflearn.cpu.RFFPCA`        | `rfflearn.gpu.RFFPCA` |
| regression                      | `rfflearn.cpu.RFFRegression` | -                     |
| support vector classification   | `rfflearn.cpu.RFFSVC`        | `rfflearn.gpu.RFFSVC` |
| support vector regression       | `rfflearn.cpu.RFFSVR`        | -                     |

RFF can be applicable for many other machine learning algorithms, I will provide other functions soon.

For more information, see the [user manual of rfflearn](https://tiskw.gitbook.io/rfflearn/).


## Requirements and installations

The author recommend to use [docker](https://www.docker.com/) for building development environment.
See [this document](https://tiskw.gitbook.io/rfflearn/tutorial#setting-up) for more details.

If you don't mind to pollute your own environment (or you are already inside a docker container),
just run the following command for installing required packages:

```console
pip3 install -r requirements.txt
```


## Minimal example

Interfaces provided by our module is quite close to scikit-learn.
For example, the following Python code is a sample usage of `RFFSVC`
(support vector machine with random Fourier features) class.

```python
>>> import numpy as np
>>> import rfflearn.cpu as rfflearn                     # Import module
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Define input data
>>> y = np.array([1, 1, 2, 2])                          # Defile label data
>>> svc = rfflearn.RFFSVC().fit(X, y)                   # Training (on CPU)
>>> svc.score(X, y)                                     # Inference (on CPU)
1.0
>>> svc.predict(np.array([[-0.8, -1]]))
array([1])
```

This module supports training/inference on GPU.
For example, the following Python code is a sample usage of `RFFGPC`
(Gaussian process classifier with random Fourier features) on GPU.
The following code requires PyTorch (>= 1.7.0).

```python
>>> import numpy as np
>>> import rfflearn.gpu as rfflearn                     # Import module
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Define input data
>>> y = np.array([1, 1, 2, 2])                          # Defile label data
>>> gpc = rfflearn.RFFGPC().fit(X, y)                   # Training on GPU
>>> gpc.score(X, y)                                     # Inference on GPU
1.0
>>> gpc.predict(np.array([[-0.8, -1]]))
array([1])
```

See [examples](./examples/README.md) directory for more detailed examples.


## Example1: MNIST using random Fourier features

I tried SVC (support vector classifier) and GPC (Gaussian process classifire) with RFF to MNIST dataset
which is the famous benchmark dataset of the image classification task,
and I've got better performance and much faster inference speed than kernel SVM.
The following table gives a brief comparison of kernel SVM, SVM with RFF and GPC with RFF.
See the example of [RFF SVC module](./examples/svc_for_mnist/README.md)
and [RFF GP module](./examples/gpc_for_mnist/README.md) for mode details.

| Method         | RFF dimension | Inference time (us) | Score (%) |
|:--------------:|:-------------:|:-------------------:|:---------:|
| Kernel SVM     | -             | 4644.9 us           | 96.3 %    |
| RFF SVC        |  512          | 39.0 us             | 96.5 %    |
| RFF SVC        | 1024          | 96.1 us             | 97.5 %    |
| RFF SVC (GPU)  | 1024          | 2.38 us             | 97.5 %    |
| RFF GPC        | 5120          | 342.1 us            | 98.2 %    |
| RFF GPC (GPU)  | 5120          | 115.0 us            | 98.2 %    |

<div align="center">
  <img src="./figures/Inference_Time_and_Accuracy_on_MNIST_SVC_and_GPC.png" width="763" height="371" alt="Accuracy for each epochs in RFF SVC" />
</div>


## Example2: Visualization of feature importance

This module also have interfaces to some feature importance methods, like SHAP [3] and permutation importance [4].
I tried SHAP and permutation importance to `RFFGPR` trained by Boston house-price dataset,
and the following is the visualization results obtained by `rfflearn.shap_feature_importance` and `rfflearn.permutation_feature_importance`.

<div align="center">
  <img src="./examples/feature_importances_for_boston_housing/figure_boston_housing_shap_importance.png" width="400" height="300" alt="Permutation importances of Boston housing dataset" />
  <img src="./examples/feature_importances_for_boston_housing/figure_boston_housing_permutation_importance.png" width="400" height="300" alt="SHAP importances of Boston housing dataset" />
</div>


## Notes

- Name of this module is changed from `pyrff` to `rfflearn` on Oct 2020,
  because the package `pyrff` already exists in PyPI.
- If a number of training data are huge, error message like
  `RuntimeError: The task could not be sent to the workers as it is too large for 'send_bytes'`.
  will be raised from the joblib library. The reason for this error is that sklearn.svm.LinearSVC uses
  joblib as a multiprocessing backend, but joblib cannot deal huge size of the array which cannot be managed
  with 32-bit address space. In this case, please try `n_jobs = 1` option for `RFFSVC` or `ORFSVC` function.
  Default settings are `n_jobs = -1` which means automatically detect available CPUs and use them.
  (This bug information was reported by Mr. Katsuya Terahata @ Toyota Research Institute Advanced Development.
  Thank you so much for the reporting!)


## TODO

- [ ] New function: implementation of batch RFF GP (but my GPU is too poor to try this...)
- [ ] New function: implementation of RFF Logistic GP


## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)


## Reference

[1] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", NIPS, 2007.
[PDF](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)

[2] F. X. Yu, A. T. Suresh, K. Choromanski, D. Holtmann-Rice and S. Kumar, "Orthogonal Random Features", NIPS, 2016.
[PDF](https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf)

[3] S. M. Lundberg and S. Lee, "A Unified Approach to Interpreting Model Predictions", NIPS, 2017.
[PDF](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

[4] L. Breiman, "Random Forests", Machine Learning, vol. 45, pp. 5-32, Springer, 2001.
[Springer website](https://doi.org/10.1023/A:1010933404324).


## Author

Tetsuya Ishikawa ([EMail](mailto:tiskw111@gmail.com), [Website](https://tiskw.gitlab.io/home/))
