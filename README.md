# Random Fourier Features

Python module of random Fourier features (RFF) for kernel method,
like support vector machine [1], and Gaussian process model.
Features of this module are:

* interfaces of the module are quite close to the [scikit-learn](https://scikit-learn.org/),
* support vector classifier and Gaussian process regressor/classifier provides CPU/GPU training and inference,
* interface of [optuna](https://optuna.org/) for easier hyper parameter tuning,
* this repository provides example code that shows RFF is useful for actual machine learning tasks.

Now, this module supports

* canonical correlation analysis (`rfflearn.cpu.RFFCCA`).
* Gaussian process regression (`rfflearn.cpu.RFFGPR`,`rfflearn.gpu.RFFGPR`)
* Gaussian process classification (`rfflearn.cpu.RFFGPC`,`rfflearn.gpu.RFFGPC`)
* principal component analysis (`rfflearn.cpu.RFFPCA`).
* regression (`rfflearn.cpu.RFFRegression`),
* support vector classification (`rfflearn.cpu.RFFSVC`, `rfflearn.gpu.RFFSVC`),
* support vector regression (`rfflearn.cpu.RFFSVR`),

RFF can be applicable for many other machine learning algorithms, I will provide other functions soon.


## Requirements and installations

See [this document](https://tiskw.gitbook.io/rfflearn/) for more details.


## Minimal example

Interfaces provided by our module is quite close to Scikit-learn.
For example, the following Python code is a sample usage of `RFFSVC`
(support vector machine with random Fourier features) class.

```python
>>> import numpy as np
>>> import rfflearn.cpu as rfflearn                     # Import module
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Define input data
>>> y = np.array([1, 1, 2, 2])                          # Defile label data
>>> svc = rfflearn.RFFSVC().fit(X, y)                   # Training
>>> svc.score(X, y)                                     # Inference (on CPU)
1.0
>>> svc.predict(np.array([[-0.8, -1]]))
array([1])
```

This module supports training/inference on GPU.
For example, the following Python code is a sample usage of `RFFGPC`
(Gaussian process classifier with random Fourier features) class.
The following code requires GPU and tensorflow 2.x (tensorflow 1.x does not supported).

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

This module also have interfaces to optuna:

```python
>>> import numpy as np
>>> import rfflearn.cpu as rfflearn  # Import module
>>> train_set = (np.array([[-1, -1], [1, 1]]), np.array([1, 2]))     # Define training data
>>> valid_set = (np.array([[2, 1]]), np.array([2]))                  # Define validation data
>>> study = rfflearn.RFFSVC_tuner(train_set, valid_set, n_trials=10) # Start parameter tuing
>>> study.best_params                                                # Show best parameter
{'dim_kernel': 879, 'std_kernel': 0.6135046243705738}
>>> study.user_attrs["best_model"]                                   # Get best estimator
<rfflearn.cpu.rfflearn_cpu_svc.RFFSVC object at 0x7ff754049898>
```

See [examples](./examples/README.md) directory for more detailed examples.


## MNIST using random Fourier features

I applied SVC (support vector classifier) and GPC (Gaussian process classifire) with RFF to MNIST
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
- [ ] New function: feature importance (e.g. [SHAP](https://arxiv.org/abs/1705.07874))


## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)


## Reference

[1] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", NIPS, 2007.
[PDF](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)

[2] F. X. Yu, A. T. Suresh, K. Choromanski, D. Holtmann-Rice and S. Kumar, "Orthogonal Random Features", NIPS, 2016.
[PDF](https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf)


## Author

Tetsuya Ishikawa ([EMail](mailto:tiskw111@gmail.com), [Website](https://tiskw.gitlab.io/home/))
