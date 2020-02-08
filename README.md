Random Fourier Features
====

Python module of random fourier features (RFF) for regression and support vector classification [1].
Features of this RFF module is that interfaces of the module is quite close to the scikit-learn.
Also, this module needs scikit-learn as a backend of SVM solver.

Now this module only has a module for regression (PyRFF.RFFRegression) and classification (PyRFF.RFFSVC),
however I will provide other SVM functions soon.

Also, RFF support vector classifier (RyRFF.RFFSVC) support GPU inference.


## Requirement

- Python 3.6.9
- docopt 0.6.2
- Numpy 1.18.1
- Scipy 1.4.1
- scikit-learn 0.22.1
- Tensorflow 2.1.0 (for GPU inference)


## Minimal example

Regression and support vector classification is implemented in `source/PyRFF.py`
and usage of the classes provided by this module is quite close to Scikit-learn.
For example, the following Python code is a sample usage of RFF regression class:

```python
>>> import numpy as np
>>> import PyRFF as pyrff
>>> X = np.array([[1], [2], [3], [4]])
>>> y = X**2
>>> reg = pyrff.RFFRegression().fit(X, y)
>>> reg.score(X, y)
1.0
>>> reg.predict(np.array([[1.5]]))
array([[ 2.25415897]])
```

See `example/` directory for more detailed examples.

## MNIST using RFF and SVM

I applied SVM with RFF to MNIST which is a famous benchmark dataset for classification task,
and I've got a better performance and much faster inference speed than kernel SVM.
The following table gives a brief comparison of kernel SVM and SVM with RFF.
See [the example of RFFSVC mofule](./examples/rff_svc_for_mnist/README.md) for mode details.

| Method                   | Inference time (us) | Score (%) |
| :---------------------:  | :-----------------: | :-------: |
| Kernel SVM               | 4644.9 us           | 96.3 %    |
| SVM w/ RFF (d=512)       | 39.0 us             | 96.5 %    |
| SVM w/ RFF (d=1024)      | 96.1 us             | 97.5 %    |
| SVM w/ RFF (d=1024, GPU) | 2.38 us             | 97.5 %    |

<p align="center">
  <img src="./examples/rff_svc_for_mnist/Inference_Time_and_Accuracy_on_MNIST.png" width="480" height="320" alt="Accuracy for each epochs in SVM with batch RFF" />
</p>


## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)


## Reference

[1] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", NIPS, 2007.
[PDF](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)

[2] F. X. Yu, A. T. Suresh, K. Choromanski, D. Holtmann-Rice and S. Kumar, "Orthogonal Random Features", NIPS, 2016.
[PDF](https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf)


## Author

Tetsuya Ishikawa (https://gitlab.com/tiskw)

