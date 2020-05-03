Random Fourier Features
====

Python module of Random Fourier Features (RFF) for regression, support vector classification [1] and gaussian process.
Features of this RFF module are:
* interfaces of the module is quite close to the scikit-learn,
* module for Support Vector Classification provides GPU inference,
* module for Gaussian Process provides GPU train and inference.

Now, this module only supports
* regression (`PyRFF.RFFRegression`)
* support vector classification (`PyRFF.RFFSVC`, `PyRFF_GPU.RFFSVC_GPU`),
* gaussian process (`PyRFF.GaussianProcessRegression`, `PyRFF.GaussianProcessClassifire`, `PyRFF.GaussianProcessClassifier_GPU`)
however, I will provide other functions soon.


## Requirement

- Python 3.6.9
- docopt 0.6.2
- Numpy 1.18.1
- scikit-learn 0.22.1
- Tensorflow 2.1.0 (for GPU inference)


## Minimal example

Interfeces provided by our module is quite close to Scikit-learn.
For example, the following Python code is a sample usage of RFF regression class:

```python
>>> import numpy as np
>>> import PyRFF as pyrff                               # Import our module
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Define input data
>>> y = np.array([1, 1, 2, 2])                          # Defile label data
>>> svc = pyrff.RFFSVC().fit(X, y)                      # Training
>>> svc.score(X, y)                                     # Inference (on CPU)
1.0
>>> svc.predict(np.array([[-0.8, -1]]))
array([1])
```

Also, you are able to run the inference on GPU by adding only two lines, if you have Tensorflow 2.x.

```python
>>> import PyRFF_GPU as pyrff_gpu    # Import additional module
>>> svc = pyrff_gpu.RFFSVC_GPU(svc)  # Convert to GPU model
>>> svc.score(X, y)                  # Inference on GPU
1.0
```

See `examples/` directory for more detailed examples.


## MNIST using RFF and SVM

I applied SVM with RFF to MNIST which is a famous benchmark dataset for classification task,
and I've got a better performance and much faster inference speed than kernel SVM.
The following table gives a brief comparison of kernel SVM and SVM with RFF.
See [the example of RFF SVC module](./examples/rff_svc_for_mnist/README.md) for mode details.

| Method                   | Inference time (us) | Score (%) |
|:------------------------:|:-------------------:|:---------:|
| Kernel SVM               | 4644.9 us           | 96.3 %    |
| SVM w/ RFF (d=512)       | 39.0 us             | 96.5 %    |
| SVM w/ RFF (d=1024)      | 96.1 us             | 97.5 %    |
| SVM w/ RFF (d=1024, GPU) | 2.38 us             | 97.5 %    |
| GP w/ RFF (d=5120, CPU)  | 342.1 us            | 98.2 %    |

<div align="center">
  <img src="./examples/rff_svc_for_mnist/figures/figure_Inference_Time_and_Accuracy_on_MNIST.png" width="600" height="371" alt="Accuracy for each epochs in SVM with batch RFF" />
</div>


## Notes

 * If number of training data is huge, error message like
   `RuntimeError: The task could not be sent to the workers as it is too large for 'send_bytes'.`
   will be raised from the joblib library. The reason of this error is that sklearn.svm.LinearSVC uses
   joblib as a multiprocessing backend, but joblib cannot deal huge size of array which cannot be managed
   with 32 bit address space. In this case, please try `n_jobs = 1` option for `RFFSVC` or `ORFSVC` function.
   Default settings is `n_jobs = -1` which means automatically detect available CPUs and use them.
   (This bug information was reported by Mr. Katsuya Terahata @ Toyota Research Institute Advanced Development.
   Thank you so much for the reporting!)

## Licence

[MIT Licence](https://opensource.org/licenses/mit-license.php)


## Reference

[1] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", NIPS, 2007.
[PDF](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)

[2] F. X. Yu, A. T. Suresh, K. Choromanski, D. Holtmann-Rice and S. Kumar, "Orthogonal Random Features", NIPS, 2016.
[PDF](https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf)


## Author

Tetsuya Ishikawa ([EMail](mailto:tiskw111@gmail.com), [Website](https://tiskw.gitlab.io/home/))

