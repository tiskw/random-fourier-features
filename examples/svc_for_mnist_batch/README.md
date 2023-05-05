# Bacthed Support Vector Classifier using Random Fourier Features for MNIST

This python script provides an example of RFFSVC with batch learning on the MNIST dataset.

However, you do not need to pay much attention to this example because
[non-batch learning approach](../rff_svc_for_mnist/README.md)
(i.e. usual SVC training using all dataset) now shows higher performance than the batch learning approach.
We expect that the batch learning exceeds non-batch learning in case of a larger and more complicated dataset than MNIST.


## Preparation

You need to download and convert MNIST data before running the training code.
Please run the following commands:

```console
$ cd ../../dataset/mnist
$ python3 download_and_convert_mnist.py
```

The MNIST dataset will be automatically downloaded, converted to `.npy` file
and saved under `dataset/mnist/` directory.


## Usage

After generating MNIST .npy files, run the sample script by the following command:

```console
$ python3 main_rff_batch_svc_for_mnist.py
```

You possibly have many warnings from LIBLINEAR (e.g. `ConvergenceWarning: Liblinear failed to converge, ...`).
You can ignore these warnings by redirecting STDERR to /dev/null like the following:

```console
$ python3 sample_rff_batch_svc_for_mnist.py 2> /dev/null
```


## Results of support vector classification with batched RFF

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I've got the following results:

| Method                           | Training time (sec) | Prediction time (us)| Score (%) |
| :------------------------------: | :-----------------: | :-----------------: | :-------: |
| SVM w/ batched RFF <br> d = 1024 | 2062.2 sec          | 108.6 us            | 96.4 %    |

<div align="center">
  <img src="./figure_rff_batch_svc_for_mnist.png" width="480" alt="Accuracy for each epoch in SVM with batch RFF" />
</div>

As the author pointed out at the top of this document, the above results are worse than [usual SVC training](../rff_svc_for_mnist/README.md).
However, the author's guess is that batch learning exceeds non-batch learning in case of a larger and more complicated dataset than MNIST.
