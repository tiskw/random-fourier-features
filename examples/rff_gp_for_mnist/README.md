Random Fourier Features: RFF for MNIST dataset
====

These python scripts provide examples for RFF Gaussian Process with MNIST dataset.
Our module for Random Fourier Features (PyRFFF.py) needs numpy as a backend of numerical computation therefore you need to install it.
Also, these scripts support GPU for both training and inference. For running GPU training and inference,
you need to install Tensorflow 2.x (Tensorflow 1.x is not supported).


## Installation

### Install Python packages

The training end validation script requires `docopt`, `numpy` and, if you will run the inference on GPU, `tensorflow-gpu`.
If you don't have them, please run the following as root to install them:

```console
$ pip3 install docopt numpy    # Necessary packages
$ pip3 install tensorflow-gpu  # Required only for GPU inference
```

### Docker image (alternative)

If you don't like to pollute your development environment, it is a good idea to run everything inside Docker.
Scripts in this directory are executable on [this docker image](https://hub.docker.com/repository/docker/tiskw/tensorflow).

```console
$ docker pull tiskw/tensorflow:2020-01-18    # Download docker image from DockerHub
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO # Move to the root directory of this repository
$ docker run --rm -it --runtime=nvidia -v `pwd`:/work -w /work -u `id -u`:`id -u` tiskw/tensorflow:2020-01-18 bash
$ cd PATH_TO_THIS_DIRECTORY                  # Back to here (inside the docker container you launched).
```

If you don't need GPU support, `--runtime=nvidia` is not necessary.


## Dataset Preparation

Also, you need to download and convert MNIST data before running the sample code by the following command:

```console
$ cd ../../dataset/mnist
$ python3 download_and_convert_mnist.py
```

Original MNIST data will be automatically downloaded, converted to .npy file and save them under `mnist/` directory.


## Training

After generating MNIST .npy files, run sample scripts by the following command:

```console
$ python3 train_rff_svc_for_mnist_CPU.py  # CPU
$ python3 train_rff_svc_for_mnist_GPU.py  # GPU
```

Default hyperparameter settings are recommended one for test running.
However, in order to get a higher score, you need to change the parameters by command option.
The following command will generate `result.pickle` in which trained model and command arguments are stored.
See `--help` for details.


## Inference

You are able to run inference by the following command:

```console
$ python3 valid_rff_svc_for_mnist.py cpu         # Inference on CPU using scikit-learn
$ python3 valid_rff_svc_for_mnist.py tensorflow  # Inference on GPU using Tensorflow
```


## Results of Support Vector Classification with RFF

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB, GPU: GTX1050Ti), I've got the following results:

| Method                  | Training time (sec) | Inference time (us) | Score (%) | Note                           |
| :---------------------: | :-----------------: | :-----------------: | :-------: |:------------------------------:|
| Kernel SVM              | 177.8 sec           | 4644.9 us           | 96.3 %    |                                |
| GP w/ RFF <br> d = 512  |   6.2 sec           | 66.21 us            | 96.3 %    | `std_kernel=0.1,std_error=0.5` |
| GP w/ RFF <br> d = 5120 | 111.7 sec           | 342.1 us            | 98.2 %    | `std_kernel=0.1,std_error=0.5` |

<!--
<div align="center">
  <img src="./figures/figure_Inference_Time_and_Accuracy_on_MNIST.png" width="601" height="371" alt="Inference Time vs Accuracy on MNIST" />
</div>
-->

Where score means test accuracy of MNIST dataset and inference time means inference time for one image.

Commonly used techniques like data normalization and dimension reduction using PCA is also used in the above analysis.
See comments in the Python script for details.

The Score of RFF GP is better than kernel SVM, moreover, the inference time of RFF GP is amazingly faster.

The following figure shows a tradeoff between the accuracy and inference time of RFF.

<!--
<div align="center">
  <img src="./figures/figure_rff_svc_for_mnist.png" width="480" height="320" alt="Accuracy for each dimention in RFF SVC" />
</div>
-->

