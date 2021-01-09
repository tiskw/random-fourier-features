# Gaussian Process Classifier using Random Fourier Features for MNIST Dataset

This directory provides an example of the Gaussian process classifier with random Fourier features for MNIST dataset.

The training script in this directory supports both CPU/GPU training.
For the GPU training, you need to install Tensorflow 2.x (Tensorflow 1.x is not supported).


## Installation

See [this document](https://tiskw.gitbook.io/rfflearn/tutorial#setting-up) for more details.

### Install on your environment (easier, but pollute your development environment)

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install tensorflow-gpu                   # Required only for GPU training/inference
$ pip3 install optuna                           # Required only for hyper parameter tuning
```

### Docker image (recommended)

```console
$ docker pull tiskw/tensorflow:2021-01-08
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
$ docker run --rm -it --gpus=all -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/tensorflow:2021-01-08 bash
$ cd examples/gpr_sparse_data/
```

If you don't need GPU support, the option `--gpus=all` is not necessary.


## Usage

### Dataset preparation

You need to download and convert MNIST data before running the training code.
Please run the following commands:

```console
$ cd ../../dataset/mnist
$ python3 download_and_convert_mnist.py
```

The MNIST dataset will be automatically downloaded, converted to `.npy` file
and saved under `dataset/mnist/` directory.

### Training and inference

After generating `.npy` files of MNIST, run the training script by either of the following commands:

```console
$ python3 train_gpc_for_mnist.py cpu  # training on CPU
$ python3 train_gpc_for_mnist.py gpu  # training on GPU
```

The above command will show a test score, and generate `result.pickle` in which a trained model and command arguments are stored.

Default hyperparameter settings are recommended parameters for environment test.
However, to get a higher score, you need to change the parameters by command options.
See `--help` for details.

### Results of Gaussian process classification with RFF

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB, GPU: GTX1050Ti), I've got the following results:

| Method                          | Training time (sec) | Inference time (us) | Score (%) | Note                           |
|:-------------------------------:|:-------------------:|:-------------------:|:---------:|:------------------------------:|
| Kernel SVC <br> (reference)     | 177.8 sec           | 4644.9 us           | 96.3 %    |                                |
| GPC w/ RFF <br> (d = 512)       |   6.2 sec           |  66.21 us           | 96.3 %    | `std_kernel=0.1,std_error=0.5` |
| GPC w/ RFF <br> (d = 5120)      | 111.7 sec           |  342.1 us           | 98.2 %    | `std_kernel=0.1,std_error=0.5` |
| GPC w/ ORF <br> (d = 5120)      | 114.3 sec           |  337.8 us           | 98.2 %    | `std_kernel=0.1,std_error=0.5` |
| GPC w/ RFF <br> (d = 5120, GPU) | 143.3 sec           |  115.0 us           | 98.2 %    | `std_kernel=0.1,std_error=0.5` |

<div align="center">
  <img src="./figures/figure_inference_time_and_accuracy_on_MNIST.png" width="656" height="371" alt="Inference Time vs Accuracy on MNIST" />
</div>

### Notes

- The `score` in the above table means test accuracy of MNIST dataset and the `inference time` means inference time for one image.
- Commonly used techniques like data normalization and dimension reduction using PCA are also used in the above analysis.
  See comments in the training script for details.
- The Score of RFF GPC is better than kernel SVC, moreover, the training/inference time of RFF GP is amazingly faster.

