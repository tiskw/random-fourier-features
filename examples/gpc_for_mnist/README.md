# Gaussian process classifier using Random Fourier Features for MNIST dataset

This directory provides an example usage of the RFF Gaussian process classifier using MNIST dataset.

The training script in this directory supports both CPU/GPU training.
For the GPU training, you need to install Tensorflow 2.x (Tensorflow 1.x is not supported).


## Installation

See [this document](https://tiskw.gitbook.io/rfflearn/) for more details.

### Docker image (recommended)

If you don't like to pollute your development environment, it is a good idea to run everything inside a Docker container.
Scripts in this directory are executable on [this docker image](https://hub.docker.com/repository/docker/tiskw/tensorflow).

```console
$ docker pull tiskw/tensorflow:2020-05-29    # Download docker image from DockerHub
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO # Move to the root directory of this repository
$ docker run --rm -it --runtime=nvidia -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/tensorflow:2020-01-18 bash
$ cd examples/gpc_for_mnist/                 # Move to this directory
```

If you don't need GPU support, the option `--runtime=nvidia` is not necessary.

### Install Python packages (alternative)

The training end validation script requires `docopt`, `numpy`, `scipy`, `scikit-learn` and, if you need GPU support, `tensorflow-gpu`.
If you don't have them, please run the following as root to install them:

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install tensorflow-gpu                   # Required only for GPU inference
```


## Dataset Preparation

Also, you need to download and convert MNIST data before running the sample code by the following command:

```console
$ cd ../../dataset/mnist
$ python3 download_and_convert_mnist.py
```

Original MNIST data will be automatically downloaded, converted to `.npy` file,
and save them under `dataset/mnist/` directory.


## Training and inference

After generating `.npy` files of MNIST, run the training script by either of the following commands:

```console
$ python3 train_gpc_for_mnist.py cpu  # training on CPU
$ python3 train_gpc_for_mnist.py gpu  # training on GPU
```

The above command will show a test score, and generate `result.pickle` in which a trained model and command arguments are stored.

Default hyperparameter settings are recommended parameters for environment test.
However, to get a higher score, you need to change the parameters by command options.
See `--help` for details.


## Results of Support Vector Classification with RFF

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

## Notes

- The `score` in the above table means test accuracy of MNIST dataset and the `inference time` means inference time for one image.
- Commonly used techniques like data normalization and dimension reduction using PCA are also used in the above analysis.
  See comments in the training script for details.
- The Score of RFF GPC is better than kernel SVC, moreover, the training/inference time of RFF GP is amazingly faster.

