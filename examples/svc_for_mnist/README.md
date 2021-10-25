# Support Vector Classifier using Random Fourier Features for MNIST dataset

This directory provides an example of the support vector classifier with random Fourier features for MNIST dataset.

The training script in this directory supports both CPU/GPU training.
For the GPU training and inference, you need [PyTorch](https://pytorch.org/).


## Installation

See [this document](../..SETUP.md) for more details.

### Install on your environment (easier, but pollute your development environment)

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install torch                            # Required only for GPU training/inference
$ pip3 install optuna                           # Required only for hyper parameter tuning
```

### Docker image (recommended)

```console
$ docker pull tiskw/pytorch:latest
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
$ docker run --rm -it --gpus=all -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
$ cd examples/gpr_sparse_data/
```

If you don't need GPU support, the option `--gpus=all` is not necessary.


## Dataset preparation

You need to download and convert MNIST data before running the training code.
Please run the following commands:

```console
$ cd ../../dataset/mnist
$ python3 download_and_convert_mnist.py
```

The MNIST dataset will be automatically downloaded, converted to `.npy` file
and saved under `dataset/mnist/` directory.


## Training

After generating MNIST .npy files, run sample scripts by the following command:

```console
$ python3 train_rff_svc_for_mnist.py kernel           # Run kernel SVC training
$ python3 train_rff_svc_for_mnist.py cpu --rtype rff  # Run RFFSVC training on CPU
```

Default hyper parameter settings are recommended one, however, you can change the parameters by command option.
The above command will generate `result.pickle` in which a trained model, PCA matrix and command arguments are stored.
See `train_rff_svc_for_mnist.py --help` for details.

### Training on GPU

This module contains beta version of GPU training implementation.
You can try the GPU training by the following command:

```console
$ python3 train_rff_svc_for_mnist.py gpu --rtype rff  # Run kernel SVC training on GPU
```


## Inference

You can run inference by the following command:

```console
$ python3 valid_rff_svc_for_mnist.py cpu  # Inference on CPU using scikit-learn
$ python3 valid_rff_svc_for_mnist.py gpu  # Inference on GPU using PyTorch
```

Note that run inference of GPU-tranined model on CPU is not supported. Supported combinations are:
- trained on CPU, inference on CPU,
- trained on CPU, inference on GPU,
- trained on GPU, inference on GPU.

## Results of support vector classification with RFF

I've got the following results on my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB, GPU: GTX1050Ti):

| Method                   | Training time [sec] | Inference time [us] | Score [%] |
| :---------------------:  | :-----------------: | :-----------------: | :-------: |
| Kernel SVM               | 177.8 [sec]         | 4644.9 [us]         | 96.3 [%]  |
| SVM w/ RFF <br> d = 512  | 126.6 [sec]         |   39.0 [us]         | 96.5 [%]  |
| SVM w/ RFF <br> d = 1024 | 226.7 [sec]         |   96.1 [us]         | 97.5 [%]  |

As for inference using GPU, I've got the following result:

| Method     | Dimension of RFF | Device  | Batch size | Inference time (us) | Score [%] |
| :-------:  | :--------------: | :-----: | :---------:| :-----------------: | :-------: |
| SVM w/ RFF | 512              | CPU     | -          | 39.0 [us]           | 96.5 [%]  |
| SVM w/ RFF | 1024             | CPU     | -          | 96.1 [us]           | 97.5 [%]  |
| SVM w/ RFF | 1024             | TitanXp | 2,000      | 2.38 [us]           | 97.5 [%]  |

<div align="center">
  <img src="./figures/figure_Inference_Time_and_Accuracy_on_MNIST.png" width="600" height="371" alt="Inference Time vs Accuracy on MNIST" />
</div>

### Notes

- Score means test accuracy of MNIST dataset and inference time means inference time for one image.
- Now the GPU training does not show an enough high performance than the CPU training,
  but it's worth to try if you want faster training especially on higher RFF dimension than 1024 or huge training dataset.
- Commonly used techniques like data normalization and dimension reduction using PCA is also used in the above analysis.
  See comments in the Python script for details.
- The Score of RFF is slightly better than kernel SVM, moreover, the inference time of RFF is amazingly faster.
  On the other hand, the learning time of RFF can be longer than kernel SVM if the dimension of RFF is large.
- The following figure shows a tradeoff between the accuracy and inference time of RFF.

<div align="center">
  <img src="./figures/figure_rff_svc_for_mnist.png" width="480" height="320" alt="Accuracy for each dimention in RFF SVC" />
</div>

