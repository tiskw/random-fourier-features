Gaussian Process Classifier using Random Fourier Features for MNIST Dataset
====================================================================================================

This directory provides an example of the Gaussian process classifier with random Fourier features
for the MNIST dataset.

The training script in this directory supports both CPU/GPU training.
For the GPU training and inference, you need [PyTorch](https://pytorch.org/).


Installation
----------------------------------------------------------------------------------------------------

See [this document](../../SETUP.md) for more details.

### Docker image (recommended)

```console
docker pull tiskw/pytorch:latest
cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
docker run --rm -it --gpus=all -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
cd examples/gpc_for_mnist/
```

If you need GPU support, add the option `--gpus=all` to the above `docker run` command.

### Install on your environment (easier, but pollute your development environment)

```console
pip3 install docopt numpy scipy scikit-learn  # Necessary packages
pip3 install torch                            # Required only for GPU training/inference
```


Usage
----------------------------------------------------------------------------------------------------

### Dataset preparation

You need to download and convert MNIST data before running the training code.
Please run the following commands:

```console
cd ../../dataset/mnist
python3 download_and_convert_mnist.py
```

The MNIST dataset will be automatically downloaded, converted to `.npy` file,
and saved under the `dataset/mnist/` directory.

### Training and inference

After generating `.npy` files of MNIST, run the training script by either of the following commands:

```console
python3 train_gpc_for_mnist.py cpu  # training on CPU
python3 train_gpc_for_mnist.py gpu  # training on GPU
```

The above command will show a test score, and generate `result.pickle` in which a trained model
and command arguments are stored.

Default hyperparameter settings are recommended parameters for environment test.
However, to get a higher score, you need to change the parameters by command options.
See `--help` for details.

### Results of Gaussian process classification with RFF

In the author's computing environment (CPU: Intel Core i5-9300H, RAM: 32GB, GPU: GeForce GTX1660Ti),
the author has got the following results:

| Method     | RFF dim | Device    | Training time (sec) | Inference time (us) | Score (%) | std_kernel | std_error |
|:----------:|:-------:|:---------:|:-------------------:|:-------------------:|:---------:|:----------:|:---------:|
| Kernel SVC | -       | CPU       |  47.63 sec          | 1312.6 us           | 96.30 %   | -          | -         |
| GPC w/ RFF | 1536    | CPU       |   3.76 sec          |   49.3 us           | 96.37 %   | 0.1        | 0.5       |
| GPC w/ RFF | 1536    | GTX1660Ti |      -              |   7.46 us           | 96.37 %   | 0.1        | 0.5       |
| GPC w/ RFF | 10000   | CPU       |  96.95 sec          |  269.3 us           | 98.24 %   | 0.1        | 0.5       |
| GPC w/ RFF | 10000   | GTX1660Ti |      -              |   33.6 us           | 98.24 %   | 0.1        | 0.5       |
| GPC w/ RFF | 20000   | CPU       | 516.19 sec          |  517.9 us           | 98.38 %   | 0.1        | 0.5       |
| GPC w/ RFF | 20000   | GTX1660Ti |      -              |   61.5 us           | 98.38 %   | 0.1        | 0.5       |

<div align="center">
  <img src="./figures/Inference_time_and_acc_on_MNIST_gpc.svg" width="640" alt="Inference Time vs Accuracy on MNIST" />
</div>

### Notes

- The `score` in the above table means the test accuracy of the MNIST dataset and the `inference time`
  means the inference time for one image.
- Commonly used techniques like data normalization and dimension reduction using PCA are also used
  in the above analysis. See comments in the training script for details.
- The Score of RFF GPC is better than kernel SVC, moreover, the training/inference time of RFF GP
  is amazingly faster.

