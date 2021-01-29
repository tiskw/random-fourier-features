# Princaipal Component Analysis using Random Fourier Features

This python script provides an example of PCA with Random Fourier Features (RFF PCA).
Our module for Random Fourier Features (PyRFFF.py) needs scikit-learn as a backend of PCA solver,
therefore you need to install scikit-learn.


## Installation

See [this document](https://tiskw.gitbook.io/rfflearn/tutorial#setting-up) for more details.

### Install on your environment (easier, but pollute your development environment)

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install torch                            # Required only for GPU training/inference
$ pip3 install optuna                           # Required only for hyper parameter tuning
```

### Docker image (recommended)

```console
$ docker pull tiskw/pytorch:2021-01-23
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
$ docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:2021-01-23 bash
$ cd examples/gpr_sparse_data/
```


## Usage

```console
$ python3 main_pca_for_swissroll.py rff     # RFF PCA
$ python3 main_pca_for_swissroll.py kernel  # Kernel PCA
```

### Results of princaipal component analysis using random Fourier features

The following figure shows the input data (10,000 points of swiss roll) and results of RFF PCA.

<div align="center">
  <img src="./figure_pca_for_swissroll_3d.png" width="640" height="480" alt="3D plot of input data (10,000 points of swiss roll)" />
  <img src="./figure_pca_for_swissroll_rffpca.png" width="640" height="480" alt="2D plot of 1st/2nd PC obtained by RFF PCA" />
</div>


### Computation time

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I've got the following results:

| Method                | Training time (sec) |
| :------------------:  | :-----------------: |
| Kernel PCA            | 6.01 sec            |
| RFF PCA <br> d = 1024 | 0.88 sec            |

### Training on GPU

Open the script file, replace `rfflearn.cpu` as `rfflean.gpu` and run the script again.
