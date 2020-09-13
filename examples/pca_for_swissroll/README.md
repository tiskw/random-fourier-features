# Princaipal Component Analysis using Random Fourier Features

This python script provides an example of PCA with Random Fourier Features (RFF PCA).
Our module for Random Fourier Features (PyRFFF.py) needs scikit-learn as a backend of PCA solver therefore you need to install scikit-learn.


## Usage

If you don't have scikit-learn or matplotlib, please run the following as root to install it:

```console
$ pip3 install scikit-learn matplorlib
```

You can run the example code by the following command:

```console
$ python3 main_pca_for_swissroll.py rff     # RFF PCA
$ python3 main_pca_for_swissroll.py kernel  # Kernel PCA
```


## Results of Regression with RFF

The following figure shows the input data (10,000 points of swiss roll) and results of RFF PCA.

<div align="center">
  <img src="./figure_pca_for_swissroll_3d.png" width="600" height="480" alt="3D plot of input data (10,000 points of swiss roll)" />
  <img src="./figure_pca_for_swissroll_rffpca.png" width="600" height="480" alt="2D plot of 1st/2nd PC obtained by RFF PCA" />
</div>


## Computation Time

In my computing environment (CPU: Intl Core i5 5250U, RAM: 4GB), I've got the following results:

| Method                | Training time (sec) |
| :------------------:  | :-----------------: |
| Kernel PCA            | 6.01 sec            |
| RFF PCA <br> d = 1024 | 0.88 sec            |

