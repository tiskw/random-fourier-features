Canonical Correlation Analysis using Random Fourier Features
====================================================================================================

This Python script provides examples of canonical correlation analysis with random Fourier features.


Installation
----------------------------------------------------------------------------------------------------

See [this document](../../SETUP.md) for more details.

### Docker image (recommended)

```console
docker pull tiskw/pytorch:latest
cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
cd examples/cca_for_artificial_data/
```

### Install on your environment (easier, but pollute your development environment)

```console
pip3 install docopt numpy scipy scikit-learn
```


Usage
----------------------------------------------------------------------------------------------------

```console
python3 main_cca_for_artificial_data.py
```

### Results of canonical correlation analysis with RFF

The input data X and Y have shape `(number_of_samples, dimension) = (500, 2)`,
and the data is composed of 2 parts, correlated and noise part.
As shown in the following figure, `X[:, 0]` and `Y[:, 0]` have strong correlation,
however, `X[:, 1]` and `Y[:, 1]` are completely independent.
The linear CCA failed to find the correlation, but CCA with random Fourier features succeeded
because of its nonlinearity.

<div align="center">
  <img src="./figure_cca_for_artificial_data.png" width="840" alt="CCA results for artificial dataset" />
</div>
