Setting Up
====================================================================================================


Environment setup
----------------------------------------------------------------------------------------------------

### Using Docker (recommended)

If you don't like to pollute your development environment, it is a good idea to run everything
inside a Docker container. The rfflearn and it's sample code are executable on
[this docker image](https://hub.docker.com/repository/docker/tiskw/pytorch).
Please run the following command to download the docker image:

```console
docker pull tiskw/pytorch:latest
```

The following command is the typical usage of the docker image:

```
cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
```

If you need GPU support, add `--gpus all` option to the above `docker run` command above.
Also if the version of your docker is lower than 19, use `--runtime=nvidia` instead of `--gpus all`.

Also, if you want to use PyTorch 2.x, use `tiskw/pytorch2:latest` instead of `tiskw/pytorch:latest`.

### Installing on your environment (easier, but pollute your development environment)

If you don't mind to pollute your own environment (or you are already inside a docker container),
just run the following command for installing required packages:

```console
pip3 install -r requirements.txt
```


Installation
----------------------------------------------------------------------------------------------------

Copy the `rfflearn` directory to your `PYTHONPATH`, or register the parent directory of `rfflearn`
to your `PYTHONPATH` by, for example, using `sys.path` variable. If you need a concrete example,
please refer [the sample code](/examples) that use the latter apporach.


Quick Tutorial
----------------------------------------------------------------------------------------------------

At first, please clone the `random-fourier-features` repository from GitHub:

```console
git clone https://github.com/tiskw/random-fourier-features.git
cd random-fourier-features
```

If you are using the docker image, enter into the docker container by the following command
(not necessary to run thw following if you don't use docker):

```console
docker run --rm -it --gpus all -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
```

Then, launch python3 and try the following minimal code that runs support vector classification
with random Fourier features on an artificial tiny dataset.

```python
>>> import numpy as np                                  # Import Numpy
>>> import rfflearn.cpu as rfflearn                     # Import our module
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Define input data
>>> y = np.array([1, 1, 2, 2])                          # Defile label data
>>> svc = rfflearn.RFFSVC().fit(X, y)                   # Training
>>> svc.score(X, y)                                     # Inference (on CPU)
1.0
>>> svc.predict(np.array([[-0.8, -1]]))
array([1])
```


Next Step
----------------------------------------------------------------------------------------------------

Now you succeeded in installing the `rfflearn` module.
The author's recommendation for the next step is to see the [examples directory](/examples)
and try a code you are interested in.

