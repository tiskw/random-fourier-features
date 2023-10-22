# Documents

### Ask for help!!

The author would like to register the document in this directory to [arXiv](https://arxiv.org/),
however, I need to be endorsed by an endorser of stat.ML subject class on arXiv
because it is my first time posting an article on arXiv.
Please contact me if you are an endorser of stat.ML and interested in one of the articles in this directory.
Thanks!

## Document 1: Random Fourier Features for Gaussian Process Model

### Abstract

This article describes the procedure for applying random Fourier features [1] to Gaussian process model [2].
This makes it possible to speed up the training and inference of the Gaussian process model,
and to apply the model to large-scale data.

Gaussian process model [2] is one of the supervised machine learning frameworks
designed on a probability space, and is widely used for regression and classification tasks, like
support vector machine and random forest. The major difference between Gaussian process model and
other machine learning models is that Gaussian process model is a *stochastic* model.
In other words, since the Gaussian process model is formulated as a stochastic model,
it can provide not only the predicted value but also a measure of uncertainty for the prediction.
This is a very useful property that can improve the explainability of machine learning model.

On the other hand, Gaussian process model is also known for its high computational cost
of training and inrefence. If the total number of training data is $N \in \mathbb{Z}^{+}$,
the computational cost required for training Gaussian process model is $O(N^3)$, and
computational cost required for inference is $O(N^2)$, where $O$ is \textit{Bachmann–Landau notation}.
The problem is that the computational cost is given by a power of the total number of training data $N$,
which can be an obstacle when appliying the model to large-scale data.
This comes from the fact that Gaussian process model has the same mathematical structure as
the kernel method, in other words, the kernel support vector machine also has the same problem.

One of the methods to speed up the kernel method is random Fourier features [1] (hereinafter abbreviated as RFF).
This method can significantly reduces the computational cost while keeping the flexibility of the kernel method
by approximating the kernel function as the inner product of finite dimensional vectors.
Specifically, the compurational cost required for training can be reduced to $O(N D^2)$, and the amount of calculation
required for inference can be reduced to $O(D^2)$, where $D \in \mathbb{Z}^{+}$ is
a hyperparameter of RFF and can be specified independently of the total number of training data $N$.
Since Gaussian process model has the same mathematical structure as the kernel method,
RFF can be applied to Gaussian process model as well. This evolves Gaussian process model
into a more powerful, easy-to-use, and highly reliable ML tool.

However, when applying RFF to Gaussian process model, some mathematical techniques are required
that are not straightforward. Unfortunately, there seems to be no articles in the world
that mentions it's difficulties and solutions, so I decided to leave this document.

### References

[1] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines", Neural Information Processing Systems, 2007. <br/>
[2] C. Rasmussen and C. Williams, ``Gaussian Processes for Machine Learning'', MIT Press, 2006.
