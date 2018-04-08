mecm
=================



The MECM python package provides an implementation of the modified expectation maximization
algorithm developed by Q. Baghi et al., for which a reference can be found in
https://arxiv.org/abs/1608.08530



Short Description
-----------------

The MECM package is a tool to perform gaussian linear regressions on time series affected
by colored stationary noise and missing data. It is based on an algorithm close to the
expectation-maximization, which is efficiently implemented by taking advantage of fast
Fourier transforms, fast matrix-to-vector multiplications, and sparse linear algebra.

Let's consider a data model that can be written on the form

.. math::

  y = A \beta + n

where:

  * y is the measured time series data (size N), evenly sampled.

  * A is the design matrix (size N x K)

  * beta is the vector of parameters to estimate (size K)

  * n is the noise vector, assumed to follow a Gaussian stationary distribution with a given smooth spectral density S(f)

Now assume that only some entries of the vector y are observed. The indices of
observed and missing data are provided by a binary mask vector M, whose entries
are equal to 1 when data are observed, 0 otherwise.
So in fact we observe only a vector y_obs such that

.. code-block::

  y_obs = y[M==1]

The mecm package implements a method to estimate $\beta$ and $S(f)$ given y_obs,
A and M.


The main methods of the package are:

  * maxlike: quasi-maximum likelihood estimation with missing data for gaussian stationary models of the form $y = A*beta + epsilon$

  * PSD_estimate: a class to perform power spectral density estimation with local linear smoothers.

  * conditionalDraw: a function computing the conditional expectation of the missing data conditionally on the observed data, assuming a Gaussian stationary model.




Required Packages
-----------------

Prior to installation make sure that the following python packages are already installed:

* NumPy: http://www.numpy.org/

* SciPy: https://www.scipy.org/

* pyFFTW: https://pypi.python.org/pypi/pyFFTW

* cvxopt: http://cvxopt.org/

* numba: https://numba.pydata.org/



Installation
------------

mecm can be installed by unzipping the source code in one directory, then open up a terminal (or execute a CMD on Windows) and using this command: ::

    sudo python setup.py install

You can also install it directly from the Python Package Index with this command: ::

    sudo pip mecm install





Licence
-------

See file:

.. include:: ../../LICENCE.txt


Quick start guide
-----------------

MECM can be basically used to perform any multilinear regression analysis where
the distribution of the noise is assumed to be Gaussian and stationary in the
wide sense, with a smooth power spectral density (PSD).

Let us show how it works with an example.

1. Data generation

To begin with, we generate some simple time series which contains noise and signal.
To generate the noise, we start with a white, zero-mean Gaussian noise that
we then filter to obtain a stationary colored noise:

.. code-block::

  # Import mecm and other useful packages
  import mecm
  import numpy as np
  import random
  from scipy import signal
  # Choose size of data
  N = 2**14
  # Generate Gaussian white noise
  noise = np.random.normal(loc=0.0, scale=1.0, size = N)
  # Apply filtering to turn it into colored noise
  r = 0.01
  b, a = signal.butter(3, 0.1/0.5, btype='high', analog=False)
  n = signal.lfilter(b,a, noise, axis=-1, zi=None) + noise*r

Then we need a deterministic signal to add. We choose a sinusoid with some
frequency f0 and amplitude a0:

.. code-block::

  t = np.arange(0,N)
  f0 = 1e-2
  a0 = 5e-3
  s = a0*np.sin(2*np.pi*f0*t)

We just have generated a time series that can be written in the form

.. math::

  y = A \beta + n

Now assume that some data are missing, i.e. the time series is cut by random gaps.
The pattern is represented by a mask vector M with entries equal to 1 when data
is observed, and 0 otherwise:

.. code-block::

  M = np.ones(N)
  Ngaps = 30
  gapstarts = (N*np.random.random(Ngaps)).astype(int)
  gaplength = 10
  gapends = (gapstarts+gaplength).astype(int)
  for k in range(Ngaps): M[gapstarts[k]:gapends[k]]= 0

Therefore, we do not observe y but its masked version, M*y.

2. Linear regression

Now let's assume that we observed M*y and that we want to estimate the amplitude
of the sine wave whose frequency and phase are known, along with the PSD of the
noise residuals.
The available data is

.. code-block::

  y = M*(s+n)

We must specify the design matrix (i.e. the data model) by:

.. code-block::

  A = np.array([np.sin(2*np.pi*f0*t)]).T

Then we can just run the mecm maximum likelihood estimator, by writing:

.. code-block::

  a0_est,a0_cov,a0_vect,y_rec,I_condMean,PSD = mecm.maxlike(y,M,A)

The result of this function is, in the order provided: the estimated amplitude,
its estimated covariance, the vector containing the amplitude updates at each
iteration of the algorithm, the estimated complete-data vector, the conditional
expectation of the data periodogram (at Fourier frequencies), and an instance of
the PSD_estimate class.

Documentation
-------------

For a more detailed description of the outputs and information about how to tune
the mecm algorithm, please have a look at the documentation:

.. include:: ../mecm/LICENCE.txt 

Contribute
----------
mecm is an open-source software. Everyone is welcome to contribute !
Please site the original paper in scientific contributions:
https://arxiv.org/abs/1608.08530
