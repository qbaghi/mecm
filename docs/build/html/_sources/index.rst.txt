.. mecm documentation master file, created by
   sphinx-quickstart on Fri Apr  6 15:50:52 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mecm's documentation!
================================

MECM stands for Modified Expectation Conditional Maximization. It is an
method to estimate model parameters through linear regression, from time
series affected by stationary colored noise and missing data.
System inversions are efficiently performed by using preconditioned conjugate
gradients, where matrix-to vector products are efficiently computed using
FFT algorithm and element-wise multiplications.

This package provides an implementation of the MECM algorithm as described in
the reference:

https://arxiv.org/abs/1608.08530


.. toctree::
   :maxdepth: 2
   :caption: Contents:




What this package does
======================

The regression problem that this package tackle is the following.
Let's consider a data model that can be written on the form

.. math::

 y = A \beta + n

where:

 * y is the measured time series data (size N), evenly sampled.

 * A is the design matrix (size N x K)

 * :math:`\beta` is the vector of parameters to estimate (size K)

 * n is the noise vector, assumed to follow a Gaussian stationary distribution with a given smooth spectral density S(f)

Now assume that only some entries of the vector y are observed. The indices of
observed and missing data are provided by a binary mask vector M, whose entries
are equal to 1 when data are observed, 0 otherwise.
So in fact we observe only a vector y_obs such that

.. code-block:: python

  y_obs = y[M==1]

The mecm package implements a method to estimate :math:`\beta` and :math:`S(f)` given y_obs,
A and M.



Installation
============

mecm can be installed by unzipping the source code in one directory, open up a terminal and using this command: ::

   sudo python setup.py install

You can also install it directly from the Python Package Index with this command: ::

   sudo pip mecm install


.. _quick-start-label:

Quick start guide
=================

MECM can be basically used to perform any multilinear regression analysis where
the distribution of the noise is assumed to be Gaussian and stationary in the
wide sense, with a smooth power spectral density (PSD).

Let us show how it works with an example.

1. Data generation

To begin with, we generate some simple time series which contains noise and signal.
To generate the noise, we start with a white, zero-mean Gaussian noise that
we then filter to obtain a stationary colored noise:

.. code-block:: python

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

.. code-block:: python

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

.. code-block:: python

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

.. code-block:: python

 y = M*(s+n)

We must specify the design matrix (i.e. the data model) by:

.. code-block:: python

 A = np.array([np.sin(2*np.pi*f0*t)]).T

Then we can just run the mecm maximum likelihood estimator, by writing:

.. code-block:: python

  a0_est,a0_cov,a0_vect,y_rec,I_condMean,PSD = mecm.maxlike(y,M,A)

The result of this function is, in the order provided: the estimated amplitude,
its estimated covariance, the vector containing the amplitude updates at each
iteration of the algorithm, the estimated complete-data vector, the conditional
expectation of the data periodogram (at Fourier frequencies), and an instance of
the PSD_estimate class.



MECM module
===========

The mecm module is the core method of the package. It allows one to perform
efficient linear regression on time series with stationary noise and missing
data.

Maxlike function
----------------

The main function that you may use is the maximum likelihood algorithm provided
by maxlike, which computes the maximum likelihood estimate of the regression
parameter :math:`\beta`.

This function can be used simply as in section :ref:`quick-start-label`.
However, it can be useful to tune some additional parameters to increase
accuracy of the results or the efficiency of the computation.

See the full inputs and outputs of the function as well as more details on how
to specify its parameters below:

.. autofunction:: mecm.maxlike

Here we explain in more details the meaning and effects of the optional
arguments of the maxlike function.

The MECM algorithm iterates between two basic steps:

    1. expectation steps: estimation the missing data and their second
    orders moments) and

    2. maximization steps: estimation of the regression parameter beta and
    of the noise PSD function S(f).

The number of iterations of the algorithm is driven by N_it_max, which is
the maximum number of iterations, and eps, which is the tolerance criterium
below which the algorithms stops, i.e. if

.. math::

    || \beta_{i} - \beta_{i-1}|| / || \beta_{i-1} || < \epsilon

The expectation step 1 involves the estimation of the missing data vector, which
can be written as

.. math::

  y_m = A_m \beta + C_{mo} C_{oo}^{-1} \left( y_o - A_o \beta \right)

The storage and full computation of the covariance matrix :math:`C_{oo}` is not
feasible on standard machines.
Rather, we solve the system:

.. math::

  C_{oo} x = y_o

by using a preconditioned conjugate gradient (PCG) algorithm, which iteratively
decrease the norm of the residuals :math:`C_{oo}x - y_o`.
The maximum number of iterations and convergence tolerence of the PCG algorithm
are respectively given by the parameters N_it_cg and tol_cg.
This convergence criterium is met when

.. math::

  ||C_{oo} x - y_o || /|| y_o || < \rm{tol}_{cg}

However, for the convergence to be fast enough, and the problem to be well-posed,
one needs to solve the following system instead:

.. math::

  P C_{oo} x = P y_o

where :math:`P` is a matrix which looks like :math:`C_{oo}` but which is easier
to compute.
In the MECM algorithm, we choose :math:`P^{-1}` as being a sparse version of
:math:`C_{oo}`. That is, the autocovariance function of the noise is approximated
by a truncated version of the true noise autocovariance. The autocorrelation
length of this approximate matrix is specified by the parameter p.
**The choice of p plays an important role in the algorithm: if p is large,
the convergence will be more stable, and the number of iterations needed to
reach convergence will we smaller, but the computational cost to perform a single iteration will be higher,
and so will be the amount of memory needed to compute the matrix P**, which scales as
:math:`p^2 N` where :math:`N` is the size of the analyzed time series. Therefore,
the parameter p must be carefully tuned if you deal with large data sets (i.e. :math:`N>10^{6}`),
in order to prevent memory overflow.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
