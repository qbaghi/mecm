import numpy as np
from scipy import linalg as LA
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
pyfftw.interfaces.cache.enable()


def covariance_matrix_time(psd_func, fs, n_points):
    """

    Parameters
    ----------
    psd_func : callable
        function giving the 2-sided noise PSD (in A/Hz) as a function of frequency (in Hz)
    fs : scalar float
        sampling frequency
    n_points : scalar integer
        size of covariance matrix to display

    Returns
    -------
    cov : 2d numpy array
        covariance matrix in the time domain

    """

    freq = np.fft.fftfreq(2 * n_points) * fs
    autocorr = np.real(ifft(psd_func(freq))) * fs

    cov = LA.toeplitz(autocorr[0:n_points])

    return cov


def fourier_transform_matrix(n_points):

    k = np.array([np.arange(0, n_points)]).T
    n = np.array([np.arange(0, n_points)])

    tf_mat = np.exp(-2 * np.pi * np.dot(k, n) / n_points)

    return tf_mat






