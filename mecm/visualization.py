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
    freq_pos = np.abs(freq)
    freq_pos[0] = freq_pos[1]
    autocorr = np.real(ifft(psd_func(freq_pos))) * fs

    cov = LA.toeplitz(autocorr[0:n_points])

    return cov


def fourier_transform_matrix(n_points):

    k = np.array([np.arange(0, n_points)]).T
    n = np.array([np.arange(0, n_points)])

    tf_mat = np.exp(-2 * np.pi * 1j * np.dot(k, n) / np.float(n_points))

    return tf_mat






