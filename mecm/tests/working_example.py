import mecm
from mecm import localestimator, psd
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from pyfftw.interfaces.numpy_fft import fft
import time


if __name__ == '__main__':

    # Fix the seed
    np.random.seed(12354)
    # Choose size of data
    n_data = 2**14
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=n_data)
    # Apply filtering to turn it into colored noise
    r = 0.01
    b, a = signal.butter(3, 0.1/0.5, btype='high', analog=False)
    n = signal.lfilter(b, a, noise, axis=-1, zi=None) + noise*r
    # Generate periodic deterministic signal
    t = np.arange(0, n_data)
    f0 = 1e-2
    a0 = 5e-3
    s = a0 * np.sin(2 * np.pi * f0 * t)
    # Create a mask vector indicating missing data points
    mask = np.ones(n_data)
    n_gaps = 30
    gapstarts = (n_data*np.random.random(n_gaps)).astype(int)
    gaplength = 10
    gapends = (gapstarts+gaplength).astype(int)
    for k in range(n_gaps):
        mask[gapstarts[k]:gapends[k]] = 0
    # Create the masked data vector
    y = mask*(s+n)
    # Create the design matrix for the leastsquares estimation
    a_mat = np.array([np.sin(2*np.pi*f0*t)]).T
    # Specify noise PSD model
    f_knots = np.array([1e-3, 1e-2, 2e-2, 3e-2, 1e-1, 2e-1])
    psd_cls = psd.PSDSpline(n_data, 1.0, f_knots=f_knots, n_knots=6, d=3)
    # psd_cls = localestimator.PSD_estimate(100, n_data, 2*n_data,
    #                                       kind='linear')
    # Run the MECM algoritm to perform a joint estimation of the sine amplitude
    # and noise PSD
    t1 = time.time()
    res = mecm.maxlike(y, mask, a_mat, psd_cls=psd_cls, pcg_algo='scipy')
    t2 = time.time()
    a0_est, a0_cov, a0_vect, y_rec, p_cond_mean, psd_cls, success, diff = res
    print("Computation took " + str(t2 - t1) + " seconds.")
    # Plot the results
    f = np.fft.fftfreq(n_data)
    wN, H = signal.freqz(b, a, worN=f[f > 0] * (2 * np.pi), whole=False)
    s_th = np.abs(r+H)**2
    s_n = psd_cls.calculate(n_data)
    w = np.hanning(n_data)
    s_est = np.dot(a_mat, a0_est)

    true_signal = np.abs(fft(w*s))*2./np.sum(w**2)
    estimated_signal = np.abs(fft(w*s_est))*2./np.sum(w**2)
    complete_data = np.abs(fft(w * (s+n)))/np.sqrt(np.sum(w**2))
    masked_data = np.abs(fft(w * y))/np.sqrt(np.sum(mask*w**2))
    reconstructed_data = np.abs(fft(w*(y_rec)))/np.sqrt(np.sum(w**2))

    plt.figure("Signal estimation (time domain)")
    plt.plot(t, s, 'k', label='True signal')
    plt.plot(t, s_est, 'b', label='Estimated signal')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.draw()
    plt.figure("Signal estimation (frequency domain)")
    plt.loglog(f, true_signal, 'k', label='True signal')
    plt.loglog(f, estimated_signal, 'b', label='Estimated signal')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.draw()
    plt.figure("Noise estimation (frequency domain)")
    plt.loglog(f, complete_data, 'k', label='Complete data')
    plt.loglog(f, masked_data, '0.75', label='Masked data')
    plt.loglog(f, reconstructed_data, 'b', label='Reconstructed data')
    plt.loglog(f[f > 0], np.sqrt(s_th), 'g', label='True noise PSD')
    plt.loglog(f[f > 0], np.sqrt(s_n[f > 0]), 'r--', label='Estimated PSD')
    plt.xlabel("Frequency")
    plt.ylabel("PSD")
    plt.draw()

    plt.show()
