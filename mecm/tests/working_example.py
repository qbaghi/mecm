import mecm
import numpy as np
import random
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from pyfftw.interfaces.numpy_fft import fft, ifft

if __name__ == '__main__':

    # Choose size of data
    N = 2**14
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size = N)
    # Apply filtering to turn it into colored noise
    r = 0.01
    b, a = signal.butter(3, 0.1/0.5, btype='high', analog=False)
    n = signal.lfilter(b,a, noise, axis=-1, zi=None) + noise*r
    # Generate periodic deterministic signal
    t = np.arange(0,N)
    f0 = 1e-2
    a0 = 5e-3
    s = a0*np.sin(2*np.pi*f0*t)
    # Create a mask vector indicating missing data points
    M = np.ones(N)
    Ngaps = 30
    gapstarts = (N*np.random.random(Ngaps)).astype(int)
    gaplength = 10
    gapends = (gapstarts+gaplength).astype(int)
    for k in range(Ngaps): M[gapstarts[k]:gapends[k]]= 0
    # Create the masked data vector
    y = M*(s+n)
    # Create the design matrix for the leastsquares estimation
    A = np.array([np.sin(2*np.pi*f0*t)]).T
    # Run the MECM algoritm to perform a joint estimation of the sine amplitude a0
    # and noise PSD
    a0_est,a0_cov,a0_vect,y_rec,I_condMean,PSD = mecm.maxlike(y,M,A)
    # Plot the results
    f = np.fft.fftfreq(N)
    wN,H = signal.freqz(b, a, worN=f[f>0]*(2*np.pi), whole=False)
    S_th = np.abs(r+H)**2
    S_est = PSD.calculate(N)
    w = np.hanning(N)
    s_est = np.dot(A,a0_est)
    plt.figure("Signal estimation (time domain)")
    plt.plot(t,s,'k',label = 'True signal')
    plt.plot(t,s_est,'b',label = 'Estimated signal')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.draw()
    plt.figure("Signal estimation (frequency domain)")
    plt.loglog(f,np.abs(fft(w*s))*2./np.sum(w**2),'k',label = 'True signal')
    plt.loglog(f,np.abs(fft(w*s_est))*2./np.sum(w**2),'b',label = 'Estimated signal')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.draw()
    plt.figure("Noise estimation (frequency domain)")
    plt.loglog(f,np.abs(fft(w*(s+n)))/np.sqrt(np.sum(w**2)),'k',label = 'Complete data')
    plt.loglog(f,np.abs(fft(w*y))/np.sqrt(np.sum(M*w**2)),'0.75',label = 'Masked data')
    plt.loglog(f,np.abs(fft(w*(y_rec)))/np.sqrt(np.sum(w**2)),'b',label = 'Periodogram of \
    estimated complete data')
    plt.loglog(f[f>0],np.sqrt(S_th),'g',label = 'True noise PSD')
    plt.loglog(f[f>0],np.sqrt(S_est)[f>0],'r--',label = 'Estimated noise PSD')
    plt.xlabel("Frequency")
    plt.ylabel("PSD")
    plt.draw()

    plt.show()
