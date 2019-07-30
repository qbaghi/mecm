# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# ==============================================================================
# This code provides algorithms to generate realizations of colored stationary
# processes
# ==============================================================================
import numpy as np
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
pyfftw.interfaces.cache.enable()


def symmetrize(values,N_DSP) :
    """
    Function returning the full symmetrized noise DSP in UNIT/sqrt(Hz)
    from DSP values ranging from fe/N_DSP to fe where :
    fe is the sampling frequency
    N_DSP is the data size (number of required DSP points)


    Parameters
    ----------
    values : 1D-array
        Positive frequencies DSP values (vector of size N_fft)
    N_DSP : scalar integer
        Number of required DSP points

    Returns
    -------
    DSP : numpy array
        Noise DSP (N vector)
    """


    # If N is even
    if (N_DSP % 2 == 0 ) :


        #N is odd :
        N_fft = np.int(N_DSP/2-1)

        # Frequencies
        f = np.fft.fftfreq(N_DSP)
        # Initialize the DSP
        DSP = np.zeros(N_DSP)

        DSP[1:N_fft+1] = values[0:N_fft]
        # For f=0 extrapolate the data
        p = np.polyfit(f[1:11],values[0:10],6)
        poly = np.poly1d(p)
        DSP[0] = poly(0)

        # Include the symmetric values for negative frequencies
        DSP[N_fft+2:N_DSP] = (DSP[1:N_fft+1])[::-1]

        # Extrapolate value for f = - fe/2 from the values f <= fe/2 -fe/N
        p2 = np.polyfit(f[N_fft-9:N_fft],values[N_fft-9:N_fft],2)
        poly2 = np.poly1d(p2)
        DSP[N_fft+1] = poly2(1./2.)


    else :


        N_fft = np.int((N_DSP-1)/2)


        # Frequencies
        f = np.fft.fftfreq(N_DSP)

        # Initialize the DSP
        DSP = np.zeros(N_DSP)
        DSP[1:N_fft+1] = values[0:N_fft]
        # For f=0  extrapolate the data
        p = np.polyfit(f[1:11],values[0:10],6)
        poly = np.poly1d(p)
        DSP[0] = poly(0)
        # Include the symmetric values for negative frequencies
        DSP[N_fft+1:N_DSP] = (DSP[1:N_fft+1])[::-1]

    return DSP



def symmetrize_shift(values,N,fe) :
    """
    Function returning the full symmetrized noise DSP in UNIT/sqrt(Hz)
    from DSP values ranging from fe/N to (N_fft+1)*fe/N where :
    fe is the sampling frequency
    N is the data size (number of required temporal observations)
    N_fft is (N-1)/2 if N is odd and N/2-1 if N is even

    This function differs from symmetrize function in that it "shifts" the real DSP
    by -fe/N so that the zero frequency point actually corresponds to DSP(fe/N).
    This is another way to handle the singularity at zero.

    Parameters
    ----------
    DSPvalues : 1D-array
        Positifve frequencies DSP values (vector of size N_fft)

    Returns
    -------
    DSP : numpy array
        Noise DSP (N vector)
    """
    # If N is even
    if (N % 2 == 0 ) :


        #N is odd :
        N_fft = N/2-1

        # Initialize the DSP
        DSP = np.zeros(N)

        DSP[0:N_fft+1] = values[0:N_fft+1]

        # Include the symmetric values for negative frequencies
        DSP[N_fft+2:N] = (DSP[1:N_fft+1])[::-1]

        # Extrapolate value for f = - fe/2 from the value at fe/2
        DSP[N_fft+1] = values[N_fft+1]


    else :


        N_fft = (N-1)/2

        # Initialize the DSP
        DSP = np.zeros(N)
        DSP[0:N_fft+1] = values[0:N_fft+1]

        # Include the symmetric values for negative frequencies
        DSP[N_fft+1:N] = (DSP[1:N_fft+1])[::-1]

    return DSP


def generateFreqNoiseFromDSP(DSP, fe, myseed=None):
    """
    Generate noise in the frequency domain from the values of the DSP. 
    """
 
    
    """
    Function generating a colored noise from a vector containing the DSP.
    The DSP contains Np points such that Np > 2N and the output noise should
    only contain N points in order to avoid boundary effects. However, the
    output is a 2N vector containing all the generated data. The troncature
    should be done afterwards.

    References : Timmer & König, "On generating power law noise", 1995

    Parameters
    ----------
    DSP : array_like
        vector of size N_DSP continaing the noise DSP calculated at frequencies
        between -fe/N_DSP and fe/N_DSP where fe is the sampling frequency and N
        is the size of the time series (it will be the size of the returned
        temporal noise vector b)
    N : scalar integer
        Size of the output time series
    fe : scalar float
        sampling frequency
    myseed : scalar integer or None
        seed of the random number generator

    Returns
    -------
        bf : numpy array
        frequency sample of the colored noise (size N)
    """

    # Size of the DSP
    N_DSP = len(DSP)
    # Initialize seed for generating random numbers
    np.random.seed(myseed)

    N_fft = np.int((N_DSP-1)/2)
    # Real part of the Noise fft : it is a gaussian random variable
    Noise_TF_real = np.sqrt(0.5)*DSP[0:N_fft+1]*np.random.normal(loc=0.0, scale=1.0, size=N_fft+1) #[random.gauss(0,1.) for _ in range(N_fft+1)]
    # Imaginary part of the Noise fft :
    Noise_TF_im = np.sqrt(0.5)*DSP[0:N_fft+1]*np.random.normal(loc=0.0, scale=1.0, size=N_fft+1)#*[random.gauss(0,1.) for _ in range(N_fft+1)]
    # The Fourier transform must be real in f = 0
    Noise_TF_im[0] = 0.
    Noise_TF_real[0] = Noise_TF_real[0]*np.sqrt(2.)

    # Create the NoiseTF complex numbers for positive frequencies
    Noise_TF = Noise_TF_real + 1j*Noise_TF_im

    # To get a real valued signal we must have NoiseTF(-f) = NoiseTF*
    if N_DSP % 2 == 0 :
        # The TF at Nyquist frequency must be real in the case of an even number of data
        Noise_sym0 = np.array([ DSP[N_fft+1]*np.random.normal(0,1) ])
        # Add the symmetric part corresponding to negative frequencies
        Noise_TF = np.hstack( (Noise_TF, Noise_sym0, np.conj(Noise_TF[1:N_fft+1])[::-1]) )

    else :

        # Noise_TF = np.hstack( (Noise_TF, Noise_sym[::-1]) )
        Noise_TF = np.hstack( (Noise_TF, np.conj(Noise_TF[1:N_fft+1])[::-1]) )    
        
    return np.sqrt(N_DSP*fe/2.) * Noise_TF
    
    

def generateNoiseFromDSP(DSP, fe, myseed=None) :
    """
    Function generating a colored noise from a vector containing the DSP.
    The DSP contains Np points such that Np > 2N and the output noise should
    only contain N points in order to avoid boundary effects. However, the
    output is a 2N vector containing all the generated data. The troncature
    should be done afterwards.

    References : Timmer & König, "On generating power law noise", 1995

    Parameters
    ----------
    DSP : array_like
        vector of size N_DSP continaing the noise DSP calculated at frequencies
        between -fe/N_DSP and fe/N_DSP where fe is the sampling frequency and N
        is the size of the time series (it will be the size of the returned
        temporal noise vector b)
    N : scalar integer
        Size of the output time series
    fe : scalar float
        sampling frequency
    myseed : scalar integer or None
        seed of the random number generator

    Returns
    -------
        b : numpy array
        time sample of the colored noise (size N)
    """


#    # Inverse Fourier transform to get the noise time series (and apply the right normalization)
#    b = ifft(Noise_TF)*np.sqrt(N_DSP*fe/2.) # One must multiply by fe (to get the right dimension) and divide by 2 because of symmetrization !
#                                    # otherwise you say that you have both an uncertainty on positive and negative frequencies values
#                                    # which is wrong  because we know that they are equal.

    # Noise spectrum :
    #S = fe/2.*DSP**2

    #return b[N_DSP/2:N_DSP/2+N]*np.sqrt(np.var(b)/np.var(b[N_DSP/2:N_DSP/2+N])),S
    return ifft(generateFreqNoiseFromDSP(DSP,fe,myseed = myseed))#,S,Noise_TF#*np.sqrt(np.var(b)/np.var(b[0:N])),S



