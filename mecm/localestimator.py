# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# ==============================================================================
# This code provides algorithms for estimating pronbability and sptrum densities
# by using local linear smoothing
# ==============================================================================
import numpy as np
from scipy import interpolate
from scipy import stats
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
pyfftw.interfaces.cache.enable()
from .noise import symmetrize
#from numba import jit

# ==============================================================================
# Kernels
# ==============================================================================
def nextpow2(N):
    """
    Gives the lowest integer q such that y = <2^q (next power of 2)

    Parameters
    ----------
    y : scalar integer
        any positive number (usually the size of the data set)

    Returns
    -------
    q : scalar integer
        the lowest integer q such 2^q is greater or equal than the input number
    """

    return int( np.log(N)/np.log(2.) ) + 1


def gaussianKernel(y):
    """
    Gaussian smoothing kernel function.

    Parameters
    ----------
    y : array_like
        input abscissa (size N)

    Returns
    -------
    ker : array_like (size N)
        value of the kernel function at y
    """
    return 1./np.sqrt(2*np.pi) * np.exp(-(y**2)/2.)


def epanechnikovKernel(y):
    """
    Epanechnikov smoothing kernel function.

    Parameters
    ----------
    y : array_like
        input abscissa (size N)

    Returns
    -------
    ker : array_like (size N)
        value of the kernel function at y
    """
    out = np.zeros(len(y))
    out[y<=1.] = 3./4.*(1-y[y<=1.] **2)

    return out


def kernel(y,kind):
    """
    Smoothing kernel function.

    Parameters
    ----------
    y : array_like
        input data vector to smooth (size N)
    kind : {'epa','ker'}
        Type of smoothing kernel

    Returns
    -------
    ker(y) : array_like (size N)
        value of the kernel function at y

    """


    if kind == 'epa':
        return epanechnikovKernel(y)
    elif kind == 'gauss' :
        return gaussianKernel(y)


# ==============================================================================
# The basic linear smoother
# ==============================================================================
def localLinearSmoother(data,f,fj,h,ker='epa',variance=False,ST_given = None):
    """
    Function computing the local linear estimate of the input data at points
    fj, given that the input data are available at points f.


    Parameters
    ----------
    data : array_like
        Input data array (size N)
    f : array_like
        Abscissa correponding to the input data (size N)
    fj : array_like
        Abscissa correponding to the output estimate (size J)
    h : array_like
        smoothing parameter vector (size J)
    ker : {'epa','ker'}, optional
        Type of smoothing kernel
    variance : boolean, optional
        If True the estiamated variance of the local linear estimate is provided
        as an output


    Returns
    -------
    m_est : array_like
        Output estimated smooth function (intersect points, size J)
    b_est : array_like
        Output estimated slopes (size J)
    V : array_like
        normalized variance of the estimate (size J)
    sigma2_0 : scalar float
        normalized weighted residual sum of squares
        V * pi^2/6
    ST : array_like
        kernel-dependant quantity that does not depend on the data (may be
        useful to perform several calculations with the same kernel and data size)

    """
    # Length of the points at wich to estimate the unknown smooth function
    J = len(fj)

    # Initialization of the log-PSD estimate
    m_est = np.zeros( J ) + 1j*np.zeros( J )
    # Initialization of the bias estimate, which is also the local slope of the
    # log-PSD
    b_est = np.zeros( J ) + 1j*np.zeros( J )

    # For variance calculation
    ST_star = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )
    V = np.zeros( J ) + 1j*np.zeros( J )
    sigma2_0 = np.zeros( J ) + 1j*np.zeros( J )

    if ST_given is None:
        ST = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )

    for j in range(J):

        # b is the range over which the kernel is significantly larger than zero
        if ker == 'epa' :
            b = h[j]
        elif ker == 'gauss':
            b = 10*h[j]

        # Difference between periodogram frequency and frequency at which
        # estimate the PSD
        df = f - fj[j]
        # Restrict the calculation to the range where the kernel is significant
        ii = np.where( np.abs( df ) <= b )[0]

        df1 = df[ii]
        df2 = df[ii]**2

        # Caluclate K((fj-f0)/h)
        K = kernel( df[ii]/h[j] , ker )
        K2 = K**2

        if ST_given is None :
            ST[j,0] = np.sum(  K  )
            ST[j,1] = np.sum(  K*df1  )
            ST[j,2] = np.sum(  K*df2  )
        else:
            ST = ST_given[:]

        KY = np.sum(  K*data[ii] )
        XKY = np.sum( K*data[ii]*df1 )

        denom = ST[j,0]*ST[j,2]-ST[j,1]**2

        # Estimate of the local intercept a and the slope b
        m_est[j] = ( ST[j,2]*KY - ST[j,1]*XKY )/denom
        b_est[j] = ( -ST[j,1]*KY + ST[j,0]*XKY )/denom

        # If variance calculation is required, compute the necessary quantities
        if variance == True:

            ST_star[j,0] = np.sum(  K2  )
            ST_star[j,1] = np.sum(  K2*df1  )
            ST_star[j,2] = np.sum(  K2*df2  )
            # Estimate of the variance factor V[j,:] = (1./denom**2) * ( ST[:,2]**2 * ST_star[:,0] + ST[:,1]**2 * ST_star[:,2] - 2.*ST[:,2]*ST[:,1]*ST_star[:,1] )
            V[j] = (ST_star[j,0]*ST[j,2]**2 - 2.*ST[j,2]*ST[j,1]*ST_star[j,1] + ST[j,1]**2*ST_star[j,2])/(denom)**2

            # Estimate of the denominator of sigma2_0
            trace = ST[j,0] - (ST[j,2]*ST_star[j,0] + ST[j,0]*ST_star[j,2] - 2.*ST[j,1]*ST_star[j,1])/denom

            res2 = (data[ii] - m_est[j])**2# + b_est[k]*df[ii])**2#
            res2_sum = np.sum( K*res2 )

            sigma2_0[j] = res2_sum/trace


    return m_est,b_est,V,sigma2_0,ST

# ==============================================================================
def localLinearEstimatorFromY(Y,f0,h,ker = 'epa',variance = True,ST_given = None):
    """
    Function estimating the power spectral density at frequencies f from a given
    data log-periodogram Y using a local linear Kernel estimation with bandwidth h

    References :
    [1] Jianqing Fan and Qiwei Yao, Nonlinear Time Series (2003), p. 284
    [2] Jianqing Fan and Irene Gijbels, Data-Driven Bandwidth Selection in Local
    Polynomial Fitting: Variable Bandwidth and Spatial Adaptation Journal of the
    Royal Statistical Society. Series B (Methodological) Vol. 57, No. 2 (1995),
    pp. 371-394

    Parameters
    ----------
    Y : 1-D numpy array of size N
        log-periodogram of the analysed data
    f0 : 1-D numpy array or scalar
        vector of frequencies where to estimate the PSD
    h : array_like
        smoothing parameter vector (size J)
    wind : character string
        type of apodization window to apply (hanning or rectangular)
    ker : {'epa','ker'}, optional
        Type of smoothing kernel
    variance : boolean (True or False), optional
        determines wether to calculate quantities required to estimate the
        variance of the PSD estimate
    ST_given : None or scalar (float), optional
        quantity used in the computation, that only depends on the chosen kernel
        and smoothing parameter h. It can be already calcualted from a previous
        computation. Otherwise, leave it as None.

    Returns
    -------
    S_est : numpy array
        estimated spectrum (size J)
    b_est : numpy array
        estimated bias (size J)
    V : numpy array
        normalized estimator variance: V * pi^2/6 is the estimated variance of log(S_est)
    sigma2_0 : scalar float
        normalized weighted residual sum of squares
    ST : array_like
        quantity used in the computation, that only depends on the chosen kernel,
        the data size N and smoothing parameter h. It can be used again for
        another computation involving different data of same size.

    """

    # Data size
    N = len(Y)
    # Number of positive frequencies
    n = np.int((N-1)/2.)
    # Number of frequencies where to estimate the PSD
    J = len(f0)
    # Mean of the logarithm of the Chi squared distribution
    C0 = -0.57721
    # The actual quantity to smooth
    YC = Y - C0
    # Normalized frequencies (such that Nyquist is 1/2)
    f_all = np.fft.fftfreq(N)
    # Strictly positive frequencies
    f = f_all[1:n+1]
    # The data to smooth
    data = YC[1:n+1]
    # Apply the basical local linear smoother to the data
    m_est,b_est,V,sigma2_0,ST = localLinearSmoother(data,f,f0,h,ker=ker,
    variance=variance,ST_given = ST_given)


    return np.exp(m_est),b_est,V,sigma2_0,ST


# ==============================================================================
def localLinearEstimatorFromI(I,f,h,ker = 'epa',variance = False,ST_given = None):
    """
    Function estimating the power spectral density at frequencies f from the
    intput periodogram I using a local linear Kernel estimation with bandwidth h

    References :
    [1] Jianqing Fan and Qiwei Yao, Nonlinear Time Series (2003), p. 284
    [2] Jianqing Fan and Irene Gijbels, Data-Driven Bandwidth Selection in Local
    Polynomial Fitting: Variable Bandwidth and Spatial Adaptation Journal of the
    Royal Statistical Society. Series B (Methodological) Vol. 57, No. 2 (1995),
    pp. 371-394

    Parameters
    ----------
    I : 1-D numpy array of size N
        periodogram of the analysed data
    f : 1-D numpy array or scalar
        frequency
    h : array_like
        smoothing parameter vector (size J)
    wind : character string
        type of apodization window to apply (hanning or rectangular)
    ker : {'epa','ker'}, optional
        Type of smoothing kernel
    variance : boolean (True or False), optional
        determines wether to calculate quantities required to estimate the
        variance of the PSD estimate
    ST_given : None or scalar (float), optional
        quantity used in the computation, that only depends on the chosen kernel
        and smoothing parameter h. It can be already calcualted from a previous
        computation. Otherwise, leave it as None.

    Returns
    -------
    S_est : numpy array
        estimated spectrum (size J)
    b_est : numpy array
        estimated bias (size J)
    V : numpy array
        normalized estimator variance: V * pi^2/6 is the estimated variance of log(S_est)
    sigma2_0 : scalar float
        normalized weighted residual sum of squares
    ST : array_like
        quantity used in the computation, that only depends on the chosen kernel,
        the data size N and smoothing parameter h. It can be used again for
        another computation involving different data of same size.

    """

    Y = np.log(I)

    S_est,b_est,V,sigma2_0,ST = localLinearEstimatorFromY(Y,f,h,ker = ker,
    variance = variance,ST_given = ST_given)

    return S_est,b_est,V,sigma2_0,ST


# ==============================================================================
def localLinearEstimator(x,f,h,wind = 'hanning',ker = 'epa',variance = True,ST_given = None):
    """
    Function estimating the power spectral density at frequencies f from a given
    data log-periodogram Y using a local linear Kernel estimation with bandwidth h

    References :
    [1] Jianqing Fan and Qiwei Yao, Nonlinear Time Series (2003), p. 284
    [2] Jianqing Fan and Irene Gijbels, Data-Driven Bandwidth Selection in Local
    Polynomial Fitting: Variable Bandwidth and Spatial Adaptation Journal of the
    Royal Statistical Society. Series B (Methodological) Vol. 57, No. 2 (1995),
    pp. 371-394

    Parameters
    ----------
    x : 1-D numpy array
        the intput data (time series of size N)
    f : 1-D numpy array or scalar
        frequency
    h : array_like
        smoothing parameter vector (size J)
    wind : character string
        type of apodization window to apply (hanning or rectangular)
    ker : {'epa','ker'}, optional
        Type of smoothing kernel
    variance : boolean (True or False), optional
        determines wether to calculate quantities required to estimate the
        variance of the PSD estimate
    ST_given : None or scalar (float), optional
        quantity used in the computation, that only depends on the chosen kernel
        and smoothing parameter h. It can be already calcualted from a previous
        computation. Otherwise, leave it as None.

    Returns
    -------
    S_est : numpy array
        estimated spectrum (size J)
    b_est : numpy array
        estimated bias (size J)
    V : numpy array
        normalized estimator variance: V * pi^2/6 is the estimated variance of log(S_est)
    sigma2_0 : scalar float
        normalized weighted residual sum of squares
    I : array_like
        periodogram of the input data (size N)

    """

    N = len(x)


    J = len(f)
    # Candidate bandwidths
    #gh = np.log(h_max/h_min)
    #h = h_min*np.exp( gh * np.arange(0,J)/(J-1.) )

    # Windowing
    if wind == 'hanning':
        w = np.hanning(N)
    elif wind == 'ones':
        w = np.ones(N)

    I = np.abs(fft(x*w))**2 / np.sum(w**2)

    S_est,b_est,V,sigma2_0,_ = localLinearEstimatorFromI(I,f,h,ker = ker,variance = variance,ST_given = ST_given)

    return S_est,b_est,V,sigma2_0,I



# ==============================================================================
def localMLEstimatorFromI(I,f,h,ker = 'epa',Niter = 1,ST_given = None):

    N = len(I)
    n = np.int((N-1)/2.)
    J = len(f)

    S_est,b0,V,sigma2_0,ST = localLinearEstimatorFromI(I,f,h,ker = ker,
    variance = False,ST_given = ST_given)
    a0 = np.log(S_est)

    data = I[1:n+1]

    ET = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )
    a_est = np.zeros( J ) + 1j*np.zeros( J )
    b_est = np.zeros( J ) + 1j*np.zeros( J )

    a_est[:] = a0
    b_est[:] = b0

    f_all = np.fft.fftfreq(N)
    f_j = f_all[1:n+1]



    # Begin maximization of the local likelihood
    for i in range(Niter):

        for k in range(J):

            if ker == 'epa' :
                b = h[k]
            elif ker == 'gauss':
                b = 10*h[k]

            df = f_j - f[k]
            ii = np.where( np.abs( df ) <= b )[0]

            df1 = df[ii]
            df2 = df[ii]**2

            #K = gaussianKernel( df[ii]/h[k] )
            K = kernel( df[ii]/h[k] , ker )
            K2 = K**2

            KIe = K * data[ii] * np.exp(-a_est[k] - df1*b_est[k]  )

            ET[k,0] = np.sum(  KIe )
            ET[k,1] = np.sum(  KIe *df1  )
            ET[k,2] = np.sum(  KIe *df2  )



            L0 = ET[k,0] - ST[k,0]
            L1 = ET[k,1] - ST[k,1]

            denom = ET[k,0]*ET[k,2]-ET[k,1]**2

            # Estimate of the local intercept a and the slope b
            a_est[k] = a_est[k] - ( ET[k,1]*L1 - ET[k,2]*L0 )/denom
            b_est[k] = b_est[k] - ( ET[k,1]*L0 - ET[k,0]*L1 )/denom


    return np.exp(a_est),b_est



# ==============================================================================
class PSD_estimate:
    """
    Class providing methods to estimate and calculate the power spectral density
    of 1-D stationary processes with continuous, smooth spectrum.


    Parameters
    ----------
    N_est : scalar integer
        number of frequency points where to estimate the noise power spectral
        density (on a logarithmic grid)
    N : scalar integer
        size of the analysed time series
    Npoints : scalar integer
        Size of the Fourier grid on which the PSD is estimated. This means that
        the PSD can be evaluated at normalized frequencies between 1/Npoints and
        1/2
    h_min : scalar float, optional
        minimal value of the smoothing parameter
    h_max : scalar float, optional
        maximal value of the smoothing parameter


    Attributes
    ----------
    N_est : scalar integer
        number of frequency points where to estimate the noise power spectral
        density (on a logarithmic grid)
    N : scalar integer
        size of the analysed time series
    f_est : array_like
        vector of normalized frequencies (size N_est) at which the PSD is
        estimated
    S_est : array_like
        vector of PSD estimates at frequencies contained in f_est (size N_est)
    h : array_like
        vector of smoothing parameter values corresponding to the PSD estimates
        at frequencies contained in f_est (size N_est)
    I : array_like
        periodogram of the data (size N)
    V : numpy array
        normalized estimator variance: V * pi^2/6 is the estimated variance of log(S_est)
    ST : array_like
        quantity used in the computation of the PSD estimates, that only depends
        on the chosen kernel, the data size N and smoothing parameter h. It can
        be used again for another computation involving different data of same
        size.
    PSD_function : function of one variable
        function giving the value of the estimated PSD at any normalized
        frequency between 1/Npoints and 1/2.
    PSD_variance_function : function of one variable
        function giving the value of the estimated normalized log-PSD variance at
        any normalized frequency between 1/Npoints and 1/2.


    """

    def __init__(self,N_est,N,Npoints,h_min = None,h_max = None):

        self.N = N
        # Frequency vector where to estimate the PSD
        self.N_est = N_est
        self.f_est = np.zeros(N_est+1)

        self.f_est[1:N_est+1] = 1./Npoints * np.exp( np.log(Npoints/2.)*np.arange(0,N_est)/(N_est-1))
        # Initialize the bandwidths parameters of the PSD estimate
        if h_min is None:
            h_min = 3./self.N
        if h_max is None:
            h_max = 0.05

        J = len(self.f_est)
        gh = np.log(h_max/h_min)
        self.h = h_min*np.exp( gh * np.arange(0,J)/(J-1.) )

        self.PSD_function = None
        self.I = None
        self.S_est = None

        self.ST,self.V = self.calculateST(self.f_est,self.h,N)

        self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],self.V[self.f_est>0]*np.pi**2/6.)

    def estimate(self,x,w='hanning',periodogram=False):
        """
        method computing the PSD estimate of the input data x at frequencies
        fj contained in f_est, using the local least-squares technique
        Calculate or update the values of the attributes S_est, PSD_function,
        PSD_variance_function, and possibly I


        Parameters
        ----------
        x : array_like
            Input data array (size N)
        w : characted string
            type of apodization window to apply
        periodogram : boolean
            if True, the periodogram is stored in the attribute "I"

        """
        # Estimation of the PSD function
        self.S_est,b_est,V_est,sigma2_0_est,I=localLinearEstimator(x,self.f_est,
        self.h,wind=w,ST_given = self.ST)
        # Calcualte the interpolation function
        self.PSD_function = interpolate.interp1d(self.f_est,np.log(self.S_est))
        self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],
        V_est[self.f_est>0]*np.pi**2/6.)

        if periodogram == True:

            self.I = I

    def estimateFromI(self,I):
        """
        method computing the PSD estimate from the values of the input
        periodogram I. Calculate or update the values of the attributes S_est,
        PSD_function, PSD_variance_function


        Parameters
        ----------
        x : array_like
            Input data array (size N)
        w : characted string
            type of apodization window to apply
        periodogram : boolean
            if True, the periodogram is stored in the attribute "I"


        """


        # Estimation of the PSD function
        self.S_est,b_est,V_est,sigma2_0_est,_=localLinearEstimatorFromI(I,self.f_est,self.h,ST_given = self.ST)
        # Calcualte the interpolation function
        self.PSD_function = interpolate.interp1d(self.f_est,np.log(self.S_est))
        self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],V_est[self.f_est>0]*np.pi**2/6.)

    def MLestimateFromI(self,I,Niter = 1):
        """
        method computing the PSD estimate from the input periodogram I at
        frequencies fj contained in f_est, using the local maximum likelihood
        technique.
        Calculate or update the values of the attributes S_est, PSD_function


        Parameters
        ----------
        I : array_like
            periodogram array (size N)
        Niter : scalar integer
            number of iterations in the Newton-Raphson algorithm used to compute
            the maximum likelihood estimate.

        """
        # Estimation of the PSD function
        self.S_est,b_est = localMLEstimatorFromI(I,self.f_est,self.h,Niter = Niter,ST_given = self.ST)
        # Calcualte the interpolation function
        self.PSD_function = interpolate.interp1d(self.f_est,np.log(self.S_est))
        #self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],V_est[self.f_est>0]*np.pi**2/6.)

    def MLestimate(self,x,w='hanning',Niter = 1):
        """
        method computing the PSD estimate of the input data x at frequencies
        fj contained in f_est, using the local maximum likelihood technique.
        Calculate or update the values of the attributes S_est, PSD_function


        Parameters
        ----------
        x : array_like
            input data array (size N)
        w : characted string
            type of apodization window to apply
        Niter : scalar integer
            number of iterations in the Newton-Raphson algorithm used to compute
            the maximum likelihood estimate.

        """
        # Compute the periodogram
        I = periodogram(x,len(x),wind = w)
        # Compute local maximum likelihood estimator
        self.MLestimateFromI(I,Niter = Niter)

    def conditionalEstimate(self,condDraws,wind='hanning'):

        (Nd,N) = np.shape(condDraws)

        # Initialization of the conditional mean of the periodogram if necessary
        S_est_mat= np.zeros((Nd,N))


        for i in range(Nd):

            # Estimation of the PSD function
            S_est_mat[i,:],b_est,V_est,sigma2_0_est,I=localLinearEstimator(condDraws[i,:],self.f_est,self.h,wind=wind,ST_given = ST)

        # Calcualte the average
        S_est_mean = np.mean(S_est_mat,axis=0)
        # Store the value of S_est
        self.S_est = S_est_mean

        # Calcualte the interpolation function
        f = np.fft.fftfreq(N)
        n = np.int( (N-1)/2.)

        self.PSD_function = interpolate.interp1d(self.f_est,np.log(S_est_mean))
        self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],V_est[self.f_est>0]*np.pi**2/6.)

        self.S_est = S_est_mean



    def calculate(self,arg):
        """
        method calculating the value of the PSD function at either a given vector
        of frequency values or on the Fourier grid of given size.
        The method estimate must be called beforehand with the analysed data as
        intput.


        Parameters
        ----------
        args : array_like or scalar integer
            vector of frequencies value or size of the Fourier grid where to
            evaluate the PSD function.

        Returns
        -------
        S_est_N_sym : array_like
            vector of PSD values corresponding to specified input frequencies


        """


        if type(arg)==np.int:
            N = arg
            f = np.fft.fftfreq(N)
        elif type(arg) == np.array:
            f = arg[:]
            N = len(f)


        n = np.int( (N-1)/2.)

        m_est_interp_N = self.PSD_function(f[0:n+1])

        # Symmetrize the estimates
        SN = np.exp(m_est_interp_N[1:n+1])
        S_est_N_sym = symmetrize(np.real(SN),N)
        # Take the exponential value
        S_est_N_sym[0] = np.exp(np.real(m_est_interp_N[0]))

        return S_est_N_sym

    def calculateVariance(self,arg):
        """
        method calculating the value of the PSD function at either a given
        vector of frequency values or on the Fourier grid of given size.
        The method estimate must be called beforehand with the analysed data as
        intput.


        Parameters
        ----------
        args : array_like or scalar integer
            vector of frequencies value or size of the Fourier grid where to
            evaluate the PSD function.

        Returns
        -------
        variance_N_sym : array_like
            vector of (normalized) PSD variances corresponding to specified
            input frequencies


        """

        if type(arg)==np.int:
            N = arg
            f = np.fft.fftfreq(N)
        elif type(arg) == np.array:
            f = arg[:]
            N = len(f)

        f = np.fft.fftfreq(N)
        n = np.int( (N-1)/2.)

        variance_N = self.PSD_variance_function(f[1:n+1])

        variance_N_sym = symmetrize(np.real(variance_N),N)

        return variance_N_sym

    def confidenceInterval(self,S_est,variance,alpha):

        z = stats.norm.ppf(1.-alpha/2.)

        ci_low = np.real( np.exp(np.log(S_est) - z*np.sqrt( variance ) ) )
        ci_up =  np.real( np.exp(np.log(S_est) + z*np.sqrt( variance ) ) )

        return ci_low,ci_up


    def calculateST(self,f,h,N,ker = 'epa'):
        """
        Method calculating a kernel-dependant quantity that is necessary for the
        PSD estimation.


        Parameters
        ----------
        f : array_like
            vector of frequencies where to estimate the PSD
        h : scalar float
            smoothing parameter
        N : scalar integer
            size of the analysed data
        ker : {'epa','gauss'}, optional
            type of kernel smoothing function


        Returns
        -------
        ST : array_like
            quantity used in the computation of the PSD estimates, that only
            depends on the chosen kernel, the data size N and smoothing
            parameter h. It can be used again for another computation involving
            different data samples of same size.
        V : numpy array
            normalized estimator variance: V * pi^2/6 is the estimated variance
            of log(S_est)

        """
        n = np.int((N-1)/2.)
        J = len(f)
        f_all = np.fft.fftfreq(N)
        f_j = f_all[1:n+1]

        ST = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )

        # For variance calculation
        ST_star = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )
        V = np.zeros( J ) + 1j*np.zeros( J )

        for k in range(J):

            if ker == 'epa' :
                b = h[k]
            elif ker == 'gauss':
                b = 10*h[k]

            df = f_j - f[k]
            ii = np.where( np.abs( df ) <= b )[0]

            df1 = df[ii]
            df2 = df[ii]**2

            K = kernel( df[ii]/h[k] , ker )
            ST[k,0] = np.sum(  K  )
            ST[k,1] = np.sum(  K*df1  )
            ST[k,2] = np.sum(  K*df2  )


            denom = ST[k,0]*ST[k,2]-ST[k,1]**2

            K2 = K**2
            ST_star[k,0] = np.sum(  K2  )
            ST_star[k,1] = np.sum(  K2*df1  )
            ST_star[k,2] = np.sum(  K2*df2  )
            # Estimate of the variance factor V[j,:] = (1./denom**2) * ( ST[:,2]**2 * ST_star[:,0] + ST[:,1]**2 * ST_star[:,2] - 2.*ST[:,2]*ST[:,1]*ST_star[:,1] )
            V[k] = (ST_star[k,0]*ST[k,2]**2 - 2.*ST[k,2]*ST[k,1]*ST_star[k,1] + ST[k,1]**2*ST_star[k,2])/(denom)**2

        return ST,V
