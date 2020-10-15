# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# ==============================================================================
# This code provides algorithms for estimating pronbability and sptrum densities
# by using local linear smoothing
# ==============================================================================
import numpy as np
from scipy import interpolate
from scipy import stats
from scipy import optimize
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
    else:
        ST = ST_given[:]

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
# The Local Maximum likelihood estimator
# ==============================================================================
def localMLEstimator(data,f,fj,h,a0,b0,ker = 'epa',Niter = 1,ST_given = None):
    """
    Function computing the local maximum likelihood linear estimation
    of the input data at points fj, given that the input data are available
    at points f.


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
    a0 : array_like
        first guess for the local intersects estimate
    b0 : array_like
        first guess for the local slopes estimate
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

    ET = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )
    a_est = np.zeros( J ) + 1j*np.zeros( J )
    b_est = np.zeros( J ) + 1j*np.zeros( J )

    a_est[:] = a0
    b_est[:] = b0

    if ST_given is None:
        ST = np.zeros( (J,3) ) + 1j*np.zeros( (J,3) )
    else:
        ST = ST_given[:]


    # Begin maximization of the local likelihood
    for i in range(Niter):

        for k in range(J):

            if ker == 'epa' :
                b = h[k]
            elif ker == 'gauss':
                b = 10*h[k]

            df = fj - f[k]
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

            if ST_given is None :
                ST[k,0] = np.sum(  K  )
                ST[k,1] = np.sum(  K*df1  )
                ST[k,2] = np.sum(  K*df2  )

            L0 = ET[k,0] - ST[k,0]
            L1 = ET[k,1] - ST[k,1]

            denom = ET[k,0]*ET[k,2]-ET[k,1]**2

            # Estimate of the local intercept a and the slope b
            a_est[k] = a_est[k] - ( ET[k,1]*L1 - ET[k,2]*L0 )/denom
            b_est[k] = b_est[k] - ( ET[k,1]*L0 - ET[k,0]*L1 )/denom


    return a_est,b_est




# ==============================================================================
class PSDEstimate:
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
    Npoints : scalar integer
        Define the minimum and maximum frequencies of the logarithmic grid
        (of size N_est) where the PSD is estimated
    f_est : array_like
        vector of normalized frequencies (size N_est) at which the PSD is
        estimated
    S_est : array_like
        vector of PSD estimates at frequencies contained in f_est (size N_est)
    h : array_like
        vector of smoothing parameter values corresponding to the PSD estimates
        at frequencies contained in f_est (size N_est)
    per : array_like
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

    def __init__(self, N_est, N, Npoints, h_min=None, h_max=None,
                 kind='linear', wind='hanning', ker='epa'):

        self.N = N
        self.f = np.fft.fftfreq(N)
        n = np.int((N-1)/2.)
        # Frequency vector where to estimate the PSD
        self.N_est = N_est

        # Interpolation type
        self.kind = kind
        # Possible windowing of the periodogram
        self.wind = wind
        if wind == 'hanning':
            self.w = np.hanning(N)
            self.s2 = np.sum(self.w**2)
        elif wind == 'ones':
            self.w = np.ones(N)
            self.s2 = N
        # Mean of the logarithm of the Chi squared distribution
        self.C0 = -0.57721

        # Initialize the bandwidths parameters of the PSD estimate
        if h_min is None:
            h_min = 3./self.N
        if h_max is None:
            h_max = 0.05

        # self.f_est = 1./Npoints * np.exp( np.log(Npoints/2.)*np.arange(0,N_est)/(N_est-1))
        # gh = np.log(h_max/h_min)
        # self.h = h_min*np.exp( gh * np.arange(0,N_est)/(N_est-1.) )

        # Test another method
        self.f_est = self.choose_knots(N_est,1.0/Npoints,1.0/2.)
        self.h = self.choose_knots(N_est,h_min,h_max)


        #self.PSD_function = None
        self.logf_est = np.log(self.f_est)
        self.logf = np.log(self.f[1:n+1])
        self.logPSD_function = None
        self.m_est = None
        self.ker = ker
        self.ST,self.V = self.calculateST(self.f_est,self.h,N,ker = ker)
        self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],
        self.V[self.f_est>0]*np.pi**2/6.,kind = self.kind)

    def choose_knots(self,J,fmin,fmax):
        """

        Choose frequency knots such that

        f_knots = 10^-n_knots

        where the difference
        n_knots[j+1] - n_knots[j] = dn[j]

        is a geometric series.

        Parameters
        ----------
        J : scalar integer
            number of knots
        fmin : scalar float
            minimum frequency knot
        fmax : scalar float
            maximum frequency knot


        """

        ns = - np.log(fmax)/np.log(10)
        n0 = - np.log(fmin)/np.log(10) #6
        jvect = np.arange(0,J)
        #b = (dn_f - dn_0)/(J-1)
        #n_knots = n0 - dn_0*jvect + jvect*(jvect-1)/2.*b
        alpha_guess = 0.8

        targetfunc = lambda x : n0 - (1-x**(J))/(1-x) - ns
        result = optimize.fsolve(targetfunc, alpha_guess)
        alpha = result[0]
        n_knots = n0 - (1-alpha**jvect)/(1-alpha)
        #f_knots = 10**(-n_knots[0:J-1])
        f_knots = 10**(-n_knots)

        # Force the first and last knots
        f_knots[0] = fmin
        f_knots[J-1] = fmax

        return f_knots

    def compute_periodogram(self,x):
        """
        Compute the windowed periodogram from time series x,
        along with Fourier frequency grid
        """
        # If size of analysed data is the same as the presets
        if (x.shape[0] == self.N):
            if self.wind == 'hanning':
                # Compute periodogram
                per = np.abs(fft(x*self.w))**2 / self.s2
            elif self.wind == 'ones':
                per = np.abs(fft(x))**2 / self.s2
        else:
            if self.wind == 'hanning':
                w = np.hanning(x.shape[0])
                s2 = np.sum(w**2)
                per = np.abs(fft(x*w))**2 / s2
            elif self.wind == 'ones':
                s2 = x.shape[0]
                per = np.abs(fft(x))**2 / s2

        return per

    def estimate(self, x, variance=False):
        """
        method computing the PSD estimate of the input data x at frequencies
        fj contained in f_est, using the local least-squares technique
        Calculate or update the values of the attributes S_est, PSD_function,
        PSD_variance_function, and possibly per


        Parameters
        ----------
        x : array_like
            Input data array (size N)
        w : characted string
            type of apodization window to apply
        variance : boolean
            if True, compute an estimate of the variance of the PSD estimator

        """
        # Compute periodogram
        per = self.compute_periodogram(x)
        # Compute log-psd estimate from periodogram
        self.estimate_from_periodogram(per, variance=variance)

    def estimate_from_periodogram(self, per, variance=False):
        """
        method computing the PSD estimate from the values of the input
        periodogram per. Calculate or update the values of the attributes S_est,
        PSD_function, PSD_variance_function


        Parameters
        ----------
        x : array_like
            Input data array (size N)
        w : characted string
            type of apodization window to apply
        periodogram : boolean
            if True, the periodogram is stored in the attribute "per"


        """

        # Compute Fourier grid
        if (per.shape[0] == self.N):
            f = self.f[:]
        else:
            f = np.fft.fftfreq(per.shape[0])
        n = np.int((per.shape[0]-1)/2.)

        # Estimate log-psd
        self.m_est, b_est, V, sigma2_0, ST = localLinearSmoother(
            np.log(per[1:n+1])-self.C0, f[1:n+1], self.f_est, self.h,
            ker=self.ker, variance=variance, ST_given=self.ST)

        # Calculate the interpolation function
        self.logPSD_function = interpolate.interp1d(self.logf_est, self.m_est,
                                                    kind=self.kind,
                                                    fill_value="extrapolate")

        if variance:
            self.PSD_variance_function = interpolate.interp1d(
                self.logf_est, V[self.f_est > 0] * np.pi**2/6., kind=self.kind,
                fill_value='extrapolate')

    def ml_estimate_from_per(self, per, Niter=1, variance=False):
        """
        method computing the PSD estimate from the input periodogram per at
        frequencies fj contained in f_est, using the local maximum likelihood
        technique.
        Calculate or update the values of the attributes S_est, PSD_function


        Parameters
        ----------
        per : array_like
            periodogram array (size N)
        Niter : scalar integer
            number of iterations in the Newton-Raphson algorithm used to
            compute the maximum likelihood estimate.

        """

        # Compute Fourier grid
        if (per.shape[0] == self.N):
            f = self.f[:]
        else:
            f = np.fft.fftfreq(per.shape[0])
        n = np.int((per.shape[0]-1)/2.)

        # First estimate log-psd using least-square estimator
        a_est, b_est, V, sigma2_0, ST = localLinearSmoother(
            np.log(per[1:n+1])-self.C0, f[1:n+1], self.f_est, self.h,
            ker=self.ker, variance=variance, ST_given=self.ST)
        # Refine estimate using maximum likelihood
        self.m_est, b_est = localMLEstimator(per[1:n+1], f[1:n+1], self.f_est,
                                             self.h, a_est, b_est,
                                             ker=self.ker,
                                             Niter=Niter, ST_given=self.ST)

        # # Estimation of the PSD function
        # self.S_est,b_est = localMLEstimatorFromI(per,self.f_est,self.h,Niter = Niter,
        # ST_given = self.ST)
        # # Calcualte the interpolation function
        # self.PSD_function = interpolate.interp1d(self.f_est,np.log(self.S_est),
        # kind=self.kind)
        # Calcualte the interpolation function
        self.logPSD_function = interpolate.interp1d(self.logf,
                                                    self.m_est,
                                                    kind=self.kind,
                                                    fill_value="extrapolate")
        # Interpolate the PSD itself, not the log-psd
        # self.PSD_function = interpolate.interp1d(self.f_est,self.S_est,
        # kind = self.kind)

        if variance:
            # Calcualte the variance interpolation function
            self.logPSD_function = interpolate.interp1d(self.logf,
            V*np.pi**2/6., kind=self.kind, fill_value="extrapolate")
        #self.PSD_variance_function = interpolate.interp1d(self.f_est[self.f_est>0],V_est[self.f_est>0]*np.pi**2/6.)

    def MLestimate(self, x, Niter=1, variance=False):
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
        per = self.compute_periodogram(x)
        # Compute local maximum likelihood estimator
        self.ml_estimate_from_per(per,Niter = Niter,variance = variance)


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
            n = np.int( (N-1)/2.)


            # Symmetrize the estimates
            if (N % 2 == 0): # if N is even
                # Compute PSD from f=fs/N to f = fs/2
                fS = np.concatenate(([f[1]/10.],np.abs(f[1:n+2])))
                S = np.real( np.exp( self.logPSD_function(np.log(fS)) ) )
                SN_sym = np.concatenate((S[0:n+1],S[1:n+2][::-1]))

            else: # if N is odd
                fS = np.concatenate(([f[1]/10.],np.abs(f[1:n+1])))
                S = np.real( np.exp( self.logPSD_function(np.log(np.abs(fS))) ) )
                SN_sym = np.concatenate((S[0:n+1],S[1:n+1][::-1]))

            # Symmetrize the estimates
            # S = np.concatenate( ( [S0] ,
            # np.exp( np.real(self.logPSD_function(np.log(f[1:n+1])) )) ))
            # SN_sym = symmetrize(S[1:n+1],N)
            # SN_sym[0] = S0

        elif type(arg) == np.ndarray:
            f = arg[:]
            SN_sym = np.real( np.exp(self.logPSD_function(np.log(np.abs(f)))) )
            #S_est_N_sym = np.real(self.PSD_function(f))

        else:
            raise TypeError("Argument must be integer or ndarray")


        return SN_sym

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
            n = np.int( (N-1)/2.)

            variance_N = self.PSD_variance_function(f[1:n+1])
            variance_N_sym = symmetrize(np.real(variance_N),N)

        elif type(arg) == np.ndarray:
            f = arg[:]
            variance_N_sym = self.PSD_variance_function(f)

        else:
            raise TypeError("Argument must be integer or ndarray")


        return variance_N_sym

    def confidenceInterval(self,S_est,variance,alpha):

        z = stats.norm.ppf(1.-alpha/2.)

        ci_low = np.real( np.exp(np.log(S_est) - z*np.sqrt( variance ) ) )
        ci_up =  np.real( np.exp(np.log(S_est) + z*np.sqrt( variance ) ) )

        return ci_low,ci_up
    
    def calculate_autocorr(self, n_data):
        """
        Compute the autocovariance function from the PSD.

        """

        return np.real(ifft(self.calculate(2*n_data))[0:n_data])


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
