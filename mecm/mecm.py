# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# ==============================================================================
# This routine perform the Modified expectation maximization algorithm (M-ECM)
# based on missing data imputation by conditional distribution using FFT.
# ==============================================================================
import numpy as np
from scipy import sparse
from scipy import stats
from scipy import interpolate
from numpy import linalg as LA
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
pyfftw.interfaces.cache.enable()
import time
import sys
import random
import multiprocessing as mp
#from numba import jit
import copy

from .matrixalgebra import * #covLinearOp,precondLinearOp,matPrecondBiCGSTAB,precondBiCGSTAB,matVectProd,printPCGstatus
from .noise import generateNoiseFromDSP, symmetrize
from .leastsquares import pmesureWeighted,pmesure_optimized_TF
from .localestimator import nextpow2,PSD_estimate



def maxlike(y,M,A,N_it_max=15,eps=1e-3,p=20,Nd=10,N_est=1000,Nit_cg=200,
tol_cg=1e-4,compute_cov = True,verbose = True,PCGalgo = 'scipy'):
    """

    Function estimating the regression parameters for a problem of
    multivariate Gaussian maximum likelihood with missing data,
    using the M-ECM algorithm.

    Parameters
    ----------
    y : numpy array (size N)
        masked data vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    A : numpy array (size N x K)
        design matrix (contains partial derivatives of signal wrt parameters)
    N_it_max : scalar integer
        number of iterations of the M-ECM algorithm
    eps : scalar float
        tolerance criterium to stop the ECM algorithm (default is 1e-3)
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the conjugate
        gradients.
    Nd : scalar integer
        number of Monte-Carlo draws to estimate the conditional expectation of
        the noise periodogram with respect to the observed data
    N_est : scalar integer
        number of frequency points where to estimate the noise power spectral
        density (on a logarithmic grid)
    N_it_cg : scalar integer
        maximum number of iterations for the conjugate gradient algorithm.
    tol_cg : scalar float
        tolerance criterium to stop the PCG algorithm (default is 1e-4). Stops
        when the relative residual error ||A x - b||/||b|| is below tol_cg.
    verbose : boolean
        if True, a message is printed at the end of each iteration, displaying
        the value of the convergence criterion
    compute_cov : boolean
        if True, the covariance of the estimator is computed
    PCGalgo : string {'mine','scipy','scipy.bicgstab','scipy.bicg','scipy.cg','scipy.cgs'}
        Type of preconditioned conjugate gradient (PCG) algorithm to use among

    Returns
    -------
    beta_new : array_like (size K)
        final value of the estimated parameter vector
    cov : None or array_like (size K x K)
        estimated covariance matrix of the parameter vector. If compute_cov is
        False, then cov is None
    beta : array_like (size N_iterations x K)
        vector storing the updates of the parameter vector at each iteration
    y_rec : array_like (size N)
        reconstructed data vector, i.e. conditional expectation of the data
        given the available observations.
    I_condMean : array_like (size N)
        conditional expectation of the noise periodogram
    PSD : PSD_estimate class instance
        class containing all the information regarding the estimated noise PSD


    References
    ----------

    .. [1] Q. Baghi et al, "Gaussian regression and power spectral density estimation with missing data: The MICROSCOPE space mission as a case study," Physical Review D, vol. 93, num. 12, 2016


    Notes
    -----

    """

    # First guess: ordinary least squares
    beta0 = pmesureWeighted(y*M, A, M)
    beta0_norm = LA.norm(beta0)

    # Initialization of variables
    beta_old = np.zeros( np.shape(beta0) )
    beta_vect = copy.deepcopy(beta0)
    beta_new = copy.deepcopy(beta0)



    # Difference from one update to the other
    diff = 1.

    # Number of data points
    N = len(y)
    # Number of Fourier frequencies used for the spectrum calculation
    if N % 2 == 0:
        P = np.int(2*N)
    else:
        P = np.int(2**nextpow2(2*N))

    # Number of positive Fourier frequencies
    n = np.int( (N-1)/2. )

    # Extended mask to the circulant process
    PSD = PSD_estimate(N_est,N,P,h_min = None,h_max = None)
    counter = 0


    # PSD initialization
    PSD.estimate(M*(y-np.dot(A,beta0)))
    S_N = PSD.calculate(N)
    S_2N = PSD.calculate(P)

    diff = 1.0

    if 'scipy' in PCGalgo:
        print("The chosen preconditioned conjugate gradient algorithm is from scipy.")


    while ( ( diff > eps) & (counter <= N_it_max) ) :

        # The new beta becomes the old one
        beta_old = beta_new

        # E step : missing data reconstruction
        # ------------------------------------
        R = np.real(ifft(S_2N))
        y_rec,solver = conjGradImputation(y,A,beta_old,M,S_2N,R,p,Nit_cg,PCGalgo,tol=tol_cg)

        # Conditional draws
        condDraws = conditionalMonteCarlo(M*(y - np.dot(A,beta_old)),solver,M,
        Nd,S_2N,Nit_cg,PCGalgo,tol=tol_cg)
        # Periodogram of the draws
        I_mat = periodogram(condDraws,condDraws.shape[1])
        # Conditional average
        I_condMean = np.mean(I_mat,axis=0)
        #logI_condMean = np.mean( np.log(I_mat,axis=0) )


        # CE1 step : estimation of deterministic model parameters
        # -------------------------------------------------------
        beta_new = np.real(pmesure_optimized_TF(y_rec,A,S_N))
        beta_vect = np.vstack( (beta_vect,beta_new) )

        # CE2 step : estimation of noise model parameters (PSD)
        # -------------------------------------------------------
        PSD.MLestimateFromI(I_condMean,Niter = 1)
        S_N = PSD.calculate(N)
        S_2N = PSD.calculate(P)


        # Update counter
        counter += 1
        # Calculate the relative difference between new update and old
        # parameter estimate
        diff = LA.norm(beta_new-beta_old)/beta0_norm

        if verbose:
            print('--------------------------')
            print('Iteration ' + str(counter) + ' completed, de/e =' + str(diff))
            print('--------------------------')

    print("------ MECM iterations completed. ------")
    if diff > eps:
        raise Warning("Attention: MECM algorithm ended \
        without reaching the specified convergence criterium.")
        print("Current criterium: " +str(diff) + " > " + str(eps))

    print("Computation of the covariance...")
    if compute_cov:
        cov,U = mecmcovariance(A,M,S_2N,solver,PCGalgo,Nit=Nit_cg,tol=tol_cg)
    else:
        cov = None


    return beta_new,cov,beta_vect,y_rec[0:N],I_condMean,PSD



# ==============================================================================
def mecmcovariance(A,M,S_2N,solve,PCGalgo,Nit=150,tol=1e-7,r=1e-15):
    """
    Function estimating the covariance of regression parameters for a problem of
    multivariate Gaussian maximum likelihood with missing data.

    Parameters
    ----------
    A : numpy array (size N x K)
        design matrix (contains partial derivatives of signal wrt parameters)
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    S_2N : numpy array (size P >= 2N)
        PSD vector
    solve : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    Nit : scalar integer, optional
        maximum number of iterations for the conjugate gradient algorithm.
    tol : scalar float, optional
        tolerance criterium to stop the PCG algorithm (default is 1e-7)
    r : scalar float, optional (default is 1e-15)
        Cutoff for small singular values when computing the inverse matrix.
        Singular values smaller (in modulus) than rcond * largest_singular_value
        (in modulus) are set to zero.

    Returns
    -------
    cov : array_like (size K x K)
        estimated covariance matrix of the parameter vector

    """

    print("The chosen BICGSTAB algorithm is scipy code.")

    A_obs =  A[M==1,:]
    (N_o,K) = np.shape(A_obs)
    U = np.zeros((N_o,K),dtype=np.float64)
    # Missing data indices
    ind_mis = np.where( M==0 )[0]
    ind_obs = np.where( M!=0 )[0]
    N = len(M)

    # Construct the operator calculating C_OO*x precondBiCGSTAB(x0,b,A_func,Nit,stp,P)
    x0 = np.zeros(N_o)

    if PCGalgo == 'mine':
        Coo_func = lambda x: matVectProd(x,ind_obs,ind_obs,M,S_2N)
        U = matPrecondBiCGSTAB(x0,A_obs,Coo_func,Nit,tol,solve,PCGalgo=PCGalgo)
    elif 'scipy' in PCGalgo:
        Coo_op = covLinearOp(ind_obs,ind_obs,M,S_2N)
        P_op = precondLinearOp(solve,len(ind_obs),len(ind_obs))
        U = matPrecondBiCGSTAB(x0,A_obs,Coo_op,Nit,tol,P_op,PCGalgo=PCGalgo)
        #innerPrecondBiCGSTAB(U,x0,A_obs,Coo_op,Nit,tol,P_op,PCGalgo)


    try:
        cov = LA.pinv( np.dot( np.conj(A_obs.T), U ) ,rcond=r)
    except:
        cov = []
        "SVD did not converge"


    return cov,U

# ==============================================================================
def periodogram(x_mat,nfft,wind = 'hanning'):
    """
    Function computing the periodogram of the input data with a specified number
    of Fourier frequencies nfft (possible zero-padding) and apodization window.

    Parameters
    ----------
    x_mat : numpy array (size Nd x N)
        data for which the periodogram must be computed. Can be a one-dimentional
        vector or a matrix containing a time series in each row. The function
        computes as many periodograms as there are rows in x_mat, and returns
        a matrix of same size as x_mat.

    Returns
    -------
    Per : numpy array (size Nd x N)
        matrix (or vector) of periodogram

    """

    l = len(np.shape(x_mat))

    if l == 1 :
        N = len(x_mat)
        # Windowing
        if wind == 'hanning':
            w = np.hanning(N)
        elif wind == 'ones':
            w = np.ones(N)
        elif isinstance(wind, (list, tuple, np.ndarray)):
            w = wind
        else:
            raise TypeError("Window argument is wrong")

        K2 = np.sum(w**2)
        Per = np.abs( fft( x_mat*w, nfft ) )**2 / K2

    elif l==2 :
        (Nd,N) = np.shape(x_mat)
        # Windowing
        if wind == 'hanning':
            w = np.hanning(N)
        elif wind == 'ones':
            w = np.ones(N)
        elif isinstance(wind, (list, tuple, np.ndarray)):
            w = wind
        else:
            raise TypeError("Window argument is wrong")

        K2 = np.sum(w**2)
        Per = np.abs( fft( np.multiply(x_mat,w), n = nfft, axis = 1 ) )**2 / K2

    return Per


# ==============================================================================
def conditionalDraw(Np,Nit,S_2N,solver,z_o,mu_m_given_o,ind_obs,
ind_mis,M,PCGalgo,tol=1e-7):
    """
    Function performing random draws of the complete data noise vector
    conditionnaly on the observed data.

    Parameters
    ----------
    Np : scalar integer
        Size of the initial random vector which is drawn. The initial vector
        follows a stationary distribution whose covariance matrix is circulant.
        It is then truncated to N_final to obtain the desired distribution whose
        covariance matrix is Toeplitz.
    Nit : scalar integer
        Maximum number of conjugate gradient iterations.
    S_2N : numpy array (size P >= 2N)
        PSD vector
    Coo_func : function
        Linear operator that calculates the matrix-to-vector product Coo x for
        any vector x of size No (number of observed data points)
    solver : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    z_o : array_like (size No)
        vector of observed residuals
    mu_m_given_o : array_like (size Nm)
        vector of conditional expectation of the missing data given the observed
        data
    ind_obs : array_like (size No)
        vector of chronological indices of the observed data points in the
        complete data vector
    ind_mis : array_like (size No)
        vector of chronological indices of the missing data points in the
        complete data vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    tol : scalar float
        Stoping criteria for the conjugate gradient algorithm, optional

    Returns
    -------
    eps : numpy array (size N)
        realization of the vector of conditional noise given the observed data

    """

    # New implementation: the size of the vector that is randomly drawn is
    # equal to the size of the mask. If one wants a larger vector, one must
    # have to zero-pad the mask: M = np.concatenate((M,np.zeros(Npad)))
    N = len(M)
    DSP = np.sqrt(S_2N*2.)
    v = generateNoiseFromDSP(DSP,1.)
    #v = np.array([random.gauss(0,1.) for _ in range(Np)])
    #e = np.sqrt(Np/np.float(N_final))*np.sqrt(Np)*np.real( ifft( np.sqrt(S_2N) * v )[0:N_final] )
    e = np.real(v[0:N]*np.sqrt(np.var(v)/np.var(v[0:N])))

    #u0 = solve(e[self.ind_obs])

    N_o = len(ind_obs)
    u0 = np.zeros(N_o)
    stop = np.std(e[ind_obs])
    #stop = 0

    u = PCGsolve(ind_obs,M,S_2N,e[ind_obs],u0,tol,Nit,solver,PCGalgo)

    #u,sr = conjugateGradientSolvePrecond(u0,e[ind_obs],Coo_func,Nit,stop,solver)

    u_m_o = matVectProd(u,ind_obs,ind_mis,M,S_2N)

    eps = np.zeros(N)

    # Observed data points
    eps[ind_obs] = z_o
    # Missing data points and extension to 2N points
    eps[ind_mis] = mu_m_given_o + e[ind_mis] - u_m_o

    return eps

# ==============================================================================
def conditionalMonteCarlo(eps,solve,M,Nd,S_2N,Nit,PCGalgo,seed = None,tol=1e-7):
    """
    Function performing random draws of the complete data noise vector
    conditionnaly on the observed data.

    Parameters
    ----------
    eps : numpy array (size N)
        input vector of residuals (difference between data and model)
    solve : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    S_2N : numpy array (size P >= 2N)
        PSD vector
    Nit : scalar integer
        Maximum number of conjugate gradient iterations.
    seed : character string, optional
        Regenerate the seed with system time (if None) or chosen integer.
    tol : scalar float
        Stoping criteria for the conjugate gradient algorithm, optional


    Returns
    -------
    condDraws : numpy array (size Nd x N)
        Matrix whose rows are realization of the conditional noise vector

    """

    # Missing data indices
    ind_mis = np.where( M==0 )[0]
    ind_obs = np.where( M!=0 )[0]
    N = len(M)
    Np = len(S_2N)

    # Calculate the residuals at the observed data points
    eps_o = eps[ind_obs]

    # Calculate the conditional mean
    N_o = len(ind_obs)
    x0 = np.zeros(N_o)
    u = PCGsolve(ind_obs,M,S_2N,eps_o,x0,tol,Nit,solve,PCGalgo)

    #u,sr = conjugateGradientSolvePrecond(np.zeros(len(ind_obs)),eps_o,Coo_func,Nit,np.std(eps_o),solve)
    mu_m_given_o = matVectProd(u,ind_obs,ind_mis,M,S_2N)
    # Calculate the stopping criteria
    stop = np.std(eps_o)

    condDraws = np.zeros((Nd,N))
    # Regenerate the seed with system time or chosen integer
    random.seed(seed)

    # ==========================================================================
    # Without parallelization:
    # ==========================================================================
    for i in range(Nd):
        condDraws[i,:] = conditionalDraw(Np,Nit,S_2N,solve,eps_o,mu_m_given_o,
        ind_obs,ind_mis,M,PCGalgo,tol=tol)

    return condDraws





# ==============================================================================
def buildSparseCov2(R,p,N,form=None,taper = 'Wendland2'):
    """
    This function constructs a sparse matrix which is a tappered, approximate
    version of the covariance matrix of a stationary process of autocovariance
    function R and size N x N.

    Parameters
    ----------
    R : numpy array
        input autocovariance functions at each lag (size N)
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function.
    N : scalar integer
        Size of the complete data vector
    form : character string (or None), optional
        storage format of the sparse matrix (default is None)
    taper : string
        type of taper function to smoothly decrease the tapered covariance
        function down to zero.


    Returns
    -------
    C_tap : scipy.sparse matrix
        Tappered covariance matrix (size N x N) with p non-zero diagonals.


    """
    k = np.array([])
    values = list()

    tap = taperCovariance(np.arange(0,N),p,taper = taper)
    R_tap = R[0:N]*tap

    # calculate the rows and columns indices of the non-zero values
    # Do it diagonal per diagonal
    for i in range(p+1):

        Rf = np.ones(N-i)*R_tap[i]
        values.append(Rf)

        k = np.hstack((k,i))



        # Symmetric values
        if i!=0 :
            values.append(Rf)
            k = np.hstack((k,-i))


    return sparse.diags(values, k.astype(int), format=form)

# ==============================================================================
def conjGradImputation(s,A,beta,M,S_2N,R,p,Nit,PCGalgo,tol=1e-7):
    """
    Function performing universal kriging with the model y = A beta + n
    using the conjugate gradient algorithm to solve for C_OO x = b.
    But this imputation function can be generalized to any solver algorithm.
    See "imputation" function for a more general formulation

    Parameters
    ----------
    s : array_like (size N)
        Input signal array
    A : numpy array (size N x K)
        design matrix (contains partial derivatives of signal wrt parameters)
    beta : array_like (size K)
        regression parameter vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    S_2N : numpy array (size P)
        spectrum array: value of the unilateral PSD times fs
    R : numpy array (size P)
        tappered autocovariance array: value of the autocovariance at all lags
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the conjugate
        gradients.
    Nit : scalar integer
        maximum number of iterations for the conjugate gradient algorithm
    tol : scalar float
        tolerance criterium to stop the PCG algorithm

    Returns
    -------
    z : float or ndarray
        conditional expectation of the data vector
    solve : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    """



    N = len(s)
    # Preconditionning : use sparse matrix
    # ======================================================================
    ind_mis = np.where( M==0 )[0]#.astype(int)
    ind_obs = np.where( M!=0 )[0]#.astype(int)


    # Deterministic model for observed and missing data
    # ======================================================================
    A_obs =  A[M==1,:]
    A_mis =  A[M==0,:]
    beta_est = beta.T

    # Preconditionner
    # ======================================================================
    solve = computePrecond(R,M,p=p,taper = 'Wendland2')

    # Residuals of observed data - model
    eps = s[ind_obs]-np.dot(A_obs,beta_est)

    # Conjugate gradients
    # ======================================================================
    # Solve the linear system C_oo x = eps
    #u,sr = conjugateGradientSolvePrecond(np.zeros(len(ind_obs)),eps,Coo_func,Nit,np.std(eps),solve)
    N_o = len(ind_obs)
    x0 = np.zeros(N_o)
    u = PCGsolve(ind_obs,M,S_2N,eps,x0,tol,Nit,solve,PCGalgo)

    # Calculate the conditional expectation of missing data
    x = np.dot(A_mis,beta) +  matVectProd(u,ind_obs,ind_mis,M,S_2N)

    z = np.zeros(N)
    z[ind_obs] = s[ind_obs]
    z[ind_mis] = x


    return z,solve


# ==============================================================================
def taperCovariance(h,theta,taper = 'Wendland1',tau=10):
    """
    Function calculating a taper covariance that soothly goes to zero when h
    goes to theta. This taper is to be mutiplied by the estimated covariance R
    of the process, to discard correlations larger than theta.

    Ref : Reinhard FURRER, Marc G. GENTON, and Douglas NYCHKA,
    Covariance Tapering for Interpolation of Large Spatial Datasets,
    Journal of Computational and Graphical Statistics, Volume 15, Number 3,
    Pages 502–523,2006

    Parameters
    ----------
    h : numpy array of size N
        lag
    theta : scalar float
        taper parameter
    taper : string {'Wendland1','Wendland2','Spherical'}
        name of the taper function


    Returns
    -------
    C_tap : numpy array
        the taper covariance function values (vector of size N)
    """

    ii = np.where(h<=theta)[0]

    if taper == 'Wendland1' :

        C = np.zeros(len(h))

        C[ii] = (1.-h[ii]/np.float(theta))**4 * (1 + 4.*h[ii]/np.float(theta))

    elif taper == 'Wendland2' :

        C = np.zeros(len(h))

        C[ii] = (1-h[ii]/np.float(theta))**6 * (1 + 6.*h[ii]/theta + \
        35*h[ii]**2/(3.*theta**2))

    elif taper == 'Spherical' :

        C = np.zeros(len(h))

        C[ii] = (1-h[ii]/np.float(theta))**2 * (1 + h[ii]/(2*np.float(theta)) )

    elif taper == 'Hanning':
        C = np.zeros(len(h))
        C[ii] = (np.hanning(2*theta)[theta:2*theta])**2


    elif taper == 'Gaussian' :

        C = np.zeros(len(h))
        sigma = theta/5.
        C[ii] = np.sqrt( 1./(2*np.pi*sigma**2) ) * np.exp( - h[ii]**2/(2*sigma**2) )

    elif taper == 'rectSmooth':

        C = np.zeros(len(h))
        #tau = np.int(theta/3)


        C[h<=theta-tau] = 1.
        jj = np.where( (h>=theta-tau) & (h<=theta) )[0]
        C[jj] = 0.5*(1 + np.cos( np.pi*(h[jj]-theta+tau)/np.float(tau) ) )

    elif taper == 'modifiedWendland2':


        C = np.zeros(len(h))

        C[h<=theta-tau] = 1.
        jj = np.where( (h>=theta-tau) & (h<=theta) )[0]

        C[jj] = (1-(h[jj]-theta+tau)/np.float(tau))**6 * (1 + \
        6.*(h[jj]-theta+tau)/tau + 35*(h[jj]-theta+tau)**2/(3.*tau**2))



    return C


# ==============================================================================
def computePrecond(R,M,p=10,ptype = 'sparse',taper = 'Wendland2'):
    """
    For a given mask and a given PSD function, the function computes the linear
    operator x = C_OO^{-1} b for any vector b, where C_OO is the covariance matrix
    of the observed data (at points where M==1).


    Parameters
    ----------
    R : numpy array
        input autocovariance functions at each lag (size N)
    M : numpy array
        mask vector
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the conjugate
        gradients.
    ptype : string {'sparse','circulant'}
        specifies the type of preconditioner matrix (sparse approximation of the
        covariance or circulant approximation of the covariance)
    taper : string {'Wendland1','Wendland2','Spherical'}
        name of the taper function. This argument is only used if ptype = 'sparse'

    Returns
    -------
    solve : sparse.linalg.factorized instance
        preconditionner operator, calculating P x for all vectors x

    """


    N = len(M)

    # ======================================================================
    ind_mis = np.where( M==0 )[0]#.astype(int)
    ind_obs = np.where( M!=0 )[0]#.astype(int)
    # Calculate the covariance matrix of the complete data

    if ptype == 'sparse':
        # Preconditionning : use sparse matrix
        C = buildSparseCov2(R,p,N,form="csc",taper = taper)
        # Calculate the covariance matrix of the observed data
        C_temp = C[:,ind_obs]
        #C_OO = C_temp[ind_obs,:]
        #del C_temp
        #del C
        # Preconditionner
        #solve =  sparse.linalg.factorized(C_OO)
        solve =  sparse.linalg.factorized(C_temp[ind_obs,:])

    elif ptype == 'circulant':

        S = np.real(fft(R))
        N_fft = len(S)
        solve = lambda v: np.real(ifft(fft(v,N_fft)/S,len(v)))

    return solve
