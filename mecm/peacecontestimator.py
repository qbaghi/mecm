#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# This code provides routines for PSD estimation using a peace-continuous model
# that assumes that the logarithm of the PSD is linear per peaces.
import numpy as np
from scipy import linalg as LA
from scipy import sparse
from mecm import mecm
from mecm import noise


def findclosestfrequencies(f_in, f_target):
    """
    Function finding the closest frequencies to f_target in the vector f_in

    Parameters
    ----------
    f_in : numpy array of size N
        input frequency vector
    f_target : numpy array of size N_est
        target frequency vector

    Returns
    -------

    i_out : numpy array size < N_est
        indices where f_in is close to f_target
    """

    i_out = np.array([])

    for i in range(len(f_target)):

        i_found = np.argmin( np.abs(f_target[i]-f_in) )

        if i_found not in i_out :
            i_out = np.append( i_out, i_found )

    return i_out.astype(np.int)

def logpsdfunction(a,b,x_grid,x):
    """
    PSD function whose logarithm is peacewise linear


    y_k = a_j * x_k + b_j for xk in [x_j , x_j+1]

    where

    y_k = log(S(fk))
    x_k = log(fk)


    Parameters
    ----------
    a : numpy array of size J-1
        slope coefficients of the PSD model
    b : numpy array of size J-1
        local intercept coefficients of the PSD model
    x_grid : array_like
        frequencies defining each frequency segment
    x : numpy array of size N
        logarithm of Fourier frequencies where to compute the PSD


    Returns
    -------

    logS : numpy array size N
        PSD values calculated at frequencies f
    """

    N = len(f)
    logS = np.zeros(N,dtype = np.float64)
    J = len(inds)
    y_list = []
    # For frequencies below the first node of the grid
    i0 = np.where(x<x_grid[0])
    if i0[0]>0:
        y_list.append( a[0]*x[x<x_grid[0]] + b[0] )


    y_list.extend( [a[j]*x[(x>=x_grid[j])&(x<x_grid[j+1])] + b[j] for j in range(J-1)] )

    # If there is a frequency that is equal to or larger than the last node of the grid
    iJ = np.where(x>=x_grid[J-1])
    if len(iJ[0])>0:
        y_list.append( a[J-1]*x[iJ] + b[J-1] )


    # for j in range(J-1):
    #
    #     logS[inds[j]] = a[j]*x[inds[j]] + b[j]

    # # Last frequency
    # inds = np.where( f==fq[J-1] )
    # logS[inds] = a[J-1]*x[inds] + b[J-1]

    return np.concatenate(y_list)


def logpsdfunction_fourier(a,b,inds,x):
    """
    PSD function whose logarithm is peacewise linear


    y_k = a_j * x_k + b_j for xk in [x_j , x_j+1]

    where

    y_k = log(S(fk))
    x_k = log(fk)


    Parameters
    ----------
    a : numpy array of size J-1
        slope coefficients of the PSD model
    b : numpy array of size J-1
        local intercept coefficients of the PSD model
    inds : list of array_like
        list of frequencies indices correponding to each frequency segment
    x : numpy array of size N
        logarithm of Fourier frequencies in increasing order


    Returns
    -------

    logS : numpy array size N
        PSD values calculated at frequencies f
    """

    N = len(f)
    logS = np.zeros(N,dtype = np.float64)
    J = len(inds)

    y_list = [a[j]*x[inds[j]] + b[j] for j in range(J-1)]

    # for j in range(J-1):
    #
    #     logS[inds[j]] = a[j]*x[inds[j]] + b[j]

    # # Last frequency
    # inds = np.where( f==fq[J-1] )
    # logS[inds] = a[J-1]*x[inds] + b[J-1]

    return np.concatenate(y_list)


def loglike(logPer,a,b,inds,x):
    """

    logarithm of the likelihood


    Parameters
    ----------
    logPer : array_like
        logarithm of the periodogram computed at Fourier frequencies
    a : numpy array of size J-1
        slope coefficients of the PSD model
    b : numpy array of size J-1
        local intercept coefficients of the PSD model
    inds : list of array_like
        list of frequencies indices correponding to each frequency segment
    x : numpy array of size N
        logarithm of frequencies where to compute the PSD, in increasing order

    """

    m = logpsdfunction_fourier(a,b,inds,x)

    res = np.log(Per-m)

    return np.sum( -np.exp(-res) + res )


def loglike_grad(logPer,a,b,inds,x):
    """

    Gradient of the log-PSD with respect to the coefficients theta = [a,b]


    Parameters
    ----------
    logPer : array_like
        logarithm of the periodogram computed at Fourier frequencies
    a : numpy array of size J-1
        slope coefficients of the PSD model
    b : numpy array of size J-1
        local intercept coefficients of the PSD model
    inds : list of array_like
        list of frequencies indices correponding to each frequency segment
    x : numpy array of size N
        logarithn of frequencies where to compute the PSD, in increasing order



    Returns
    -------

    ll_grad : numpy array size N
        values of the derivatives of the log-PSD calculated at frequencies f

    """

    # Derivatives with respect to a
    J = len(inds)
    res = logPer - logpsdfunction_fourier(a,b,inds,x)
    dll_a = np.zeros(J-1,dtype = np.float64)
    dll_a = np.array([np.sum( - x[inds[j]] *( np.exp(-res[inds[j]]) + 1)) for j in range(J-1)] )

    # Derivatives with respect to b
    dll_b = np.zeros(J-1,dtype = np.float64)
    dll_a = np.array([np.sum( - ( np.exp(-res[inds[j]]) + 1)) for j in range(J-1)] )

    return np.concatenate((dll_a,dll_b))


def loglike_hess(logPer,a,b,inds,x,full = True):
    """

    Hessian matrix of the log-PSD with respect to the coefficients theta = [a,b]


    Parameters
    ----------
    logPer : array_like
        logarithm of the periodogram computed at Fourier frequencies
    a : numpy array of size J-1
        slope coefficients of the PSD model
    b : numpy array of size J-1
        local intercept coefficients of the PSD model
    inds : list of array_like
        list of frequencies indices correponding to each frequency segment
    x : numpy array of size N
        logarithn of frequencies where to compute the PSD, in increasing order
    full : boolean
        if True, the result is given as a full matrix. If faulse, only
        the diagonals of the Hessian matrix blocks are given



    Returns
    -------

    ll_grad : numpy array size N
        values of the derivatives of the log-PSD calculated at frequencies f

    """

    # Derivatives with respect to a
    J = len(inds)
    res = logPer - logpsdfunction_fourier(a,b,inds,x)

    diag_ddll_aa = np.array([ np.sum(- x[inds[j]]**2 * np.exp(-res[inds[j]])) for j in range(J-1) ])
    diag_ddll_ab = np.array([ np.sum(- x[inds[j]] * np.exp(-res[inds[j]])) for j in range(J-1) ])
    diag_ddll_bb = np.array([ np.sum(- np.exp(-res[inds[j]])) for j in range(J-1) ])

    if not full:

        return diag_ddll_aa,diag_ddll_ab,diag_ddll_bb

    else:

        #H = np.block([[np.diag(diag_ddll_aa)        , diag_ddll_ab],
        #              [np.diag(diag_ddll_ab).conj() , np.diag(diag_ddll_bb)] ])

        k = np.array([0])
        Haa = sparse.diags(diag_ddll_aa, k.astype(int), format="csc")
        Hab = sparse.diags(diag_ddll_ab, k.astype(int), format="csc")
        Hbb = sparse.diags(diag_ddll_bb, k.astype(int), format="csc")

        H = sparse.bmat([[Haa          , Hab],
                         [Hab.conj().T , Hbb]])


        return H


def psd_leastsq(inds,x,y):
    """

    compute the coefficients a,b of the peace-wise linear PSD

    such that for x in [xj,xj+1]

    y = a_j * x + b_j

    where

    y = log(S)
    x = log(f)



    Parameters
    ----------

    inds : list of array_like
        list of frequencies indices correponding to each frequency segment
    x_grid : array_like
        vector of size J containing the frequencies of logarithmic grid
    x : numpy array of size N
        logarithm of Fourier frequencies where in increasing order
    y : array_like
        logarithm of the periodogram computed at Fourier frequencies

    """

    # First frequency
    A = np.array([x[inds[0]],np.ones(len(inds[0]))])
    # Compute a[0] and b[0] beta = [a,b].T
    beta = LA.inv(A.T.dot(A)).dot(A.T.dot(y[inds[0]]))

    a_list = []
    b_list = []
    a_list.append(beta[0])
    b_list.append(beta[1])

    for j in range(1,len(inds)):

        y_reg = y[inds[j]] - x[j]*a_list[j-1] - b_list[j-1]
        x_reg = x[inds[j]] - x[j]

        a_list.append( np.sum( x_reg*y_reg ) / np.sum( x_reg**2 ) )
        b_list.append( b_list[j-1] - x[j]*(a_list[j]-a_list[j-1]) )


    return np.array(a_list),np.array(b_list)


class PSD_estimate(object):

    def __init__(self,N_est,Npoints):

        self.N_est = N_est

        # self.xn = np.zeros(N)
        # self.xn[0]= -np.inf


        # Logarithmic grid of frequencies
        #self.f_est = np.zeros(N_est+1)
        self.f_est = 1./Npoints * np.exp( np.log(Npoints/2.)*np.arange(0,N_est)/(N_est-1))
        self.x_est = np.log(f_est)

        self.a = []
        self.b = []

        # Indices of frequency segments
        self.inds = []

    def estimate_lsqr(self,x):
        """
        Estimate the PSD parameters using least squares

        """

        if ( (self.N is None) | (len(x) != self.N) ):
            self.N = len(x)
            self.f = np.fft.fftfreq(self.N)
            self.n = np.int((self.N-1)/2.)
            self.inds = [np.where( (self.fn>=self.f_est[j]) & (self.fn<self.f_est[j+1]) ) for j in range(N_est)]
            self.fn = self.f[1:n+1]
            self.xn = np.log(fn)


        Per = mecm.periodogram(x,self.N,wind = 'hanning')
        self.a,self.b = psd_leastsq(self.inds,self.xn,np.log(Per))

    def calculate(self,arg):
        """
        Calculate the PSD at frequencies x

        """

        if type(arg)==np.int:
            N = arg
            f = np.fft.fftfreq(N)
            n = np.int( (N-1)/2.)

            # Symmetrize the estimates
            if N % 0 : # if N is even
                # Compute PSD from f=0 to f = fs/2
                m_N = logpsdfunction(self.a,self.b,self.x_grid,np.log(np.abs(f[0:n+2])) )
                m_N_sym = np.concatenate((m_N[0:n+1],m_N[1:n+2][::-1]))

            else: # if N is odd
                m_N = logpsdfunction(self.a,self.b,self.x_grid,np.log(np.abs(f[0:n+1])) )
                m_N_sym = np.concatenate((m_N[0:n+1],m_N[1:n+1][::-1]))


            S_N_sym = np.exp(m_N_sym)


        elif type(arg) == np.ndarray:
            f = arg[:]
            m_est = logpsdfunction(self.a,self.b,self.x_grid,np.log(f))
            S_N_sym = np.exp(m_est_interp)

        else:
            raise TypeError("Argument must be integer or ndarray")


        return S_est_N_sym



if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import tdi
    fftwisdom.load_wisdom()

    N = 2**19

    t = np.arange(0,N)
    f = np.fft.fftfreq(N)

    f[0] = f[1]
    scale = 1e-21
    S = tdi.noisepsd_AE(np.abs(f), model='Proposal')/scale**2
    f2N = np.fft.fftfreq(2*N)
    f2N[0] = f2N[1]
    S_2N = tdi.noisepsd_AE(np.abs(f2N), model='Proposal')/scale**2

    # ==========================================================================
    # Generate data
    print("Generating data...")
    b,S,Noise_TF = noise.generateNoiseFromDSP(np.sqrt(S[0:N]),fs,myseed = 1354561)
    y = np.real(b[0:N])
    print("Data generated")

    N_est = 100
    PSD = PSD_estimate(N_est,N)
    PSD.estimate_lsqr(y)
