#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# This code provides routines for PSD estimation using a peace-continuous model
# that assumes that the logarithm of the PSD is linear per peaces.




def findclosestfrequencies(f_in,f_target):
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



def logpsdfunction(a,b,fq,f):
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
    fq : numpy array of size J
        frequency vector locating the bounds of the segments
    f : numpy array of size N
        frequencies where to compute the PSD, in increasing order


    Returns
    -------

    logS : numpy array size N
        PSD values calculated at frequencies f
    """

    N = len(f)
    logS = np.zeros(N,dtype = np.float64)
    J = len(fq)
    x = np.log(f)

    for j in range(J-1):

        inds = np.where( (f>=fq[j]) & (f<fq[j+1]) )
        logS[inds] = a[j]*x[inds] + b[j]

    # Last frequency
    inds = np.where( f==fq[J-1] )
    logS[inds] = a[J-1]*x[inds] + b[J-1]

    return logS


def loglike(logPer,a,b,fq,f):
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
    fq : numpy array of size J
        frequency vector locating the bounds of the segments
    f : numpy array of size N
        frequencies where to compute the PSD, in increasing order

    """

    m = logpsdfunction(a,b,fq,f)

    res = np.log(Per-m)

    return np.sum( -np.exp(-res) + res )

def loglike_grad(logPer,a,b,fq,f):
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
    fq : numpy array of size J
        frequency vector locating the bounds of the segments
    f : numpy array of size N
        frequencies where to compute the PSD, in increasing order


    Returns
    -------

    ll_grad : numpy array size N
        values of the derivatives of the log-PSD calculated at frequencies f

    """

    # Derivatives with respect to a
    dll_a = np.zeros(J-1,dtype = np.float64)
    J = len(fq)
    x = np.log(f)

    for j in range(J-1):
        inds = np.where( (f>=fq[j]) & (f<fq[j+1]) )
        dll_a[j] = np.sum( -  )




    return ll_grad


def jac(a,fq,f,z_fft):
    """
    Jacobian of the log-likelihood
    loglike = -1/2 * Sum_k ( log Sk + np.abs(zk)^2 /Sk  )

    """

    J = len(a)

    jacob = np.zeros(J)

    for j in range(J):

        jacob[j] = np.sum( dlogSda(j,fq,f) * (1 - np.abs(z_fft)**2 / psdfunction(a,fq,f) ) )

    return jacob




def d2logSdamdaj(m,j,fq,f):
    """
    Second derivatives of the log-PSD with respect to the coefficients a's


    Parameters
    ----------
    m : scalar integer
        index of the first derivative d2 logS / dajdam
    j : scalar integer
        index of the second derivative d2 logS / dajdam
    fq : numpy array of size J
        frequency vector locating the bounds of the segments
    f : numpy array of size N
        frequencies where to compute the PSD, in increasing order


    Returns
    -------

    dlogSda : numpy array size N
        values of the derivatives of the log-PSD calculated at frequencies f


    """

    N = len(f)
    J = len(fq)
    Hmj = np.zeros(N)

    f1 = np.max([fq[j-1],fq[m-1]])
    f2 = np.min([fq[j-1],fq[m+1]])

    ik = np.where((f>=f1) & (f<f2))

    if ik!=():
        inds = ik[0]
        Hmj[inds] = dlogSda[m,fq,f[inds]]*dlogSda[j,fq,f[inds]]
