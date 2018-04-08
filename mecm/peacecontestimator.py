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



def psdfunction(a,fq,f):
    """
    PSD function whose logarithm is peacewise linear

    Parameters
    ----------
    a : numpy array of size J
        coefficients of the PSD model
    fq : numpy array of size J
        frequency vector locating the bounds of the segments
    f : numpy array of size N
        frequencies where to compute the PSD, in increasing order


    Returns
    -------

    S : numpy array size N
        PSD values calculated at frequencies f
    """

    N = len(f)
    J = len(fq)
    logS = np.zeros(N)

    ind0 = np.int( np.min( np.where(fq <= f[0])  ) )
    ind1 = np.int( np.max( np.where(fq > f[N-1]) ) )

    i=0

    for j in range(ind0,ind1):

        fk = f[ (f>=fq[j]) & (f<fq[j+1]) ]

        logS[i:i+len(fk)] = a[j] + (a[j+1]-a[j])*(fk-fq[j])/(fq[j+1]-fq[j])

        i = i + len(fk)

    return np.exp(logS)


def dlogSda(j,fq,f):
    """
    Derivatives of the log-PSD with respect to the coefficients a's


    Parameters
    ----------
    j : scalar integer
        index of the derivative d logS / daj
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
    dlogSda = np.zeros(N)

    if j>0:
        k1 = np.where( (fq[j-1]<=f) & (f < fq[j]) )[0]
        dlogSda[k1] = (f[k1]-fq[j-1])/(fq[j]-fq[j-1])
    if j<J-1:
        k2 = np.where( (fq[j]<=f) & (f < fq[j+1]) )[0]
        dlogSda[k2] = (fq[j+1]-f[k1])/(fq[j+1]-fq[j])

    return dlogSda


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
