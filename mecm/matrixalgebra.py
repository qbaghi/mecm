#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
from __future__ import print_function, division, absolute_import

# ==============================================================================
# This code provides algorithms for solving large sparse linear algebra problems
# ==============================================================================
import numpy as np
from numpy import linalg as LA
from scipy import sparse
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
#from numba import jit
# For parallel loops
#from numba import njit, prange
pyfftw.interfaces.cache.enable()


import threading
from timeit import repeat

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
#@jit#(nopython=True, nogil=True)


class PCG:

    def __init__(self, a_func, nit, stp, precond):

        self.a_func = a_func
        self.precond = precond

        self.nit = nit
        self.stp = stp

        # Initialization of residual vector
        self.sr = np.zeros(nit + 1)
        self.k = 0
        # Intialization of the solution
        self.b_norm = 1
        self.x = 0
        self.k = 0
        self.p = []
        self.z = []

    def initialize(self, x0, b):

        # Initialization of residual vector
        self.sr = np.zeros(self.nit + 1)
        # sz = np.zeros(Nit+1)
        self.k = 0

        # Intialization of the solution
        self.b_norm = LA.norm(b)
        self.x = x0
        self.r = b - self.a_func(x0)
        self.z_hat = self.precond(self.r)
        self.z = self.z_hat[:]
        self.p = np.zeros(len(self.r))
        self.p[:] = self.z
        self.sr[0] = LA.norm(self.r)

    def pcg_update(self, verbose=False):

        ap = self.a_func(self.p)
        q = self.precond(ap)
        a = np.sum(np.conj(self.z) * self.z_hat) / np.sum(np.conj(q) * self.z_hat)

        x_12 = self.x + a * self.p
        r_12 = self.r - a * ap
        z_12 = self.z - a * q

        az_12 = self.a_func(z_12)
        s_12 = self.precond(az_12)

        w = np.sum(np.conj(z_12) * s_12) / np.sum(np.conj(s_12) * s_12)

        self.x = x_12 + w * z_12
        self.r = r_12 - w * az_12
        z_new = z_12 - w * s_12

        b = a / w * np.sum(np.conj(z_new) * self.z_hat) / np.sum(np.conj(self.z) * self.z_hat)

        self.p = z_new + b * (self.p - w * q)

        self.z[:] = z_new
        self.sr[self.k + 1] = LA.norm(self.r)

        # increment
        self.k = self.k + 1

        if verbose:
            if self.k % 20 == 0:
                print('PCG Iteration ' + str(self.k) + ' completed')
                print('Residuals = ' + str(self.sr[self.k]) + ' compared to criterion = '+str(self.stp*self.b_norm))

    def solve(self, x0, b, verbose=False):

        self.initialize(self, x0, b)

        [self.pcg_update(verbose=verbose) for k in range(self.nit)]

        return self.x, self.sr


def precondBiCGSTAB(x0,b,A_func,Nit,stp,P,z0_hat = None,verbose=True):
    """
    Function solving the linear system
    x = A^-1 b
    with preconditioned bi-conjuage gradient algorithm


    Parameters
    ----------
    x0 : numpy array of size No
        initial guess for the solution (can be zeros(No) array)
    b : numpy array of size No
        observed vector (right hand side of the system)
    A_func: linear operator
        linear function of a vector x calculating A*x
    Nit : scalar integer
        number of maximal iterations
    stp : scalar float
        stp: stopping criteria
    P : scipy.sparse. operator
        preconditionner operator, calculating Px for all vectors x
    z0_hat : array_like (size N)
        first guess for solution, optional (default is None)



    Returns
    -------
    x : the reconstructed vector (numpy array of size N)
    """

    # Default first guess
    #z0_hat = None
    # Initialization of residual vector
    sr = np.zeros(Nit+1)
    #sz = np.zeros(Nit+1)
    k=0

    # Intialization of the solution
    b_norm = LA.norm(b)
    x = np.zeros(len(x0))
    x[:] = x0
    r = b - A_func(x0)
    z = P(r)

    p = np.zeros(len(r))
    p[:] = z

    z_hat = np.zeros(len(z))
    if z0_hat is None:
        z_hat[:] = z

    else :
        z_hat[:] = z0_hat

    sr[0] = LA.norm(r)


    # Iteration
    while (k<Nit) & (sr[k]>stp*b_norm):

        # Ap_k-1
        Ap = A_func(p)
        # Mq_k-1=Ap_k-1
        q = P(Ap)

        a = np.sum(np.conj(z)*z_hat) / np.sum(np.conj(q)*z_hat)

        x_12 = x + a*p
        r_12 = r - a*Ap
        z_12 = z - a*q

        Az_12 = A_func(z_12)
        s_12 = P(Az_12)

        w = np.sum(np.conj(z_12)*s_12) / np.sum( np.conj(s_12)*s_12)

        x = x_12 + w*z_12
        r = r_12 - w*Az_12
        z_new = z_12 - w*s_12

        b = a/w * np.sum( np.conj(z_new)*z_hat ) / np.sum( np.conj(z)*z_hat )

        p = z_new + b*(p - w*q)

        z[:] = z_new
        sr[k+1] = LA.norm(r)
        #zr[k+1] = LA.norm(z_new)

        # increment
        k = k+1

        if verbose:
            if k%20 == 0 :
                print('PCG Iteration ' + str(k) + ' completed')
                print('Residuals = ' + str(sr[k]) + ' compared to criterion = '+str(stp*b_norm))

    print("Preconditioned BiCGSTAB algorithm ended with:")
    print(str(k) + "iterations." )

    if sr[k-1]>stp*b_norm:
        print("Attention: Preconditioned BiCGSTAB algorithm ended \
        without reaching the specified convergence criterium. Check quality of \
        reconstruction.")
        print("Current criterium: " +str(sr[k-1]/b_norm) + " > " + str(stp))

    return x,sr#,sz


#@jit('void(float64[:,:], float64[:], float64[:,:],float64[:](float64[:]),float64)', nopython=True, nogil=True)
#@jit(nopython=True, nogil=True)
#@jit(nogil=True)
def innerPrecondBiCGSTAB(result,x0,B,A_func,Nit,stp,P,pcg_algo):
    """
    Inner function of the multithreading process.
    This is used to solve the linear system
    X = A^-1 B where B is a matrix, by applying the preconditioned stabilized
    bi-conjuage gradient algorithm to each rows of B


    Parameters
    ----------
    result : 2D numpy array
        the result X that will be updated (can be empty at the beginning)
    x0 : numpy array of size No
        initial guess for the solution (can be zeros(No) array)
    B : 2D numpy array
        Matrix of observed vectors (right hand side of the system)
    A_func: linear operator
        linear function of a vector x calculating A*x
    Nit : scalar integer
        number of maximal iterations
    stp : scalar float
        stp: stopping criteria
    P : scipy.sparse. operator
        preconditionner operator, calculating Px for all vectors x
    z0_hat : array_like (size N)
        first guess for solution, optional (default is None)



    Returns
    -------
    Nothing, acts on the argument "result"

    """

    if pcg_algo == 'mine':
        for k in range(result.shape[1]):
            # With my code:
            result[:,k],sr = precondBiCGSTAB(x0,B[:,k],A_func,Nit,stp,P)
            print("PCG complete for parameter "+str(k))


    elif 'scipy' in pcg_algo:
        for k in range(result.shape[1]):
            tol_eff = np.min([stp,stp*LA.norm(B[:,k])])
            if (pcg_algo == 'scipy') | (pcg_algo == 'scipy.bicgstab'):
                result[:,k],info = sparse.linalg.bicgstab(A_func, B[:,k], x0=x0,
                tol=tol_eff,maxiter=Nit,M=P,callback=None)
            elif (pcg_algo == 'scipy.bicg'):
                result[:,k],info = sparse.linalg.bicg(A_func, B[:,k], x0=x0,
                tol=tol_eff,maxiter=Nit,M=P,callback=None)
            elif (pcg_algo == 'scipy.cg'):
                result[:,k],info = sparse.linalg.cg(A_func, B[:,k], x0=x0,
                tol=tol_eff,maxiter=Nit,M=P,callback=None)
            elif (pcg_algo == 'scipy.cgs'):
                result[:,k],info = sparse.linalg.cgs(A_func, B[:,k], x0=x0,
                tol=tol_eff,maxiter=Nit,M=P,callback=None)
            elif (pcg_algo == 'scipy.gmres'):
                result[:,k],info = sparse.linalg.cgs(A_func, B[:,k], x0=x0,
                tol=tol_eff,maxiter=Nit,M=P)
            elif (pcg_algo == 'scipy.lgmres'):
                result[:,k],info = sparse.linalg.cgs(A_func, B[:,k], x0=x0,
                tol=tol_eff,maxiter=Nit,M=P)
            else:
                raise ValueError("Unknown PCG algorithm name")
            print("PCG complete for parameter "+str(k) + ", with exit status:")
            printPCGstatus(info)
            print("Value of || A * x - b ||/||b|| at exit:")
            print(str(LA.norm(result[:,k]-B[:,k])/LA.norm(B[:,k])))






def printPCGstatus(info):
    """
    Function that takes the status result of the scipy.sparse.linalg.bicgstab
    algorithm and print it in an understandable way.
    """
    if info == 0:
        print("successful exit!")
    elif info>0 :
        print("convergence to tolerance not achieved")
        print("number of iterations: " + str(info))
    elif info<0 :
        print("illegal input or breakdown.")


def make_singlethread(inner_func):
    """
    Run the precondBiCGSTAB algorithm inside a single thread.
    """
    def func(*args):
        # Number of linear systems to inverse
        length = args[1].shape[1]
        # Size of the ouput vectors that are solutions
        No = len(args[0])
        result = np.empty((No,length),dtype=np.float64)
        matprecondBiCGSTAB(result,*args)
        return result
    return func


def make_multithread(inner_func, numthreads):
    """
    Run the precondBiCGSTAB algorithm inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(*args):
        # Number of linear systems to inverse
        length = args[1].shape[1]
        # Size of the ouput vectors that are solutions
        No = len(args[0])
        result = np.empty((No,length),dtype=np.float64)
        #args = (result,) + args

        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        # arguments are x0,B,A_func,Nit,stp,P we only change B
        chunks = [[result[:,i * chunklen:(i + 1) * chunklen],args[0],
        args[1][:,i * chunklen:(i + 1) * chunklen],args[2],
        args[3],args[4],args[5],args[6]] for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt


def matPrecondBiCGSTAB(x0,B,A_func,Nit,stp,P,pcg_algo = 'scipy', nthreads=4):
    """
    Function that solves the linear system
    X = A^-1 B where B is a matrix, by applying the preconditioned stabilized
    bi-conjuage gradient algorithm to each rows of B
    It uses multithreading.


    Parameters
    ----------
    result : 2D numpy array
        the result X that will be uptdated (can be empty at the beginning)
    x0 : numpy array of size No
        initial guess for the solution (can be zeros(No) array)
    B : 2D numpy array
        Matrix of observed vectors (right hand side of the system)
    A_func: linear operator
        linear function of a vector x calculating A*x
    Nit : scalar integer
        number of maximal iterations
    stp : scalar float
        stp: stopping criteria
    P : scipy.sparse. operator
        preconditionner operator, calculating Px for all vectors x
    z0_hat : array_like (size N)
        first guess for solution, optional (default is None)



    Returns
    -------
    X : 2D numpy array
        solution of linear systems (size No x K)

    """

    if nthreads > 1 :
        func_PrecondBiCGSTAB_mt = make_multithread(innerPrecondBiCGSTAB, nthreads)
        res = func_PrecondBiCGSTAB_mt(x0,B,A_func,Nit,stp,P,pcg_algo)

    elif nthreads == 1 :
        res = np.empty((len(x0),B.shape[1]),dtype=np.float64)
        innerPrecondBiCGSTAB(res,x0,B,A_func,Nit,stp,P,pcg_algo)


    return res



# ==============================================================================
def matVectProd(y_in,ind_in,ind_out,M,S_2N):
    """
    Linear operator that calculate Com y_in assuming that we can write:

    Com = M_o F* Lambda F M_m^T

    Parameters
    ----------
    y_in : numpy array
        input data vector
    ind_in : array_like
        array or list containing the chronological indices of the values
        contained in the input vector in the complete data vector
    ind_out : array_like
        array or list containing the chronological indices of the values
        contained in the output vector in the complete data vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    N : scalar integer
        Size of the complete data vector
    S_2N : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    y_out : numpy array
        y_out = Com * y_in transformed output vector of size N_out


    """

    # calculation of the matrix product Coo y, where y is a vector
    y = np.zeros(len(M)) #+ 1j*np.zeros(N)
    y[ind_in] = y_in

    N_fft = len(S_2N)

    #return np.real( ifft( S_2N * fft(M*y,N_fft) )[ind_out] )
    return np.real( ifft( S_2N * fft(y,N_fft) )[ind_out] )

# ==============================================================================
def matmatProd(A_in,ind_in,ind_out,M,S_2N):
    """
    Linear operator that calculates Coi * A_in assuming that we can write:

    Com = M_o F* Lambda F M_m^T

    Parameters
    ----------
    y_in : 2D numpy array
        input matrix of size (N_in x K)
    ind_in : array_like
        array or list containing the chronological indices of the values
        contained in the input vector in the complete data vector (size N_in)
    ind_out : array_like
        array or list containing the chronological indices of the values
        contained in the output vector in the complete data vector (size N_out)
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    N : scalar integer
        Size of the complete data vector
    S_2N : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    A_out : numpy array
        Matrix (size N_out x K) equal to A_out = Com * A_in

    """
    N_in = len(ind_in)
    N_out = len(ind_out)

    (N_in_A,K) = np.shape(A_in)

    if N_in_A != N_in :
        raise TypeError("Matrix dimensions do not match")

    A_out = np.empty((N_out,K),dtype = np.float64)

    for j in range(K):
        A_out[:,j] = matVectProd(A_in[:,j],ind_in,ind_out,M,S_2N)

    return A_out


# ==============================================================================
def covLinearOp(ind_in,ind_out,M,S_2N):
    """
    Construct a linear operator object that computes the operation C * v
    for any vector v, where C is a covariance matrix.


    Linear operator that calculate Com y_in assuming that we can write:

    Com = M_o F* Lambda F M_m^T

    Parameters
    ----------
    y_in : numpy array
        input data vector
    ind_in : array_like
        array or list containing the chronological indices of the values
        contained in the input vector in the complete data vector
    ind_out : array_like
        array or list containing the chronological indices of the values
        contained in the output vector in the complete data vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    S_2N : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    Coi_op : scipy.sparse.linalg.LinearOperator instance
        linear opreator that computes the vector y_out = Com * y_in for any
        vector of size N_in

    """

    C_func = lambda x: matVectProd(x,ind_in,ind_out,M,S_2N)
    CH_func = lambda x: matVectProd(x,ind_out,ind_in,M,S_2N)
    Cmat_func = lambda X: matmatProd(X,ind_in,ind_out,M,S_2N)


    N_in = len(ind_in)
    N_out = len(ind_out)
    Coi_op = sparse.linalg.LinearOperator(shape = (N_out,N_in),matvec=C_func,
    rmatvec = CH_func,matmat = Cmat_func,dtype=np.float64)

    return Coi_op

# ==============================================================================
def precondLinearOp(solver,N_out,N_in):

    P_func = lambda x: solver(x)
    PH_func = lambda x: solver(x)
    def Pmat_func(X):
        # Z = np.empty((N_out,X.shape[1]),dtype = np.float64)
        # for j in range(X.shape[1]):
        #     Z[:,j] = solver(X[:,j])
        return np.array([solver(X[:,j]) for j in range(X.shape[1])]).T

    P_op = sparse.linalg.LinearOperator(shape = (N_out,N_in),matvec=P_func,
    rmatvec = PH_func,matmat = Pmat_func,dtype=np.float64)

    return P_op


def pcg_solve(ind_obs, M, S_2N, b, x0, tol, maxiter, Psolver, pcg_algo):
    """
    Function that solves the problem Ax = b by calling iterative algorithms,
    using user-specified methods.
    Where A can be written as A = W_o F* D F W_o^T

    Parameters
    ----------
    ind_obs : array_like
        array of size N_o or list containing the chronological indices of the values
        contained in the observed data vector in the complete data vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    S_2N : numpy array (size P >= 2N)
        PSD vector
    b : numpy array
        vector of size N_o containing the right-hand side of linear system to solve
    x0 : numpy array
        vector of size N_o: first guess for the linear system to be solved
    tol : scalar float
        stopping criterium for the preconditioned conjugate gradient algorithm
    Psolver : sparse.linalg.factorized instance
        preconditionner matrix: linear operator which calculates an approximation
        of the solution: u_approx = C_OO^{-1} b for any vector b
    pcg_algo : string {'mine','scipy','scipy.bicgstab','scipy.bicg','scipy.cg','scipy.cgs'}
        Type of preconditioned conjugate gradient (PCG) algorithm to use among


    Returns
    -------
    u : numpy array
        approximate solution of the linear system

    """

    N_o = len(ind_obs)
    #x0 = np.zeros(N_o)

    if pcg_algo == 'mine':
        Coo_func = lambda x: matVectProd(x,ind_obs,ind_obs,M,S_2N)
        u,sr = precondBiCGSTAB(x0,b,Coo_func,maxiter,tol,Psolver)
    elif 'scipy' in pcg_algo:
        Coo_op = covLinearOp(ind_obs,ind_obs,M,S_2N)
        P_op = precondLinearOp(Psolver,N_o,N_o)
        tol_eff = np.min([tol,tol*LA.norm(b)])
        if (pcg_algo == 'scipy') | (pcg_algo == 'scipy.bicgstab'):
            u,info = sparse.linalg.bicgstab(Coo_op, b, x0=x0,
            tol=tol_eff,maxiter=maxiter,M=P_op,callback=None)
            printPCGstatus(info)
        elif (pcg_algo == 'scipy.bicg'):
            u,info = sparse.linalg.bicg(Coo_op, b, x0=x0,
            tol=tol_eff,maxiter=maxiter,M=P_op,callback=None)
            printPCGstatus(info)
        elif (pcg_algo == 'scipy.cg'):
            u,info = sparse.linalg.cg(Coo_op, b, x0=x0,
            tol=tol_eff,maxiter=maxiter,M=P_op,callback=None)
            printPCGstatus(info)
        elif (pcg_algo == 'scipy.cgs'):
            u,info = sparse.linalg.cgs(Coo_op, b, x0=x0,
            tol=tol_eff,maxiter=maxiter,M=P_op,callback=None)
            printPCGstatus(info)
        else:
            raise ValueError("Unknown PCG algorithm name")
        print("Value of || A * x - b ||/||b|| at exit:")
        print(str(LA.norm(Coo_op.dot(u)-b)/LA.norm(b)))

    else:
        raise ValueError("Unknown PCG algorithm name")

    return u
