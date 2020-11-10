# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# ==============================================================================
# This routine perform the Modified expectation maximization algorithm (M-ECM)
# based on missing data imputation by conditional distribution using FFT.
# ==============================================================================
import numpy as np
from scipy import sparse
from numpy import linalg as LA
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
import random
import copy
from .matrixalgebra import cov_linear_op, precondLinearOp, matprecondBiCGSTAB
from .matrixalgebra import mat_vect_prod, pcg_solve
from .noise import generateNoiseFromDSP
# from .leastsquares import pmesureWeighted, pmesure_optimized_TF
from . import leastsquares
from .localestimator import nextpow2, PSDEstimate
from .psd import PSDSpline
pyfftw.interfaces.cache.enable()


class MECM:

    def __init__(self, y, mask, a_mat, psd_cls, eps=1e-3, p=20,
                 nd=10, nit_cg=1000, tol_cg=1e-4, pcg_algo='scipy',
                 mask0=None):

        self.y = y
        self.mask = mask
        if mask0 is None:
            self.mask0 = mask[:]
        else:
            self.mask0 = mask0
        self.a_mat = a_mat
        # Number of data points
        self.n_data = y.shape[0]
        # Number of Fourier frequencies used for the spectrum calculation
        if self.n_data % 2 == 0:
            self.n_fft = np.int(2 * self.n_data)
        else:
            self.n_fft = np.int(2**nextpow2(2 * self.n_data))
        # Physical model in the frequency domain
        # ----------------------------------------------------------------------
        # Apodization
        self.wd = np.hanning(self.n_data)

        self.psd_cls = psd_cls
        self.eps = eps
        self.p = p
        self.nd = nd
        self.nit_cg = nit_cg
        self.tol_cg = tol_cg
        self.pcg_algo = pcg_algo

        # Indices of missing data
        self.ind_mis = np.where(self.mask == 0)[0]
        # Indices of observed data
        self.ind_obs = np.where(self.mask == 1)[0]

        # Store the information about the output of the PCG computations
        self.infos = []
        # PSD Initialization
        self.p_cond_mean = periodogram(self.mask0 * (self.y - self.y.mean()),
                                       self.n_data,
                                       wind=self.wd * self.mask0)

        self.psd_cls.estimate_from_periodogram(self.p_cond_mean)
        # self.psd_cls.estimate(self.res, wind=self.wd * self.mask0)
        self.s_n = psd_cls.calculate(self.n_data)
        self.s_2n = psd_cls.calculate(self.n_fft)
        # Imputed data initialization
        self.y_rec = y[:]

        # Initialization of regression parameters
        # First guess
        # Linear regression
        # ----------------------------------------------------------------------
        print("Perform linear regression...")
        a_mat_mask = a_mat * np.array([self.wd * self.mask0]).T
        # In the frequency domain
        self.a_mat_dft = fft(a_mat * np.array([self.wd]).T, axis=0)
        self.a_mat_mask_dft = fft(a_mat_mask, axis=0)
        self.y_dft = fft(self.y * self.wd * self.mask0)
        # Estimate vector of amplitudes
        self.beta = np.real(leastsquares.generalized_least_squares(
            self.a_mat_mask_dft, self.y_dft, self.s_n))
        # Estimate covariance
        print("Complete.")
        self.beta0_norm = LA.norm(self.beta)
        # History of estimates N_iterations x K
        self.beta_vect = copy.deepcopy(self.beta)
        # Initialization of variables
        self.beta_old = np.zeros(self.beta.shape[0])

        # Initialization of estimated signal
        self.signal = np.dot(self.a_mat, self.beta)
        # Initialize regression residuals
        self.res = self.mask0 * (self.y - self.signal)

        # Difference from one update to the other
        self.diff = 1.
        # Iteration counter (just for printing purposes)
        self.counter = 0
        # Preconditionning linear operator
        self.precond_solver = None

    def c_mo(self, v):
        """Covariance missing / observed data : matrix operator

        Parameters
        ----------
        v : ndarray
            vector on which to perform the product C_mo v

        Returns
        -------
        ndarray
            vector = C_mo v
        """

        return mat_vect_prod(v, self.ind_obs, self.ind_mis, self.mask,
                             self.s_2n)

    def draw_residuals(self, mu_z_m_given_o):
        """Draw a realization of the imputed vector of residuals

        Parameters
        ----------
        mu_z_m_given_o : ndarray
            Conditional expectation of the missing data residuals given the
            observed data.

        Returns
        -------
        eps : ndarray
            drawn full conditional vector of residuals

        """

        # Draws a full time series following prescribed estimated PSD
        e = np.real(generateNoiseFromDSP(
            np.sqrt(self.s_2n*2.), 1.)[0:self.n_data])
        # Solve the linear system C_oo x = Z_o. x = z*_o
        uo, info = pcg_solve(self.ind_obs, self.mask, self.s_2n,
                             e[self.ind_obs], np.zeros(len(self.ind_obs)),
                             self.tol_cg, self.nit_cg,
                             self.precond_solver, self.pcg_algo)
        self.infos.append(info)
        # Z u | o = Z_tilde_m + Cmo Coo^-1 ( Z_o - Z_star_o )
        # u = Cmo Coo^-1 Z_star_o
        u_m_o = mat_vect_prod(uo, self.ind_obs, self.ind_mis, self.mask,
                              self.s_2n)

        eps = np.zeros(self.n_data)
        # Observed data points
        eps[self.ind_obs] = self.res[self.ind_obs]
        # Missing data points and extension to 2N points
        # Z_star_m|o = mu_z_m|o + z*_m - Cmo Coo^-1 z*_o
        eps[self.ind_mis] = mu_z_m_given_o + e[self.ind_mis] - u_m_o
        # eps = e[self.ind_mis] + self.c_mo(uo)

        return eps

    def e_step(self):

        # E step : missing data reconstruction
        # ------------------------------------
        # Compute autocovariance
        autocorr = self.psd_cls.calculate_autocorr(self.n_data)
        # Compute preconditionner if not already done
        if self.precond_solver is None:
            print("Build preconditionner...")
            self.precond_solver = compute_precond(autocorr, self.mask,
                                                  p=self.p,
                                                  taper='Wendland2')
            print("Preconditionner built.")
        # First guess
        x0 = np.zeros(len(self.ind_obs))
        # Solve the linear system C_oo x = eps where eps are the residuals
        u, info = pcg_solve(self.ind_obs, self.mask, self.s_2n,
                            self.res[self.ind_obs], x0,
                            self.tol_cg, self.nit_cg,
                            self.precond_solver, self.pcg_algo)
        # Estimate the missing data residuals via z | o = Cmo Coo^-1 z_o
        y_mis_res = mat_vect_prod(u, self.ind_obs, self.ind_mis,
                                  self.mask, self.s_2n)
        # Construct the full imputed data vector
        self.y_rec = self.y[:]
        self.y_rec[self.ind_mis] = y_mis_res + self.signal[self.ind_mis]

        # To compute the expectation of second-order statistics, we should
        # do some random draws
        if self.nd > 1:
            cond_draws = np.array([self.draw_residuals(y_mis_res)
                                   for i in range(self.nd)])
            # Periodogram of the draws
            p_mat = periodogram(cond_draws, self.n_data)
            # Conditional average
            self.p_cond_mean = np.mean(p_mat, axis=0)

        elif self.nd == 1:
            cond_draw = self.draw_residuals(y_mis_res)
            self.p_cond_mean = periodogram(cond_draw, self.n_data)

        # self.y_rec, self.precond_solver = conj_grad_imputation(
        #     self.y, self.a_mat, self.beta, self.mask, self.s_2n,
        #     self.p, self.nit_cg, self.pcg_algo,
        #     precond_solver=self.precond_solver,
        #     tol=self.tol_cg, store_info=self.infos)
        #
        # # Conditional draws
        # cond_draws = cond_monte_carlo(
        #     self.mask * (self.y - np.dot(self.a_mat, self.beta)),
        #     self.precond_solver, self.mask, self.nd, self.s_2n, self.nit_cg,
        #     self.pcg_algo, tol=self.tol_cg, store_info=self.infos)

    def cm2_step(self):

        # CM2 step : estimation of deterministic model parameters
        # -------------------------------------------------------
        # Keep in mind the old value of parameters
        self.beta_old = self.beta[:]
        # Update data FFT
        self.y_dft = fft(self.y_rec * self.wd)
        # Update parameter vector
        # Estimate vector of amplitudes
        self.beta = np.real(leastsquares.generalized_least_squares(
            self.a_mat_dft, self.y_dft, self.s_n))
        self.beta_vect = np.vstack((self.beta_vect, self.beta))
        # Update the estimated signal
        self.signal = np.dot(self.a_mat, self.beta)
        # Update estimation residuals
        self.res = self.y_rec - self.signal

    def cm1_step(self):

        # CM1 step : estimation of noise model parameters (PSD)
        # -------------------------------------------------------
        # PSD.MLestimateFromI(p_cond_mean, Niter=1)
        self.psd_cls.estimate_from_periodogram(self.p_cond_mean)
        self.s_n = self.psd_cls.calculate(self.n_data)
        self.s_2n = self.psd_cls.calculate(self.n_fft)

    def ecm_step(self, verbose=False):

        if (self.diff > self.eps):

            # Expectation step
            self.e_step()
            # Maximization steps
            self.cm1_step()
            self.cm2_step()

            # Calculate the relative difference between new update and old
            # parameter estimate
            self.diff = LA.norm(self.beta - self.beta_old)/self.beta0_norm

            # Update counter
            self.counter += 1

            if verbose:
                print('--------------------------')
                print('Iteration ' + str(self.counter)
                      + ' completed, de/e =' + str(self.diff))
                print('--------------------------')

        else:
            pass

    def compute_covariance(self, nthreads=1):

        return mecmcovariance(self.a_mat, self.mask, self.s_2n,
                              self.precond_solver, self.pcg_algo,
                              n_it=self.nit_cg, tol=self.tol_cg,
                              nthreads=nthreads)


def maxlike(y, mask, a_mat, n_it_max=15, eps=1e-3, p=20, nd=10, n_est=100,
            nit_cg=1000, tol_cg=1e-4, compute_cov=True, verbose=True,
            pcg_algo='scipy', psd_cls=None, mask0=None):
    """

    Function estimating the regression parameters for a problem of
    multivariate Gaussian maximum likelihood with missing data,
    using the M-ECM algorithm.

    Parameters
    ----------
    y : numpy array (size n_data)
        masked data vector
    mask : numpy array (size n_data)
        mask vector (with entries equal to 0 or 1)
    a_mat : numpy array (size n_data x K)
        design matrix (contains partial derivatives of signal wrt parameters)
    n_it_max : scalar integer
        number of iterations of the M-ECM algorithm
    eps : scalar float
        tolerance criterium to stop the ECM algorithm (default is 1e-3)
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the
        conjugate gradients.
    nd : scalar integer
        number of Monte-Carlo draws to estimate the conditional expectation of
        the noise periodogram with respect to the observed data
    n_est : scalar integer
        number of frequency points where to estimate the noise power spectral
        density (on a logarithmic grid)
    N_it_cg : scalar integer
        maximum number of iterations for the conjugate gradient algorithm.
    tol_cg : scalar float
        tolerance criterium to stop the PCG algorithm (default is 1e-4). Stops
        when the relative residual error ||a_mat x - b||/||b|| is below tol_cg.
    verbose : boolean
        if True, a message is printed at the end of each iteration, displaying
        the value of the convergence criterion
    compute_cov : boolean
        if True, the covariance of the estimator is computed
    pcg_algo : string {'mine','scipy','scipy.bicgstab','scipy.bicg','scipy.cg',
        'scipy.cgs'}
        Type of preconditioned conjugate gradient (PCG) algorithm to use.
    psd_cls : psd.PSDSpline instance or None
        PSD model class
    mask0 : numpy array (size n_data)
        initial mask vector (with entries equal to 0 or 1), to use for the
        initialization only. Should be less conservative (less dense) than
        the regular mask.

    Returns
    -------
    beta_new : array_like (size K)
        final value of the estimated parameter vector
    cov : None or array_like (size K x K)
        estimated covariance matrix of the parameter vector. If compute_cov is
        False, then cov is None
    beta : array_like (size N_iterations x K)
        vector storing the updates of the parameter vector at each iteration
    y_rec : array_like (size n_data)
        reconstructed data vector, i.e. conditional expectation of the data
        given the available observations.
    p_cond_mean : array_like (size n_data)
        conditional expectation of the noise periodogram
    PSD : PSD_estimate class instance
        class containing all the information regarding the estimated noise PSD
    diff : float
        M-ECM convergence criterium de/e on the parameter of interest


    References
    ----------

    .. [1] Q. Baghi et al, "Gaussian regression and power spectral density
    estimation with missing data: The MICROSCOPE space mission as a case study,
    " Physical Review D, vol. 93, num. 12, 2016


    Notes
    -----

    """

    # Instantiate PSD class
    if psd_cls is None:
        n_data = y.shape[0]
        # Number of Fourier frequencies used for the spectrum calculation
        if n_data % 2 == 0:
            n_fft = np.int(2 * n_data)
        else:
            n_fft = np.int(2**nextpow2(2 * n_data))
        psd_cls = PSDEstimate(n_est, n_data, n_fft, h_min=None, h_max=None)
        # psd_cls = PSDSpline(n_data, 1.0, n_knots=30, d=3,
        #                     fmin=None, fmax=None, f_knots=None, ext=3)
    # Intantiate ECM class
    ecm = MECM(y, mask, a_mat, psd_cls, eps=eps, p=p,
               nd=nd, nit_cg=nit_cg, tol_cg=tol_cg, pcg_algo=pcg_algo,
               mask0=mask0)

    # Run M-ECM iterations alternating betweeb E-steps and M-steps
    [ecm.ecm_step(verbose=verbose) for i in range(n_it_max)]

    # Output flag
    success = np.all(np.array(ecm.infos) == 0)

    print("------ MECM iterations completed. ------")
    if ecm.diff > eps:
        raise Warning("Attention: MECM algorithm ended \
        without reaching the specified convergence criterium.")
        print("Current criterium: " + str(ecm.diff) + " > " + str(eps))
        success = False

    print("Computation of the covariance...")
    if compute_cov:
        cov, u = ecm.compute_covariance(nthreads=1)
    else:
        cov = None

    return (ecm.beta, cov, ecm.beta_vect, ecm.y_rec[0:ecm.n_data],
            ecm.p_cond_mean, ecm.psd_cls, success, ecm.diff)


# ==============================================================================
def mecmcovariance(a_mat, mask, s_2n, solve, pcg_algo, n_it=150, tol=1e-7,
                   r=1e-15, nthreads=1):
    """
    Function estimating the covariance of regression parameters for a problem
    of multivariate Gaussian maximum likelihood with missing data.

    Parameters
    ----------
    a_mat : numpy array (size n_data x K)
        design matrix (contains partial derivatives of signal wrt parameters)
    mask : numpy array (size n_data)
        mask vector (with entries equal to 0 or 1)
    s_2n : numpy array (size n_fft >= 2N)
        PSD vector
    solve : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    n_it : scalar integer, optional
        maximum number of iterations for the conjugate gradient algorithm.
    tol : scalar float, optional
        tolerance criterium to stop the PCG algorithm (default is 1e-7)
    r : scalar float, optional (default is 1e-15)
        Cutoff for small singular values when computing the inverse matrix.
        Singular values smaller (in modulus) than
        rcond * largest_singular_value (in modulus) are set to zero.

    Returns
    -------
    cov : array_like (size K x K)
        estimated covariance matrix of the parameter vector

    """

    print("The chosen BICGSTAB algorithm is scipy code.")

    A_obs = a_mat[mask == 1, :]
    (n_o, K) = np.shape(A_obs)
    U = np.zeros((n_o, K), dtype=np.float64)
    # Missing data indices
    ind_obs = np.where(mask != 0)[0]

    # Construct the operator calculating C_OO*x
    x0 = np.zeros(n_o)

    if pcg_algo == 'mine':
        def coo_func(x):
            return mat_vect_prod(x, ind_obs, ind_obs, mask, s_2n)
        U = matprecondBiCGSTAB(x0, A_obs, coo_func, n_it, tol, solve,
                               pcg_algo=pcg_algo, nthreads=nthreads)
    elif 'scipy' in pcg_algo:
        Coo_op = cov_linear_op(ind_obs, ind_obs, mask, s_2n)
        P_op = precondLinearOp(solve, len(ind_obs), len(ind_obs))
        U = matprecondBiCGSTAB(x0, A_obs, Coo_op, n_it, tol, P_op,
                               pcg_algo=pcg_algo, nthreads=nthreads)
        # innerPrecondBiCGSTAB(U,x0,A_obs,Coo_op,n_it,tol,P_op,pcg_algo)

    try:
        cov = LA.pinv(np.dot(np.conj(A_obs.T), U), rcond=r)
    except UserWarning("SVD did not converge"):
        cov = []

    return cov, U


# ==============================================================================
def periodogram(x_mat, nfft, wind='hanning'):
    """
    Function computing the periodogram of the input data with a specified
    number of Fourier frequencies nfft (possible zero-padding) and apodization
    window.

    Parameters
    ----------
    x_mat : numpy array (size nd x n_data)
        data for which the periodogram must be computed. Can be a
        one-dimentional vector or a matrix containing a time series in each
        row.  The function computes as many periodograms as there are rows in
        x_mat, and returns a matrix of same size as x_mat.
    nfft : scalar integer
        number of points to consider in the Fourier grid
    wind : string or array_like
        name of the apodization window to apply, or window values

    Returns
    -------
    Per : numpy array (size nd x n_data)
        matrix (or vector) of periodogram

    """

    ll = len(np.shape(x_mat))

    if ll == 1:
        n_data = len(x_mat)

        if type(wind) == str:
            # Windowing
            if wind == 'hanning':
                w = np.hanning(n_data)
            elif wind == 'ones':
                w = np.ones(n_data)
            elif isinstance(wind, (list, tuple, np.ndarray)):
                w = wind
            else:
                raise TypeError("Window argument is wrong")
        elif (type(wind) == list) | (type(wind) == np.ndarray):
            w = wind[:]

        K2 = np.sum(w**2)
        Per = np.abs(fft(x_mat * w, nfft))**2 / K2

    elif ll == 2:
        (nd, n_data) = np.shape(x_mat)
        # Windowing
        if type(wind) == str:
            if wind == 'hanning':
                w = np.hanning(n_data)
            elif wind == 'ones':
                w = np.ones(n_data)
            elif isinstance(wind, (list, tuple, np.ndarray)):
                w = wind
            else:
                raise TypeError("Window argument is wrong")

        elif (type(wind) == list) | (type(wind) == np.ndarray):
            w = wind[:]

        K2 = np.sum(w**2)
        Per = np.abs(fft(np.multiply(x_mat, w), n=nfft, axis=1))**2 / K2

    return Per


# ==============================================================================
def cond_draw(n_fft, n_it, s_2n, solver, z_o, mu_m_given_o, ind_obs,
              ind_mis, mask, pcg_algo, tol=1e-7, store_info=None):
    """
    Function performing random draws of the complete data noise vector
    conditionnaly on the observed data.

    Parameters
    ----------
    n_fft : scalar integer
        Size of the initial random vector which is drawn. The initial vector
        follows a stationary distribution whose covariance matrix is circulant.
        It is then truncated to N_final to obtain the desired distribution whose
        covariance matrix is Toeplitz.
    n_it : scalar integer
        Maximum number of conjugate gradient iterations.
    s_2n : numpy array (size n_fft >= 2N)
        PSD vector
    coo_func : function
        Linear operator that calculates the matrix-to-vector product Coo x for
        any vector x of size No (number of observed data points)
    solver : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    z_o : array_like (size No)
        vector of observed residuals
    mu_m_given_o : array_like (size Nm)
        vector of conditional expectation of the missing data given the
        observed data
    ind_obs : array_like (size No)
        vector of chronological indices of the observed data points in the
        complete data vector
    ind_mis : array_like (size No)
        vector of chronological indices of the missing data points in the
        complete data vector
    mask : numpy array (size n_data)
        mask vector (with entries equal to 0 or 1)
    tol : scalar float
        Stoping criteria for the conjugate gradient algorithm, optional
    store_info : list or None
        If list, append the output flag of the PCG algorithm to the list.

    Returns
    -------
    eps : numpy array (size n_data)
        realization of the vector of conditional noise given the observed data

    """

    # New implementation: the size of the vector that is randomly drawn is
    # equal to the size of the mask. If one wants a larger vector, one must
    # have to zero-pad the mask: mask = np.concatenate((mask,np.zeros(Npad)))
    n_data = len(mask)
    DSP = np.sqrt(s_2n*2.)
    v = generateNoiseFromDSP(DSP, 1.)
    # v = np.array([random.gauss(0,1.) for _ in range(n_fft)])
    # e = np.sqrt(n_fft/np.float(N_final))*np.sqrt(n_fft)
    # * np.real( ifft( np.sqrt(s_2n) * v )[0:N_final] )
    e = np.real(v[0:n_data] * np.sqrt(np.var(v)/np.var(v[0:n_data])))
    # u0 = solve(e[self.ind_obs])

    n_o = len(ind_obs)
    u0 = np.zeros(n_o)

    u, info = pcg_solve(ind_obs, mask, s_2n, e[ind_obs], u0, tol, n_it, solver,
                        pcg_algo)

    if store_info:
        store_info.append(info)

    # u, sr = conjugateGradientSolvePrecond(u0,e[ind_obs],coo_func,n_it,stop,
    #                                       solver)

    u_m_o = mat_vect_prod(u, ind_obs, ind_mis, mask, s_2n)

    eps = np.zeros(n_data)

    # Observed data points
    eps[ind_obs] = z_o
    # Missing data points and extension to 2N points
    eps[ind_mis] = mu_m_given_o + e[ind_mis] - u_m_o

    return eps


# ==============================================================================
def cond_monte_carlo(eps, solve, mask, nd, s_2n, n_it, pcg_algo, seed=None,
                     tol=1e-7, store_info=None):
    """
    Function performing random draws of the complete data noise vector
    conditionnaly on the observed data.

    Parameters
    ----------
    eps : numpy array (size n_data)
        input vector of residuals (difference between data and model)
    solve : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    mask : numpy array (size n_data)
        mask vector (with entries equal to 0 or 1)
    s_2n : numpy array (size n_fft >= 2N)
        PSD vector
    n_it : scalar integer
        Maximum number of conjugate gradient iterations.
    seed : character string, optional
        Regenerate the seed with system time (if None) or chosen integer.
    tol : scalar float
        Stoping criteria for the conjugate gradient algorithm, optional
    store_info : list or None
        If list, append the output flag of the PCG algorithm to the list.


    Returns
    -------
    cond_draws : numpy array (size nd x n_data)
        Matrix whose rows are realization of the conditional noise vector

    """

    # Missing data indices
    ind_mis = np.where(mask == 0)[0]
    ind_obs = np.where(mask != 0)[0]
    n_data = len(mask)
    n_fft = len(s_2n)

    # Calculate the residuals at the observed data points
    eps_o = eps[ind_obs]

    # Calculate the conditional mean
    n_o = len(ind_obs)
    x0 = np.zeros(n_o)
    u, info = pcg_solve(ind_obs, mask, s_2n, eps_o, x0, tol, n_it, solve,
                        pcg_algo)
    if store_info:
        store_info.append(info)

    # u,sr = conjugateGradientSolvePrecond(np.zeros(len(ind_obs)),eps_o,
    # coo_func,n_it,np.std(eps_o),solve)
    mu_m_given_o = mat_vect_prod(u, ind_obs, ind_mis, mask, s_2n)

    cond_draws = np.zeros((nd, n_data))
    # Regenerate the seed with system time or chosen integer
    random.seed(seed)

    # ==========================================================================
    # Without parallelization:
    # ==========================================================================
    cond_draws = np.array([cond_draw(n_fft, n_it, s_2n, solve, eps_o,
                                     mu_m_given_o, ind_obs, ind_mis, mask,
                                     pcg_algo, tol=tol, store_info=store_info)
                           for i in range(nd)])

    return cond_draws


# ==============================================================================
def build_sparse_cov2(autocorr, p, n_data, form=None, taper='Wendland2'):
    """
    This function constructs a sparse matrix which is a tappered, approximate
    version of the covariance matrix of a stationary process of autocovariance
    function autocorr and size n_data x n_data.

    Parameters
    ----------
    autocorr : numpy array
        input autocovariance functions at each lag (size n_data)
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function.
    n_data : scalar integer
        Size of the complete data vector
    form : character string (or None), optional
        storage format of the sparse matrix (default is None)
    taper : string
        type of taper function to smoothly decrease the tapered covariance
        function down to zero.


    Returns
    -------
    C_tap : scipy.sparse matrix
        Tappered covariance matrix (size n_data x n_data) with p non-zero diagonals.


    """
    k = np.array([])
    values = list()
    tap = taper_covariance(np.arange(0, n_data), p, taper=taper)
    r_tap = autocorr[0:n_data] * tap

    # calculate the rows and columns indices of the non-zero values
    # Do it diagonal per diagonal
    for i in range(p+1):
        rf = np.ones(n_data - i) * r_tap[i]
        values.append(rf)
        k = np.hstack((k, i))
        # Symmetric values
        if i != 0:
            values.append(rf)
            k = np.hstack((k, -i))
    # [build_diagonal(values, r_tap, k, i) for i in range(p + 1)]

    return sparse.diags(values, k.astype(int), format=form,
                        dtype=autocorr.dtype)


def build_diagonal(values, r_tap, k_vect, i):
    """
    Populate the list of diagonals to feed sparse.diags

    Parameters
    ----------
    values : list
        list of ndarrays which are the diagonals of the covariance matrix
    r_tap : ndarray
        autocovariance
    k_vect : array_like
        array of diagonal indices
    i : int
        index of the diagonal to populate

    Returns
    -------
    nothing. append values to values and k_vect

    """

    rf = np.ones(r_tap.shape[0] - i) * r_tap[i]
    values.append(rf)
    # k = np.hstack((k_vect, i))
    np.append(k_vect, i)
    # Symmetric values
    if i != 0:
        values.append(rf)
        # k = np.hstack((k_vect, -i))
        np.append(k_vect, -i)


# ==============================================================================
def conj_grad_imputation(s, a_mat, beta, mask, s_2n, p, n_it,
                         pcg_algo, precond_solver=None, tol=1e-7,
                         store_info=None):
    """
    Function performing universal kriging with the model y = a_mat beta + n
    using the conjugate gradient algorithm to solve for C_OO x = b.
    But this imputation function can be generalized to any solver algorithm.
    See "imputation" function for a more general formulation

    Parameters
    ----------
    s : array_like (size n_data)
        Input signal array
    a_mat : numpy array (size n_data x K)
        design matrix (contains partial derivatives of signal wrt parameters)
    beta : array_like (size K)
        regression parameter vector
    mask : numpy array (size n_data)
        mask vector (with entries equal to 0 or 1)
    s_2n : numpy array (size n_fft)
        spectrum array: value of the unilateral PSD times fs
    autocorr : numpy array (size n_fft)
        tappered autocovariance array: value of the autocovariance at all lags
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the
        conjugate gradients.
    n_it : scalar integer
        maximum number of iterations for the conjugate gradient algorithm
    solve : sparse.linalg.factorized instance or None
        linear operator which calculates x = C_OO^{-1} b for any vector b.
        If None, it is computed using parameter p.
    tol : scalar float
        tolerance criterium to stop the PCG algorithm

    Returns
    -------
    z : float or ndarray
        conditional expectation of the data vector
    solve : sparse.linalg.factorized instance
        linear operator which calculates x = C_OO^{-1} b for any vector b
    """

    n_data = len(s)

    # Preconditionning : use sparse matrix
    # ======================================================================
    ind_mis = np.where(mask == 0)[0]  # .astype(int)
    ind_obs = np.where(mask != 0)[0]  # .astype(int)

    # Deterministic model for observed and missing data
    # ======================================================================
    A_obs = a_mat[mask == 1, :]
    A_mis = a_mat[mask == 0, :]
    beta_est = beta.T

    # Preconditionner
    # ======================================================================
    if precond_solver is None:
        autocorr = np.real(ifft(s_2n))
        precond_solver = compute_precond(autocorr, mask, p=p,
                                         taper='Wendland2')

    # Residuals of observed data - model
    eps = s[ind_obs] - np.dot(A_obs, beta_est)

    # Conjugate gradients
    # ======================================================================
    # Solve the linear system C_oo x = eps
    # u,sr = conjugateGradientSolvePrecond(np.zeros(len(ind_obs)),eps,coo_func,
    # n_it,np.std(eps),solve)
    n_o = len(ind_obs)
    x0 = np.zeros(n_o)
    u, info = pcg_solve(ind_obs, mask, s_2n, eps, x0, tol, n_it,
                        precond_solver,
                        pcg_algo)

    if store_info:
        store_info.append(info)

    # Calculate the conditional expectation of missing data
    x = np.dot(A_mis, beta) + mat_vect_prod(u, ind_obs, ind_mis, mask, s_2n)

    z = np.zeros(n_data)
    z[ind_obs] = s[ind_obs]
    z[ind_mis] = x

    return z, precond_solver


# ==============================================================================
def taper_covariance(h, theta, taper='Wendland1', tau=10):
    """
    Function calculating a taper covariance that smoothly goes to zero when h
    goes to theta. This taper is to be mutiplied by the estimated covariance
    autocorr of the process, to discard correlations larger than theta.

    Ref : Reinhard FURRER, Marc G. GENTON, and Douglas NYCHKA,
    Covariance Tapering for Interpolation of Large Spatial Datasets,
    Journal of Computational and Graphical Statistics, Volume 15, Number 3,
    Pages 502–523,2006

    Parameters
    ----------
    h : numpy array of size n_data
        lag
    theta : scalar float
        taper parameter
    taper : string {'Wendland1','Wendland2','Spherical'}
        name of the taper function


    Returns
    -------
    C_tap : numpy array
        the taper covariance function values (vector of size n_data)
    """

    ii = np.where(h <= theta)[0]

    if taper == 'Wendland1':

        c = np.zeros(len(h))

        c[ii] = (1.-h[ii]/np.float(theta))**4 * (1 + 4.*h[ii]/np.float(theta))

    elif taper == 'Wendland2':

        c = np.zeros(len(h))

        c[ii] = (1-h[ii]/np.float(theta))**6 * (1 + 6.*h[ii]/theta + \
        35*h[ii]**2/(3.*theta**2))

    elif taper == 'Spherical':

        c = np.zeros(len(h))

        c[ii] = (1-h[ii]/np.float(theta))**2 * (1 + h[ii]/(2*np.float(theta)) )

    elif taper == 'Hanning':
        c = np.zeros(len(h))
        c[ii] = (np.hanning(2*theta)[theta:2*theta])**2

    elif taper == 'Gaussian':

        c = np.zeros(len(h))
        sigma = theta/5.
        c[ii] = np.sqrt( 1./(2*np.pi*sigma**2) ) * np.exp( - h[ii]**2/(2*sigma**2) )

    elif taper == 'rectSmooth':

        c = np.zeros(len(h))
        c[h <= theta-tau] = 1.
        jj = np.where( (h >= theta-tau) & (h <= theta) )[0]
        c[jj] = 0.5*(1 + np.cos( np.pi*(h[jj]-theta+tau)/np.float(tau) ) )

    elif taper == 'modifiedWendland2':

        c = np.zeros(len(h))
        c[h<=theta-tau] = 1.
        jj = np.where( (h>=theta-tau) & (h<=theta) )[0]

        c[jj] = (1-(h[jj]-theta+tau)/np.float(tau))**6 * (1 + \
        6.*(h[jj]-theta+tau)/tau + 35*(h[jj]-theta+tau)**2/(3.*tau**2))

    elif taper == 'rect':

        c = np.zeros(len(h))
        c[h <= theta] = 1.

    return c


# ==============================================================================
def compute_precond(autocorr, mask, p=10, ptype='sparse', taper='Wendland2',
                    square=True):
    """
    For a given mask and a given PSD function, the function computes the linear
    operator x = C_OO^{-1} b for any vector b, where C_OO is the covariance
    matrix of the observed data (at points where mask==1).


    Parameters
    ----------
    autocorr : numpy array
        input autocovariance functions at each lag (size n_data)
    mask : numpy array
        mask vector
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the
        conjugate gradients.
    ptype : string {'sparse','circulant'}
        specifies the type of preconditioner matrix (sparse approximation of
        the covariance or circulant approximation of the covariance)
    taper : string {'Wendland1','Wendland2','Spherical'}
        Name of the taper function. This argument is only used if
        ptype='sparse'
    square : bool
        whether to build a square matrix. if False, then

    Returns
    -------
    solve : sparse.linalg.factorized instance
        preconditionner operator, calculating n_fft x for all vectors x

    """

    n_data = len(mask)

    # ======================================================================
    ind_obs = np.where(mask != 0)[0]  # .astype(int)
    # Calculate the covariance matrix of the complete data

    if ptype == 'sparse':
        # Preconditionning : use sparse matrix
        c_mat = build_sparse_cov2(autocorr, p, n_data, form="csc", taper=taper)
        # Calculate the covariance matrix of the observed data
        c_temp = c_mat[:, ind_obs]
        # Preconditionner
        solve = sparse.linalg.factorized(c_temp[ind_obs, :])
        # Remove big variables
        del c_temp
        del c_mat

    elif ptype == 'circulant':

        s_2n = np.real(fft(autocorr))
        n_fft = len(s_2n)

        def solve(v):
            return np.real(ifft(fft(v, n_fft)/s_2n, len(v)))

    return solve
