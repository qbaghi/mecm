import numpy as np
import copy
import mecm
from .matrixalgebra import *



class bayesImputation:

    def __init__(self,M,R,S_2N,p=5,Nit = 150,tol = 1e-6):

        self.M = M
        self.N = len(M)
        self.p = p
        self.Nit = Nit
        self.tol = tol
        self.ind_mis = np.where( M==0 )[0]#.astype(int)
        self.ind_obs = np.where( M!=0 )[0]#.astype(int)
        self.taper = 'Wendland2'
        self.PCGalgo = 'scipy'

        # For initialization of linear system solution
        self.x0 = np.zeros(len(self.ind_obs))

        # Precomputation of the preconjugate solver
        self.S_2N = S_2N
        self.R = np.real(ifft(self.S_2N))[0:self.N]
        self.solve = self.computeSolver(self.R,self.M)
        self.N_seg = 1



    def computeSolver(self,R,M):

        return mecm.computePrecond(R,M,p=self.p,taper = self.taper)

    def noise_imputation(self,z_obs,solve):
        """

        Estimate missing data residals from observed residuals.

        Parameters
        ----------
        z_obs : array_like
            observed residual vector (size N_obs)
        solve : linear operator
            solver of  C_oo x = eps
        approx : boolean, optional
            if True, the linear system C_oo x = eps is solved by only applying
            the linear operator (solver) to eps once. This is a crude sparse
            approximation. Default is False.

        Returns
        -------
        z_mis_rec : array_like
            reconstructed missing residual vector (size N_mis)

        """

        # Solve the linear system C_oo x = eps
        # if not approx:
        u = PCGsolve(self.ind_obs,self.M,self.S_2N,z_obs,self.x0,
        self.tol,self.Nit,solve,self.PCGalgo)
        # else:
        #     u = solve(z_obs)

        # Reconstructed residuals
        z_mis_rec = matVectProd(u,self.ind_obs,self.ind_mis,
        self.M,self.S_2N)

        return z_mis_rec

    def full_imputation(self,y,A,beta):
        """

        Parameters
        ----------
        y : array_like
            masked data vector (size N)
        A : array_like
            design matrix of the determistic signal model assuming signal = A * beta
        beta : array_like
            signal extrinsic parameters
        approx : boolean, optional
            if True, the linear system C_oo x = eps is solved by only applying
            the linear operator (solver) to eps once. This is a crude sparse
            approximation. Default is False.

        Returns
        -------
        y_rec : array_like
            reconstructed data vector

        """

        # Compute observed residuals
        z_obs = y[self.ind_obs] - np.dot(A[self.ind_obs,:],beta)

        # Compute reconstructed vector
        y_rec = copy.deepcopy(y)
        y_rec[self.ind_mis] = np.dot(A[self.ind_mis,:],beta) \
        + self.noise_imputation(z_obs,self.solve)


        return y_rec
