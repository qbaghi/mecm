from unittest import TestCase

import sys
import os
import mecm
import numpy as np

class TestMecm(TestCase):

    def testmaxlike(self):

        Npts = 2**16
        s = np.loadtxt('signal.txt')[0:Npts]
        M = np.loadtxt('mask.txt')[0:Npts]
        A = np.loadtxt('Amatrix.txt')[0:Npts,:]
        n = np.loadtxt('noise.txt')[0:Npts]
        scale = 1e-9

        # Masked data : signal + noise
        y = M*(s+n)/scale



        print("------ Data loaded ------")
        beta,cov,betavector,y_rec,I_condMean,PSD = mecm.maxlike(y,M,A,N_it_max=10,
        eps=1e-4,p=20,Nd=10,N_est=1000,Nit_cg=150,tol_cg=1e-5,compute_cov = True,
        verbose = True,PCGalgo = 'scipy')
        np.savetxt('beta.txt',beta*scale)
        np.savetxt('y_rec.txt',y_rec*scale)
        print("------ End of test ------")



        if type(beta) == np.ndarray :
            print('Estimated parameters: '+str(beta))
            #self.assertTrue(isinstance(beta, np.ndarray))
            if isinstance(beta,np.ndarray):
                print("Test success.")
