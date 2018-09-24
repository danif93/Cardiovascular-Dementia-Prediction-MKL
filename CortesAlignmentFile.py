import numpy as np
import Utils as ut
#from myLasso import Lasso
#from myLasso2 import Lasso
from myLasso3 import Lasso
#from myLasso4 import Lasso

class centeredKernelAlignment:

    def centeredKernel(K): # K^c
        Nt = K.shape[0]
        Ns = K.shape[1]
        oneNt = np.ones((Nt)).reshape(-1,1)
        oneNs = np.ones((Ns)).reshape(-1,1)
        oneNt_mat = np.outer(oneNt, oneNt)
        oneNs_mat = np.outer(oneNs, oneNs)
        composite = np.outer(oneNt, oneNs)
        
        add1 = 1/Nt * np.dot(oneNt_mat, K)
        add2 = 1/Nt * np.dot(K, oneNs_mat)
        add3 = 1/(Nt*Nt) * np.dot(np.dot(oneNt.T, K), oneNs) * composite
        
        return K - add1 - add2 + add3


    def _kernelSimilarityMatrix(K_list): # M

        M = np.zeros((len(K_list), len(K_list)))

        for i, K1 in enumerate(K_list):
            for j, K2 in enumerate(K_list[i:]):

                s = ut.frobeniusInnerProduct(K1, K2)
                M[i, i+j] = s

                if j != 0:
                    M[i+j, i] = s
        return M


    def _idealSimilarityVector(K_list, IK): # a

        a = np.zeros((len(K_list)))

        for i, K in enumerate(K_list):
            a[i] = ut.frobeniusInnerProduct(K, IK)
        
        return a

    
    def coef(obj, M, a):
        eta = np.dot(np.linalg.pinv(M), a)
        return eta / np.linalg.norm(eta)

    def computeEta(K_list, IK, y = None, sparsity = 0, lamb = 0, maxIter = 0,  verbose = False):
        
        K_c_list = [centeredKernelAlignment.centeredKernel(K) for K in K_list]
        M = centeredKernelAlignment._kernelSimilarityMatrix(K_c_list)
        a = centeredKernelAlignment._idealSimilarityVector(K_c_list, IK)
        
        if sparsity != 0:
            #sp = Lasso(alpha = sparsity, verbose = verbose, max_iter = maxIter, estimator = centeredKernelAlignment())
            #eta = sp.fit(M, a, y, K_list).coef_
            eta = Lasso(M, a, sparsity, verbose = verbose)
            
        else:

            if lamb != 0:
                M -= lamb*np.identity(M.shape[0])
                    
            eta = np.dot(np.linalg.pinv(M), a)
            eta /= np.linalg.norm(eta)
            
        return eta 


    def score(k1, k2, ideal = True):
        k1c = centeredKernelAlignment.centeredKernel(k1)
        
        if ideal:
            k2c = k2
        else:
            k2c = centeredKernelAlignment.centeredKernel(k2)

        num = ut.frobeniusInnerProduct(k1c, k2c)
        den = np.sqrt(ut.frobeniusInnerProduct(k1c, k1c)*ut.frobeniusInnerProduct(k2c, k2c))
        return num/den
    
    def externalScore(obj, k1, k2):
        return centeredKernelAlignment.score(k1, k2)