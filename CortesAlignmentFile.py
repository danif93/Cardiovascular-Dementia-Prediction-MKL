import numpy as np
import Utils as ut
from myLasso import Lasso
from sklearn.preprocessing import normalize

class centeredKernelAlignment:

    def _centeredKernel(K): # K^c
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


    def computeEta(K_list, IK, sparsity = 0):
        
        if sparsity != 0:
            sp = Lasso(alpha = sparsity)
            num = normalize(sp.fit(K_list, IK).coef_)
            
        else:

            K_c_list = [centeredKernelAlignment._centeredKernel(K) for K in K_list]

            M = centeredKernelAlignment._kernelSimilarityMatrix(K_c_list)

            a = centeredKernelAlignment._idealSimilarityVector(K_c_list, IK)

            num = np.dot(np.linalg.inv(M), a)

        return num / np.linalg.norm(num)


    def score(k1, k2, ideal = True):
        k1c = centeredKernelAlignment._centeredKernel(k1)
        if ideal:
            k2c = k2
        else:
            k2c = centeredKernelAlignment._centeredKernel(k2)

        num = ut.frobeniusInnerProduct(k1c, k2c)
        den = np.sqrt(ut.frobeniusInnerProduct(k1c, k1c)*ut.frobeniusInnerProduct(k2c, k2c))
        return num/den

"""
class cortesAlignment:

    def _centeredKernel(K): # K^c

        s = K.shape
        N = s[0]
        One = np.ones((s))
        One_One_T = np.dot(One, One.T)

        return K - 1/N * np.dot(One_One_T, K) - 1/N * np.dot(K, One_One_T) + 1/(N*N) * np.dot(np.dot(np.dot(One.T, K), One), One_One_T)


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


    def computeEta(K_list, IK):

        K_c_list = [self._centeredKernel(K) for K in K_list]
        M = self._kernelSimilarityMatrix(K_c_list)
        a = self._idealSimilarityVector(K_c_list, IK)
        num = np.dot(np.linalg.inv(M), a)
        return num / np.linalg.norm(num)


    def score(k1, k2):  #ex cortesAlignment
        
        k1c = centeredKernel(k1)
        k2c = centeredKernel(k2)

        num = ut.frobeniusInnerProduct(k1c, k2c)
        den = np.sqrt(ut.frobeniusInnerProduct(k1c, k1c) * ut.frobeniusInnerProduct(k2c, k2c))
        return num/den
    
    
    def copy():    
        return centeredKernelAlignment()
"""