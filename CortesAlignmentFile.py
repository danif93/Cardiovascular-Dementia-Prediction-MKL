import numpy as np
import Utils as ut

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
