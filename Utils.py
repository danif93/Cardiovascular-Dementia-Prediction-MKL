from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import polynomial_kernel
import math as mt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet

from multiprocessing import Process
from queue import Queue

#----------------------------------------
# PREPROCESSING

def value_strategy(v, strategy):
    if strategy == 'min':
        return np.min(v)
    if strategy == 'mean':
        return np.mean(v)
    elif strategy == 'median':
        return np.median(v)
    elif strategy == 'most_frequent':
        return Counter(v).most_common(1)[0][0]


def imputing(data, strategy, axis, exception = [], floor = False):

    for i, e in enumerate(exception):
        exception[i] = np.where(data.columns == e)[0][0]

    for j in range(data.shape[axis]):

        if j in exception:
            continue

        indices = np.where(np.isnan(data.iloc[:,j]))[0] #tuple row, col

        if len(indices) == 0:
            #print("skipped {}".format(j))
            continue

        available = data.iloc[~indices, j]
        value = value_strategy(available, strategy)

        if floor:
            value = np.floor(value)

        data.iloc[indices, j] = value;
    return data


def oneHotEncoder(v):

    ohe = LabelBinarizer()
    enc = ohe.fit_transform(v)
    binary = []
    for r in enc:

        row = ''

        for c in r:
            row += str(c)

        binary.append(int(row))

    return np.asarray(binary)

# END PREPROCESSING

#-------------------------------------------

# GENERAL UTIL FUNCTIONS


def frobeniusInnerProduct(A, B):
    A = np.matrix.conjugate(A).ravel()
    B = B.ravel()
    return np.dot(A, B)


def normalization(X, norm = 'l2'):
    return normalize(X, norm = norm)

class kernelMultiparameter: # interface class to simulate a kernel which can deal with an interval of feasible parameters
    
    def __init__(self, X, K_type, param, dataset_name = 'D0'): #param = degree or sigma

        self.Xtr = X
        self.K_type = K_type
        self.dataset_name = dataset_name
        self.k_list = []
        
        for p in param:
            k_list.append(kernel(X, K_type, p))
            
        self.K_list = []

    
    def kM_child(k, X): # porcess function of kernelMatrix
        k.kernelMatrix(X)
        
    def kernelMatrix(self, X): # ask to all the kernel to compute the similarity matrix in parallel
        
        jobs = []
        for k in self.k_list:
            proc = Process(target=kM_child, args=((k, X),))
            jobs.append(proc)
            proc.start()
        
        for proc in jobs:
            proc.join()

    def gKM_child(k, queue):
        
        K = k.getKernelMatrix()
        param = k.getParam()
        queue.put([K, param])
            
    def getKernelMatrices(self):
        
        queue = Queue()
        jobs = []
        for k in self.k_list:
            proc = Process(target=gKM_child, args=((k, queue),))
            jobs.append(proc)
            proc.start()
        
        for proc in jobs:
            proc.join()
            
        info = [self.dataset_name, self.K_type]
        
        while ~queue.empty():
            info.append(queue.get())
            
        return info
            
        
        

class kernel:

    def __init__(self, X, K_type, param = 3): #param = degree or sigma

        self.Xtr = X
        self.K_type = K_type
        self.param = param

    def kernelMatrix(self, X):

        if self.K_type == 'linear':
            self.K = np.dot(X, self.Xtr.T)
            return  self.K

        if self.K_type == 'polynomial':
            # return polynomial_kernel(X, self.Xtr, degree=self.param)
            self.K = np.power(np.dot(X, self.Xtr.T)+1, self.param)
            return  self.K

        if self.K_type == 'gaussian':
            self.K = np.zeros((X.shape[0], self.Xtr.shape[0]))
            for i, sample_tr in enumerate(self.Xtr):
                for j, sample in enumerate(X):
                    d = np.linalg.norm(sample_tr-sample) ** 2
                    self.K[j, i] = np.exp(-d/(2*self.param*self.param))
            return  self.K
        
    

    def getType(self):
        return self.K_type

    def getParam(self):
        return self.param
    
    def setParam(self, param):
        self.param = param
        
    def getMatrix(self):
        return self.Xtr
    
    def getKernelMatrix(self):
        return self.K





def getParamInterval(kernel):
    param = kernel.getParam()
    g_step = param/5
    k_type = kernel.getType()
    return np.arange(np.max([param-(g_step*2), g_step]), param+(g_step*2), g_step) if k_type=='gaussian' else np.arange(np.max([param-3,1]), param+3, 1)

def getKernelList(wrapper):
    k_train_list = []
    for kernel_wrapp in wrapper:
        k_train_list.append(kernel_wrapp['kernel'].kernelMatrix(kernel_wrapp['train_ds']))
    return k_train_list

def kernelMatrixSum(wrapper, weights, size, kind='train_ds'):
    k_sumMat = np.zeros([size, size])
    # sum of all kernel train matrix
    for kernel_wrapp, w in zip(wrapper, weights):
        k_sumMat += kernel_wrapp['kernel'].kernelMatrix(kernel_wrapp[kind])*w
    return k_sumMat

# END GENERAL UTIL FUNCTIONS

#-------------------------------------------

# SOLA AKW pg 22-23 (Similarity Optimizing Linear Approach with Arbitrary Kernel Weights)
# Cortes approach

def centeredKernel(K): # K^c

    s = K.shape
    N = s[0]
    One = np.ones((s))

    return K - 1/N * np.dot(np.dot(One, One.T), K) - 1/N * np.dot(np.dot(K, One), One.T) + 1/(N*N) * np.dot(np.dot(np.dot(np.dot(One.T, K), One), One), One.T)


def kernelSimilarityMatrix(K_list): # M

    M = np.zeros((len(K_list), len(K_list)))

    for i, K1 in enumerate(K_list):
        for j, K2 in enumerate(K_list[i:]):

            s = frobeniusInnerProduct(K1, K2)
            M[i, i+j] = s

            if j != 0:
                M[i+j, i] = s

    return M


def idealSimilarityVector(K_list, y): # a

    a = np.zeros((len(K_list)))
    IK = np.dot(y.reshape(-1,1), y.reshape(-1,1).T) # ideal kernel

    for i, K in enumerate(K_list):
        a[i] = frobeniusInnerProduct(K, IK)

    return a


def centeredKernelAlignment(K_list, y):

    K_c_list = [centeredKernel(K) for K in K_list]

    M = kernelSimilarityMatrix(K_c_list)

    a = idealSimilarityVector(K_c_list, y)

    num = np.dot(np.linalg.inv(M), a)

    return num / np.linalg.norm(num)

"""
def cortesAlignment(k1, k2):
    k1c = centeredKernel(k1)
    k2c = centeredKernel(k2)

    num = frobeniusInnerProduct(k1c, k2c)
    den = np.sqrt(frobeniusInnerProduct(k1c, k1c)*frobeniusInnerProduct(k2c, k2c))
    return num/den
"""

def centeredKernelAlignmentCV(dict_kernel_param, dataset_list, y): 
    #dict_kernel_param = ['kernel_type'] -> param_list
    
    k_objects_list = []    # list of lists. every list is referred to a dataset and contains all the kernel object
    for X in dataset_list:
        k_objects_list_detaset = []
        for dkp in dict_kernel_param.items():
            k_objects_list_detaset.append(kernelMultiparameter(X, dkp[0], dkp[1]).kernelMatrix(X))
            
        k_objects_list.append(k_objects_list_detaset)
        
    
    K_list = [] # list of lists of lists. first enty is referred to the dataset, the second to the kernel type.
                # the last list has length num param +2 and 
                # is in the form [dataset name, kernel type, (kernel using param 1, param 1), 2, 3, ...]
    for k_objects_list_detaset in k_objects_list:
        K_list_dataset = []
        for k in k_objects_list_detaset:
            K_list_dataset.append(k.getMatrices())
            
        K_list.append(K_list_dataset)
        
        
    #TODO parallelized combinatorial association of kernels
    

"""
def parameterOptimization(k_dataset_wrapper, train_label, n_epoch=20, tol=0.01, kind='train_ds', verbose=False):

    previousCA = -1
    k_train_list = getKernelList(k_dataset_wrapper)
    weights = centeredKernelAlignment(k_train_list, train_label)

    for i in range(n_epoch):

        k_sumTrain = kernelMatrixSum(k_dataset_wrapper, weights, len(train_label), kind=kind)
        currentCA = cortesAlignment(k_sumTrain, np.dot(train_label.reshape(-1,1), train_label.reshape(-1,1).T))
        if verbose: print('epoch num {}; current CA is: {}'.format(i+1, currentCA))

        if previousCA>0 and currentCA>0 and np.abs(previousCA-currentCA)<tol: break
        else: previousCA = currentCA

        for kernel_idx in np.argsort(weights): # start optimizing the most impactful kernel parameter
            if verbose: print('\toptimizing {}'.format(kernel_idx))
            kernel = k_dataset_wrapper[kernel_idx]['kernel']

            if (kernel.getType()=='linear'): continue

            param_interval = getParamInterval(kernel)
            if verbose: print('\t\toptimizing over [{},{}]'.format(param_interval[0], param_interval[-1]))

            similarity_grid = np.zeros(len(param_interval))
            old_param = k_dataset_wrapper[kernel_idx]['kernel'].getParam()

            for p_idx, param in enumerate(param_interval):
                k_dataset_wrapper[kernel_idx]['kernel'].setParam(param)

                k_sumTrain = kernelMatrixSum(k_dataset_wrapper, weights, len(train_label), kind=kind)

                similarity_grid[p_idx] = cortesAlignment(k_sumTrain,
                                                         np.dot(train_label.reshape(-1,1), train_label.reshape(-1,1).T))

            selected = np.argmax(similarity_grid)

            if similarity_grid[selected] > currentCA:
                currentCA = similarity_grid[selected]
                k_dataset_wrapper[kernel_idx]['kernel'].setParam(param_interval[selected])

                # update the weights with the new configuration
                k_train_list = getKernelList(k_dataset_wrapper)
                weights = centeredKernelAlignment(k_train_list, train_label)

                if verbose: print('\t\tselected {} with sim: {}'.format(param_interval[selected], currentCA))

            else:
                k_dataset_wrapper[kernel_idx]['kernel'].setParam(old_param)
                if verbose: print('\t\tkept {} with sim: {}'.format(kernel.getParam(), currentCA))

    return k_dataset_wrapper
"""
# END SOLA AKW

#-------------------------------------------------------

# MY SOLA NKW

class myMKL_srola:

    def __init__(self, X_list, K_type_list, y):

        # X_list: datasets list
        # K_name_list: names of kernels to use
        # y: ideal output vector

        self.y = y #used later in learning
        self.IK = np.dot(y, y.T) # used later in learning
        self.error = -1 #used later in learning
        self.Xtr_list = X_list
        self.num_datasets = len(X_list)
        self.num_K_types = len(K_type_list)
        self.num_samples = self.X_list[0].shape[0]

        self.eta = np.random.rand(self.num_datasets)
        self.lamb = np.random.rand(self.num_datasets, self.num_K_types)
        self.mu_list = [] # list of lists of matrices. every matrix refers to a kernel and to all its detasets


         #---------------------------------------------------------------------------------------------
         # get kernel objects and kernel matrices
        self.K_objects_list = [] # list of lists of objects that create the kernel matrices. first list = kernel objets list of first detaset
        self.K_list = [] # list of lists of kernels. first list associated to first dataset

        for X in X_list:
            dataset_kernel_objects_list = []
            dataset_kernel_list = []
            for K_type in K_type_list:

                k = kernel(X, self.num_K_type)  #small k object, big k matrix
                dataset_kernel_objects_list.append(k)
                dataset_kernel_list.append(k.kernelMatrix(X))

            self.K_objects_list.append(dataset_kernel_objects_list)
            self.K_list.append(dataset_kernel_list)
          #---------------------------------------------------------------------------------------------
          # randomly initialize mu vectors. every vector length depends on the kernel features. then we have a matrix of mu per kernel type
        for K in self.K_list[0]:
            self.mu_list.append(np.ones(self.num_datasets, K.shape[1]))


    def learning(self, tol = 0.01):

        while(True):
            self.learnMu()
            self.learnK()
            self.learnLambda()
            self.learnEta()
            error = self.computeError()
            if self.error < 0:
                continue
            if np.abs(error-self.error) < tol:
                break


    def getParam(self):

        return self.K_objects_list, self.mu_list, self.lamb, self.eta


    def test(self, X_list, goal = 'classification'):
        #X_list = test set

        K_test_list = []

        for dataset_index, X in enumerate(X_list):
            K_test_dataset_list = []
            for k_index, mu_k in enumerate(self.mu_list):
                tmp_X = X[:, np.where(mu_k[dataset_index] != 0)]
                K_test_dataset_list.append(self.k_objects_list[i][k_index].kernelMatrix(tmp_X))

            K_test_list.append(K_test_dataset_list)


        approximation = self.computeApproximation(K_test_list)
        if goal == 'classification':
            self.y_pred = classify(approximation, self.y)
        else:
            self.y_pred = regression(approximation)


    def computeApproximation(self, kernel_list):

        approximation = 0

        for dataset, K_list in enumerate(kernel_list):
            vec = np.zeros((self.num_samples))
            for kernel, K in enumerate(K_list):
                  vec += self.lamb[dataset, kernel] * K


            dataset_vec.append(vec)
        for i, d in enumerate(dataset_vec):
            approximation += self.eta[i] * d


        return approximation


    def learnMu(self):

        for k_index, mu_k in enumerate(self.mu_list):
            for d_index, mu in enumerate(mu_k):
                X = self.Xtr_list[d_index]
                alphas = np.arange(0.001, 3, 0.007)
                cl = LassoCV(alphas = alphas)
                mu = cl.fit(X, self.y).coef_
                mu = np.where(mu == 0, 0, 1)
                k = self.k_objects_list[d_index][k_index]
                k = kernel(X[:, np.where(mu != 0)], k.getType(), k.getParam())
                self.k_objects_list[d_index][k_index] = k
                self.K_list[d_index][k_index] = k.kernelMatrix(k.getMatrix())


    def learnK(self):

        apprximation = self.computeApproximation(self.K_list)
        actualError = frobeniusInnerProduct(approximation - self.IK, approximation - self.IK)
        tmp_K_list = self.K_list

        for i, k_list in enumerate(self.K_objects_list):
            for j, k in enumerate(k_list):
                if k.getType == 'linear':
                    continue
                interval = getParamInterval(k.getParam, k.getType)
                score = []
                matrices = []
                for param in interval:
                    matrices.append(k.setParam(param).kernelMatrix(k.getMatrix()))
                    tmp_K_list[i][j] = matrices[-1]
                    tmp_approximation = self.computeApproximation(tmp_K_list)
                    score.append(frobeniusInnerProduct(tmp_approximation - self.IK, tmp_approximation - self.IK))

                if actualError > np.min(score):
                    actualError = np.min(score)
                    tmp_K_list[i][j] = matrices[np.argmin(score)]
                else:
                    tmp_K_list[i][j] = self.K_list[i][j]

        self.K_list = tmp_K_list


    def learnLambda(self):

        for dataset_index, lamb in enumerate(self.lamb):

            C = np.zeros((self.K_list[0][0].shape[0], self.K_list[0][0].shape[0], len(self.K_list[0])))
            eta_ = 1/self.eta[dataset_index]
            for i, Ki in enumerate(self.K_list[dataset_index]):
                C[:,:,i] = Ki*eta_



            for dataset_index, K_list_dataset in enumerate(self.K_list):
                Di = np.zeros((self.num_samples, self.num_samples))
                for kernel, K in enumerate(K_list_dataset):
                    Di += K * self.lamb[dataset, kernel]

                C[:,:,i] = Di

            fixed_approximation = np.zeros(self.IK.shape)
            for d_index, K_list_dataset in enumerate(self.K_list):
                if d_index == dataset_index:
                    continue

                tmp_approximation = np.zeros(self.IK.shape)
                for kernel_index, Ke in enumerate(K_list_dataset):
                    tmp_approximation += Ke * self.lamb[d_index, kernel_index]

                fixed_approximation += self.eta[d_index] * tmp_approximation

            Y = self.IK - fixed_approximation

            A = np.zeros((len(self.K_list[0]), len(self.K_list[0])))
            B = np.zeros((len(self.K_list[0])))
            for row in C:
                for elem in row:
                   B += elem
                   A += np.dot(elem, elem.T)

            A += np.identity(A.shape[0]) #np.identity(...) * gamma_lambda for cross validation
            self.lamb[dataset_index, :] = np.dot(np.linalg.inv(A), np.dot(B, Y)) # TODO control dimensions
            """
            K_ = [] #  K'
            for Ki in K:
                K_.append(Ki+ np.identity(Ki.shape[0])) #np.identity(...) * gamma_lambda for cross validation

            KTK = np.zeros((len(K_), len(K_), K_[0].shape[0], K_[0].shape[1]) # K'_ij
            for i, Ki in enumerate(K_):
                for j, Kj in enumerate(K_):
                    KTK[i, j, :, :] = np.dot(Ki, Kj)


            fixed_approximation = np.zeros(self.IK.shape)
            for d_index, K_list_dataset in enumerate(self.K_list):
                if d_index == dataset_index:
                    continue

                tmp_approximation = np.zeros(self.IK.shape)
                for kernel_index, Ke in enumerate(K_list_dataset):
                    tmp_approximation += Ke * self.lamb[d_index, kernel_index]

                fixed_approximation += self.eta[d_index] * tmp_approximation

            B = self.IK - fixed_approximation


            # HARD CODED

            K11_inv = np.linalg.inv(KTK[0,0,:,:])
            K21_K11 = np.dot(KTK[1,0,:,:], K11_inv)
            X = np.dot(K[1], self.IK) - np.dot(K21_K11, np.dot(K[0], self.IK))
            Y = KTK[1,2,:,:] - np.dot(K21_K11, KTK[0,2,:,:])
            Z = KTK[1,1,:,:] - np.dot(K21_K11, KTK[0,1,:,:])
            Z_inv = np.linalg.inv(Z)
            P = np.dot(K11_inv, (np.dot(K[0], self.IK) - np.dot(KTK[0,1,:,:], np.dot(X, Z_inv))))
            Q = np.dot(K11_inv, (np.dot(KTK[0,1,:,:], np.dot(Y, Z_inv)) - KTK[0,2,:,:]))
            M = np.dot(KTK[2,0,:,:], Q) - np.dot(KTK[2,1,:,:], np.dot(Y, Z_inv)) + KTK[2,2,:,:]
            M_inv = np.linalg.inv(M)
            N = np.dot(K[2], self.IK) - np.dot(KTK[2,0,:,:], P) - np.dot(KTK[2,1,:,:], np.dot(X, Z_inv))

            M_inv_N = np.dot(M_inv, N)
            self.lamb[dataset_index, 2] = M_inv_N[0,0]
            self.lamb[dataset_index, 0] = (P + np.dot(Q, M_inv_N))[0,0]
            self.lamb[dataset_index, 1] = np.dot((X - np.dot(Y, M_inv_N))), Z_inv)[0,0] """


    def learnEta(self):

        C = np.zeros((self.K_list[0][0].shape[0], self.K_list[0][0].shape[0], len(self.Xtr_list)))
        for dataset_index, K_list_dataset in enumerate(self.K_list):
            Di = np.zeros((self.num_samples, self.num_samples))
            for kernel, K in enumerate(K_list_dataset):
                Di += K * self.lamb[dataset, kernel]

            C[:,:,i] = Di

        Y = self.IK.ravel()
        A = np.zeros((len(self.Xtr_list), len(self.Xtr_list)))
        B = np.zeros((len(self.Xtr_list)))
        for row in C:
            for elem in row:
               B += elem
               A += np.dot(elem, elem.T)

        A += np.identity(A.shape[0]) #np.identity(...) * gamma_eta for cross validation
        self.eta = np.dot(np.linalg.inv(A), np.dot(B, Y)) # TODO control dimensions

        """
        D_ = [] #  D'
        for Di in D:
            D_.append(Di+ np.identity(Di.shape[0])) #np.identity(...) * gamma_eta for cross validation

        DTD = np.zeros((len(D_), len(D_), D_[0].shape[0], D_[0].shape[1]) # D'_ij
        for i, Di in enumerate(D_):
            for j, Dj in enumerate(D_):
                DTD[i, j, :, :] = np.dot(Di, Dj)

        # HARD CODED

        D11_inv = np.linalg.inv(DTD[0,0,:,:])
        D21_D11 = np.dot(DTD[1,0,:,:], D11_inv)
        X = np.dot(D[1], self.IK) - np.dot(D21_D11, np.dot(D[0], self.IK))
        Y = DTD[1,2,:,:] - np.dot(D21_D11, DTD[0,2,:,:])
        Z = DTD[1,1,:,:] - np.dot(D21_D11, DTD[0,1,:,:])
        Z_inv = np.linalg.inv(Z)
        P = np.dot(D11_inv, (np.dot(D[0], self.IK) - np.dot(DTD[0,1,:,:], np.dot(X, Z_inv))))
        Q = np.dot(D11_inv, (np.dot(DTD[0,1,:,:], np.dot(Y, Z_inv)) - DTD[0,2,:,:]))
        M = np.dot(DTD[2,0,:,:], Q) - np.dot(DTD[2,1,:,:], np.dot(Y, Z_inv)) + DTD[2,2,:,:]
        M_inv = np.linalg.inv(M)
        N = np.dot(D[2], self.IK) - np.dot(DTD[2,0,:,:], P) - np.dot(DTD[2,1,:,:], np.dot(X, Z_inv))

        M_inv_N = np.dot(M_inv, N)
        self.eta[2] = M_inv_N[0,0]
        self.eta[0] = (P + np.dot(Q, M_inv_N))[0,0]
        self.eta[1] = np.dot((X - np.dot(Y, M_inv_N))), Z_inv)[0,0]"""
