from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize


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

    A = np.matrix.conjugate(A)
    P = np.dot(A.T, B)
    _, s, _ = np.linalg.svd(P)
    return np.sum(s)


def normalization(X, norm = 'l2'):   
    return normalize(X, norm = norm)
                  

class kernel:
  
    def __init__(self, X, K_type, param = 3): #param = degree or sigma
                  
        self.Xtr = X
        self.K_type = K_type
        self.param = param
                                   
    def kernelMatrix(self, X):
                  
        if self.K_type == 'linear':
            return  np.dot(X, self.Xtr.T)
                  
        if self.K_type == 'polynomial':
            return np.power(np.dot(X, self.Xtr.T) + 1, self.param)
                  
        if self.K_type == 'gaussian':
            sim = np.zeros((self.Xtr.shape[0], self.Xtr.shape[0]))                 
            for i, sample_tr in enumerate(self.Xtr):
                for j, sample in enumerate(X):
                    d = np.linalg.norm(sample_tr-sample) ** 2
                    sim[i, j] = np.exp(-d/(2*self.param*self.param))                   
            return sim
                                   
    def getType(self):
        return self.K_type
    
    def getParam(self):
        return self.param
    
def getParamInterval(param, K_type):
    return np.arange(param-1.5, param+1.5, 0,5) if K_type=='polynomial' else np.arange(param-0.3, param+0.3, 0.1)
    


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
    IK = np.dot(y, y.T) # ideal kernel
    
    for i, K in enumerate(K_list):
        a[i] = frobeniusInnerProduct(K, IK)
      
    return a


def centeredKernelAlignment(K_list, y):
    
    K_c_list = []
    for K in K_list:
        K_c_list.append(centeredKernel(K))
    
    M = kernelSimilarityMatrix(K_c_list)
    
    a = idealSimilarityVector(K_c_list, y)
    
    num = np.dot(np.linalg.inv(M), a)
    
    return num / np.linalg.norm(num)

# END SOLA AKW

#-------------------------------------------------------

# MY SROLA NKW
                  
class myMKL_srola:

    def __init__(self, X_list, K_type_list, y):

        # X_list: datasets list
        # K_name_list: names of kernels to use
        # y: ideal output vector
         
        self.y = y #used later in learning
        self.error = -1 #used later in learning         
        self.Xtr_list = X_list
        self.num_datasets = len(X_list)
        self.num_K_types = len(K_type_list)
        self.num_samples = self.X_list[0].shape[0]

        self.eta = np.random.rand(self.num_datasets)
        self.lamb = np.random.rand(self.num_datasets, self.num_K_types)
        self.mu_list = [] # list of matrices. every matrix refers to a kernel and to all the datasets


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
            self.mu_list.append(np.random.rand(self.num_datasets, K.shape[1]))


    def learning(self, tol = 0.01):

        while(True):
            self.learnK()
            self.learnMu()
            self.learnLambda()
            self.learnEta()
            error = self.computeError()
            if self.error < 0:
                continue
            if np.abs(error-self.error) < tol:
                break
     
                  
    def getParam(self):

        return self.K_objects_list, self.mu_list, self.lamb, self.eta

                  
    #def test(self, X_list):
        #X_list = test set

        # TODO

                  
                  
    def learnK(self): # TO COMPLETE
            
        #sum_eta eta*lambda* sum_k k*mu
        
        dataset_vec = []
                  
        for dataset, K_list in enumerate(self.K_list):
            for kernel, K in enumerate(K_list):
                  dataset_vec.append(np.dot(self.lamb[dataset, :], np.dot(K, mu_list[kernel][dataset, :])))
               
            dataset_vec.append(vec)
                  
        approximation = np.dot(self.eta, ...)
        self.actualError = np.linalg.norm(self.y - approximation) ** 2