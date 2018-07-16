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


def normalization(X, norm = 'l2):
    
    return normalize(X, norm = norm)
                  
                  
def kernel(X, K_func):
                  
    return k_func(X)


# END GENERAL UTIL FUNCTIONS

#-------------------------------------------

# SOLA AKW pg 22-23 (Similarity Optimizing Linear Approach with Arbitrary Kernel Weights)
# Cortes approach

def centeredKernel(K): # K^c
    
    s = K.shape
    N = shape[0]
    One = np.ones((s))
    
    return K - 1/N * One * One.T * K - 1/N * K * One * One.T + 1/(N*N) * (One.T * K * One) * One * One.T


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
                  
 def myMKL_srola(X_list, K_name_list, y):
                  
         # X_list: datasets list
         # K_name_list: names of kernels to use
         # y: ideal output vector
                  
         num_datasets = len(X_list)
         K_types = len(K_name_list)

         eta = np.random.rand(num_datasets)
         lamb = np.random.rand(num_datasets, K_types)
         mu_list = [] # list of matrices. every matrix refers to a kernel and to all the datasets
         # TODO initialize mu_list