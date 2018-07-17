from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import polynomial_kernel
import math as mt

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
                  

class kernel:
  
    def __init__(self, X, K_type, param = 3): #param = degree or sigma
                  
        self.Xtr = X
        self.K_type = K_type
        self.param = param
                                   
    def kernelMatrix(self, X):
                  
        if self.K_type == 'linear':
            return  np.dot(X, self.Xtr.T)
                  
        if self.K_type == 'polynomial':
            # return polynomial_kernel(X, self.Xtr, degree=self.param)
            return np.power(np.dot(X, self.Xtr.T)+1, self.param)
                  
        if self.K_type == 'gaussian':
            sim = np.zeros((X.shape[0], self.Xtr.shape[0]))                 
            for i, sample_tr in enumerate(self.Xtr):
                for j, sample in enumerate(X):
                    d = np.linalg.norm(sample_tr-sample) ** 2
                    sim[j, i] = np.exp(-d/(2*self.param*self.param))                   
            return sim
                                   
    def getType(self):
        return self.K_type
    
    def getParam(self):
        return self.param
    def setParam(self, param):
        self.param = param
    
    
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

def cortesAlignment(k1, k2):
    k1c = centeredKernel(k1)
    k2c = centeredKernel(k2)
    
    num = frobeniusInnerProduct(k1c, k2c)
    den = np.sqrt(frobeniusInnerProduct(k1c, k1c)*frobeniusInnerProduct(k2c, k2c))
    return num/den

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