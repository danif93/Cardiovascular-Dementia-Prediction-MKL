import numpy as np

from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel, sigmoid_kernel


class kernelWrapper:
    
    def __init__(self, Xtr_list, Ktype_list, config):
                
        self.Xtr_list = Xtr_list
        self.Ktype_list = Ktype_list
        self.config = config
        
        self._k_list = []
        for dataset_index, Xtr in enumerate(Xtr_list):
            for kernel_index, Ktype in enumerate(Ktype_list):
                self._k_list.append(kernel(Xtr, Ktype, config[dataset_index][kernel_index]))
                
    def kernelMatrix(self, X_list):
        num_datasets = len(X_list)
        if num_datasets != len(self.Xtr_list):
            raise Exception("X_list and Xtr_list length differs: #datasets not the same")
            
        self.kernelMatrix_list_ = []
            
        for dataset_index, X in enumerate(X_list):
            for kernel_index, Ktype in enumerate(self.Ktype_list):
                self.kernelMatrix_list_.append(self._k_list[kernel_index+dataset_index*len(self.Ktype_list)].kernelMatrix(X))
        return self
    
    def predict(self, Xts_list, weights, tr_label):
        
        k_test_list = self.kernelMatrix(Xts_list).kernelMatrix_list_
        
        pred = np.zeros(len(Xts_list[0]))
        
        for k_test, w in zip(k_test_list, weights):
            weighted_labeled_kernel = np.multiply(w*k_test.T, tr_label).T
            pred += np.sum(weighted_labeled_kernel, axis=0)
        pred = np.sign(pred)
        return pred
    """  
    def printConfig(self)
        strOut = ""
        for kernel in self._k_list:
            strOut += kernel.K_type
    """            

class kernel:

    def __init__(self, X, K_type, param = None): #param = degree or sigma
        
        if param==None: raise ValueError("Kernel parameter not set properly")

        self.Xtr = X
        self.K_type = K_type
        self.param = param
        

    def kernelMatrix(self, X):

        if self.K_type == 'linear':
            self.K = linear_kernel(self.Xtr, X)
            return  self.K

        if self.K_type == 'polynomial':
            self.K = polynomial_kernel(self.Xtr, X, degree=self.param)
            return  self.K

        if self.K_type == 'gaussian':
            self.K = rbf_kernel(self.Xtr, X, gamma=self.param)
            return  self.K
        
        if self.K_type == 'laplacian':
            self.K = laplacian_kernel(self.Xtr, X, gamma=self.param)
            return self.K
            
        if self.K_type == 'sigmoid':
            self.K = sigmoid_kernel(self.Xtr, X, gamma=self.param)
            return self.K
        
    
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
    
"""
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
"""