import numpy as np


class kernelWrapper:
    
    def __init__(self, Xtr_list, Ktype_list, config):
        
        self.Xtr_list = Xtr_list
        self.Ktype_list = Ktype_list
        self.config = config
        
        self._k_list = []
        for dataset_index, Xtr in enumerate(Xtr_list):
            for kernel_index, Ktype in enumerate(Ktype_list):
                self._k_list.append(kernel(Xtr, Ktype, config[kernel_index * (dataset_index + 1)]))
                
                
    def kernelMatrix(self, X_list):
        
        if len(X_list) != len(self.Xtr_list):
            raise Exception("X_list and Xtr_list length differ")
            
        self.kernelMatrix_list_ = []
        for dataset_index, X in enumerate(X_list):
            for kernel_index, Ktype in enumerate(Ktype_list):
                    self.kernelMatrix_list_.append(self._k_list[kernel_index * (dataset_index + 1)].kernelMatrix())
                



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
    
