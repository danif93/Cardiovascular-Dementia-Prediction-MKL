import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel, sigmoid_kernel

from sklearn.preprocessing import normalize


class kernelWrapper:
    
    def __init__(self, Xtr_list, Ktype_list, config, normalize = False):
                
        self.Xtr_list = Xtr_list
        self.Ktype_list = Ktype_list
        self.config = config
        self.normalize = normalize
        
        self._k_list = []
        for dataset_index, Xtr in enumerate(Xtr_list):
            for kernel_index, Ktype in enumerate(Ktype_list):
                if len(Xtr_list) > 1:
                    self._k_list.append(kernel(Xtr, Ktype, config[dataset_index][kernel_index], normalize = self.normalize))
                else:
                    self._k_list.append(kernel(Xtr, Ktype, config[kernel_index], normalize = self.normalize))
                
    def kernelMatrix(self, X_list):
        num_datasets = len(X_list)
        if num_datasets != len(self.Xtr_list):
            raise Exception("X_list and Xtr_list length differs: # datasets not the same")
            
        self.kernelMatrix_list_ = []
            
        for dataset_index, X in enumerate(X_list):
            for kernel_index, Ktype in enumerate(self.Ktype_list):
                self.kernelMatrix_list_.append(self._k_list[kernel_index+dataset_index*len(self.Ktype_list)].kernelMatrix(X))
                    
        return self
    
    def predict(self, Xts_list, weights, tr_label, estimator, Ptype = 'classification'):
        
        k_test_list = self.kernelMatrix(Xts_list).kernelMatrix_list_
        
        K_eta = sum(eta*X for eta, X in zip(weights, k_test_list))        
        
        K_eta_c = estimator.centeredKernel(K_eta)
        
        if Ptype == 'classification':
            pred = np.zeros(k_test_list[0].shape)
            for idx, row in enumerate(K_eta_c):
                pred[idx, :] += np.multiply(row, tr_label)
                
            return np.sign(np.sum(pred, axis = 1))
        
        else:
            pred = np.zeros(K_eta_c.shape[0])
            for idx, row in enumerate(K_eta_c):
                pred[idx] += np.dot(row, tr_label)/np.sum(row)
                
            return pred
        

    def getConfig(self):
        
        config = {}
        for k in self._k_list:
            try:
                config[k.K_type].append(k.param)
            except:
                config[k.K_type] = []
                config[k.K_type].append(k.param)
                
        return config
        
    def printConfig(self):
        strOut = ""
        for k_idx, kernel in enumerate(self._k_list):
            if k_idx%len(self.Ktype_list)==0 and k_idx!=0: strOut+="]\n"
            if k_idx%len(self.Ktype_list)==0: strOut+="["
            strOut += kernel.K_type+":"+str(kernel.param)+", "
        return strOut+"]\n"


            
class kernel:

    def __init__(self, X, K_type, param = None, normalize = False): #param = degree or sigma
        
        if param==None: raise ValueError("Kernel parameter not set properly")

        self.Xtr = X
        self.K_type = K_type
        self.param = param
        self.normalize = normalize
        if K_type == 'linear':
            self_mu = None
        

    def kernelMatrix(self, X, y = None):

        if self.K_type == 'linear':
            """
            if y != None:
                if self.mu == None:
                    reg = Lasso(self.param) #TODO change with a model for classification and let the possibility to specify regression or classification
                    self_mu = reg.fit(X, y).coef_
                    self.Xtr = self.Xtr[:, mp.where(self_mu != 0)]

                self.X = self.X[:, mp.where(self_mu != 0)]
            """
            if self.normalize:
                self.K = normalize(linear_kernel(X,self.Xtr))
            else:
                self.K = linear_kernel(X,self.Xtr)
                
            return  self.K

        if self.K_type == 'polynomial':
            if self.normalize:
                self.K = normalize(polynomial_kernel(X, self.Xtr, degree=self.param))
            else:
                self.K = polynomial_kernel(X, self.Xtr, degree=self.param)
                
            return  self.K

        if self.K_type == 'gaussian':
            if self.normalize:
                self.K = normalize(rbf_kernel(X, self.Xtr, gamma=self.param))
            else:
                self.K = rbf_kernel(X, self.Xtr, gamma=self.param)
                
            return  self.K
        
        if self.K_type == 'laplacian':
            if self.normalize:
                self.K = normalize(laplacian_kernel(X, self.Xtr, gamma=self.param))
            else:
                self.K = laplacian_kernel(X, self.Xtr, gamma=self.param)
                
            return self.K
            
        if self.K_type == 'sigmoid':
            if self.normalize:
                self.K = normalize(sigmoid_kernel(X, self.Xtr, gamma=self.param))
            else:
                self.K = sigmoid_kernel(X, self.Xtr, gamma=self.param)
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