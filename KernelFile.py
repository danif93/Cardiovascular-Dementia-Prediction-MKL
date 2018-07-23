import numpy as np
from sklearn.linear_model import Lasso
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
    
    def predict(self, Xts_list, weights, tr_label, Ptype = 'classification'):
        
        k_test_list = self.kernelMatrix(Xts_list).kernelMatrix_list_
        
        K_eta = sum(eta*X for eta, X in zip(weights, k_test_list))        
        
            
        if Ptype == 'classification':
            pred = np.zeros(k_test_list[0].shape)
            for idx, row in enumerate(K_eta):
                pred[idx, :] += np.multiply(row, tr_label)
                
            return np.sign(np.sum(pred, axis = 1))
        
        else:
            pred = np.zeros(k_test_list[0].shape[0])
            for idx, row in enumerate(K_eta):
                pred[idx, :] += np.sum(np.multiply(row, tr_label))/np.sum(row)
                
            return pred
        
    
    def accuracy(self, Xts_list, weights, tr_label, test_labels, test_pred = None):
        
        if test_pred == None:
            test_pred = self.predict(Xts_list, weights, tr_label)
        return np.mean(np.absolute(test_pred-test_labels))/2
    
    def precision(self, Xts_list, weights, tr_label, test_labels, test_pred = None):
        
        if test_pred == None:
            test_pred = self.predict(Xts_list, weights, tr_label)
        
        s = test_pred + test_labels
        
        if np.where(s == 2) == np.asarray([]):
            return 0
        
        d = test_pred - test_labels
        TP = len(np.where(s == 2))
        FP = len(np.where(d == 2))
        return TP/(TP+FP)

    def recall(self, Xts_list, weights, tr_label, test_labels, test_pred = None):
        
        if test_pred == None:
            test_pred = self.predict(Xts_list, weights, tr_label)
            
        s = test_pred + test_labels
        
        if np.where(s == 2) == np.asarray([]):
            return 0
        
        d = test_pred - test_labels
        TP = len(np.where(s == 2))
        FN = len(np.where(d == -2))
        return TP/(TP+FN)

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
            if k_idx%len(self.Xtr_list)==0 and k_idx!=0: strOut+="]\n"
            if k_idx%len(self.Xtr_list)==0: strOut+="["
            strOut += kernel.K_type+":"+str(kernel.param)+", "
        return strOut+"]\n"


            
class kernel:

    def __init__(self, X, K_type, param = None): #param = degree or sigma
        
        if param==None: raise ValueError("Kernel parameter not set properly")

        self.Xtr = X
        self.K_type = K_type
        self.param = param
        if K_type == 'linear':
            self_mu = None
        

    def kernelMatrix(self, X, y = None):

        if self.K_type == 'linear':
            
            if y != None:
                if self.mu == None:
                    reg = Lasso(self.param)
                    self_mu = reg.fit(X, y).coef_
                    self.Xtr = self.Xtr[:, mp.where(self_mu != 0)]

                self.X = self.X[:, mp.where(self_mu != 0)]
                    
            self.K = linear_kernel(X,self.Xtr)
            return  self.K

        if self.K_type == 'polynomial':
            self.K = polynomial_kernel(X, self.Xtr, degree=self.param)
            return  self.K

        if self.K_type == 'gaussian':
            self.K = rbf_kernel(X, self.Xtr, gamma=self.param)
            return  self.K
        
        if self.K_type == 'laplacian':
            self.K = laplacian_kernel(X, self.Xtr, gamma=self.param)
            return self.K
            
        if self.K_type == 'sigmoid':
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