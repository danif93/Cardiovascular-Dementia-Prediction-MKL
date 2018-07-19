import numpy as np
from KernelFile import kernelWrapper
import itertools
from operator import itemgetter

class GridSearchCV:
    
    def __init__(self, estimator, param_grid):
        
        self.estimator = estimator
        self.Ktype_list = param_grid.keys()
        self.parameters_lists = param_grid.values()
       
    
    def fit(self, Xtr_list, IK):
        
        self.IK_ = IK
        self.Xtr_list_ = Xtr_list
        
        # all possible combination of parameters each one coming from a different kernel
        self.configuration_list_ = [list(elem) for elem in itertools.product(*self.param_grid)]
        
        # all possible combination of kernel configurations across the datasets
        for i in range(len(Xtr_list)-1):
            if i == 0:
                self.configuration_list_ = [list(elem) for elem in itertools.product(*[self.configuration_list_, [list(elem) for elem in itertools.product(*self.param_grid)]])]
            else:
                self.configuration_list_ = [list(elem[0]+[elem[1]]) for elem in itertools.product(*[self.configuration_list_, [list(elem) for elem in itertools.product(*l)]])]
        
        # list of kernels_wrappers. To each possible configuration of the hyperparameter a kernels_wrapper is given 
        self.k_wrap_list_ = []
        for config in self.configuration_list_:
            self.k_wrap_list_.append(kenrelWrapper(self.Xtr_list_, self.Ktype_list, config))
            
        return self
                

        
    def transform(self, Xtr_list):
        
        self.performances_ = [] #list of 3-tuples. x[0] = alignment, x[1] = param configuration, x[2] = eta vector
        for kw_wrap in self.k_wrap_list_:
            
            # compute the list of kernels generated from the hyperparameter configuration at hand
            kw_wrap.kernelMatrix(self.Xtr_list_)
            # compute eta vector
            kernelMatrix_list = kw_wrap.kernelMatrix_list_
            eta = self.estimator.computeEta(kernelMatrix_list, self.IK)
            # compute k_eta (approximation)
            k_eta = np.zeros(kernelMatrix_list[0].shape)
            for eta_i, Ki in zip(eta, kernelMatrix_list):
                k_eta += eta_i * ki
            # compute performances estimation    
            self.performances_.append((self.estimator.score(k_eta, self.IK), kw_wrap.config, eta))
        
        return max(self.performances_, key=itemgetter(0)) # get the best triplet CA, config, eta vector looking at CA
            