import numpy as np
from KernelFile import kernelWrapper
import itertools
from sklearn.model_selection import StratifiedKFold, KFold
import Utils as ut


class myGridSearchCV:
    
    def __init__(self, estimator, param_grid, fold=5, Ptype="classification", sparsity=0, lamb=0, normalize_kernels=False):
        
        self.estimator = estimator
        self.Ktype_list = param_grid.keys()
        self.parameters_lists = param_grid.values()
        self._fold = fold
        self.sparsity = sparsity
        self.lamb = lamb
        self.normalize_kernels = normalize_kernels
        self.Ptype = Ptype
        
        if  Ptype=="classification":
            self.kFolder = StratifiedKFold(n_splits=self._fold)
        else:
            self.kFolder = KFold(n_splits=self._fold)
       
    
    def fit(self, Xtr_list, y):
        
        self._y = y
        
        
        #NEW CODE: managing labels for a correct regression (normalizing labels such that E[y^2] = 1)
        if self.Ptype == 'regression':
            y /= np.linalg.norm(y)
        
        
        self.IK_ = np.outer(y, y)
        self.Xtr_list_ = Xtr_list
        
        # all possible combination of parameters each one coming from a different kernel
        self.configuration_list_ = [list(elem) for elem in itertools.product(*self.parameters_lists)]
                
        # all possible combination of kernel configurations across the datasets
        for i in range(len(Xtr_list)-1):
            if i == 0:
                self.configuration_list_ = [list(elem) for elem in itertools.product(*[self.configuration_list_, [list(elem) for elem in itertools.product(*self.parameters_lists)]])]
            else:
                self.configuration_list_ = [list(elem[0]+[elem[1]]) for elem in itertools.product(*[self.configuration_list_, [list(elem) for elem in itertools.product(*self.parameters_lists)]])]
            
        return self
                
        
    def transform(self, Xtr_list, verbose=False):
                
        performances = np.empty((self._fold, len(self.configuration_list_)))
        
        for fold_idx, (train_index, valid_index) in enumerate(self.kFolder.split(Xtr_list[0], self._y)):
            
            if verbose: print("Fold no. {}".format(fold_idx+1))
            
            self.train_list_ = [ Xtr[train_index] for Xtr in self.Xtr_list_]
            #self.valid_list_ = [ Xtr[valid_index] for Xtr in self.Xtr_list_]
            train_list = [ Xtr[train_index] for Xtr in Xtr_list]
            valid_list = [ Xtr[valid_index] for Xtr in Xtr_list]
            
            
            #-------------------------------
            #NEW CODE: managing labels for a correct regression (normalizing labels such that E[y^2] = 1)           
            trainLabel = self._y[train_index]
            validLabel = self._y[valid_index]
            
            if self.Ptype == 'regression':
                trainLabel /= np.linalg.norm(trainLabel)
                validLabel /= np.linalg.norm(validLabel)
            
            
            IK_tr = np.outer(trainLabel, trainLabel)
            IK_val = np.outer(validLabel, trainLabel)           
            #-------------------------------
            
            
            # list of kernels_wrappers. To each possible configuration of the hyperparameter a kernels_wrapper is given 
            self.k_wrap_list_ = []
            for config in self.configuration_list_:
                self.k_wrap_list_.append(kernelWrapper(self.train_list_, self.Ktype_list, config, normalize = self.normalize_kernels))
                 
            
            # cycle through configurations
            for kw_idx, kw_wrap in enumerate(self.k_wrap_list_):

                # compute the list of kernels generated from the hyperparameter configuration at hand
                kernelMatrix_list = kw_wrap.kernelMatrix(train_list).kernelMatrix_list_

                # compute eta vector
                eta = self.estimator.computeEta(kernelMatrix_list, IK_tr, y=trainLabel, sparsity=self.sparsity, lamb=self.lamb, verbose=verbose)

                # compute k_eta (approximation) for the validation set
                kernelMatrix_list = kw_wrap.kernelMatrix(valid_list).kernelMatrix_list_
                k_eta = np.zeros(kernelMatrix_list[0].shape)
                for eta_i, Ki in zip(eta, kernelMatrix_list):
                    k_eta += eta_i * Ki

                # compute performances estimation
                
                #-------------------------
                # NEW CODE FOR ABSOLUTE VALUE OF CA 
                score = self.estimator.score(k_eta, IK_val)
                if score < 0:
                    #score *= -1
                    eta = -1*eta
                pred = kw_wrap.predict(valid_list, eta, trainLabel, self.estimator, Ptype=self.Ptype)
                score = ut.balanced_accuracy_score(validLabel, pred)             
                #----------------------------------
                
                performances[fold_idx, kw_idx] = score
                
                if verbose and (kw_idx+1) % 200 == 0: print("\t\tPerfomances computed for {}".format(kw_idx+1))
                #if verbose: print("\t\tPerfomances computed for {}: {}".format(kw_idx, performances[-1]))

        
        # select the configuration with the highest alignment value across the whole validation procedure
        selected = np.argmax(np.mean(performances, axis=0))
        if verbose: print("Validation complete, config selected:{}".format(self.configuration_list_[selected]))
            
        # recompute the eta for the selected configuration
        k_wrap_best = kernelWrapper(self.Xtr_list_, self.Ktype_list, self.configuration_list_[selected], normalize = self.normalize_kernels)
        kernelMatrix_list = k_wrap_best.kernelMatrix(Xtr_list).kernelMatrix_list_
        
        #------------------------------------------
        #NEW CODE: managing labels for a correct regression (normalizing labels such that E[y^2] = 1)
        if self.Ptype == 'regression':
            eta = self.estimator.computeEta(kernelMatrix_list, self.IK_, y = self._y/np.linalg.norm(self._y), sparsity = self.sparsity, lamb = self.lamb, verbose = verbose)
        else:
            eta = self.estimator.computeEta(kernelMatrix_list, self.IK_, y = self._y , sparsity = self.sparsity, lamb = self.lamb, verbose = verbose)
        #------------------------------------------
        
        # sum all the kernel matrix
        k_eta = sum(eta_i * Ki for eta_i, Ki in zip(eta, kernelMatrix_list))
        
        #-------------------------
        # NEW CODE FOR ABSOLUTE VALUE OF CA 
        score = self.estimator.score(k_eta, self.IK_)

        if score < 0:
            score *= -1
            eta = -1*eta
        #----------------------------------
        

        return (score, k_wrap_best, eta)
            