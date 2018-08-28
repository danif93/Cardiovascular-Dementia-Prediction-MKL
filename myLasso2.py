from multiprocessing import Process
import numpy as np
import itertools
import operator
import time

#from sklearn.base import BaseEstimator, RegressorMixin
#from copy import deepcopy
#from Utils import frobeniusInnerProduct


class Lasso:
    
    def __init__(self, alpha=1.0, max_iter=500, tol = 0.01, estimator = False, verbose = False, proc = 10):
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.estimator = estimator
        self.verbose = verbose
        self.proc = proc
    
    def _fit_child(self, dict_, proc_idx, smart_sign, X, y, K_list, label):
        
        smart_sign = np.asarray(smart_sign)
        zero_idx = np.where(smart_sign == 0)[0]
        C = np.multiply(np.ones(self.eta_length), smart_sign)*self.alpha
        X_tmp = X
        X_tmp = np.delete(X, zero_idx, 0)
        X_tmp = np.delete(X_tmp, zero_idx, 1)
        y_tmp = y + C
        y_tmp = np.delete(y_tmp, zero_idx)

        eta = self.estimator.coef(X_tmp, y_tmp.T)
        smart_sign = np.delete(smart_sign, zero_idx)

        if np.allclose(np.sign(eta), smart_sign):
            real_eta = np.zeros(len(C))
            eta_idx = 0
            for idx, entry in enumerate(C):
                if entry == 0:
                    real_eta[idx] = 0
                else:
                    real_eta[idx] = eta[eta_idx]
                    eta_idx += 1


            K_eta = sum(eta*K for eta, K in zip(real_eta, K_list))
            dict_[proc_idx] = ((self.estimator.externalScore(K_eta, np.outer(label,label)), real_eta))

    def fit(self, X, y, label, K_list):
        
        if self.max_iter ==  0: #analytical solution
            self.eta_length = X.shape[1]
            smart_config_list = []
            
            for i in range(self.eta_length):
                smart_config_list.append([0, 1,-1])
            
            smart_sign_list = []
            
            smart_sign_list += [list(elem) for elem in itertools.product(*smart_config_list)]
            
            del_idx_list = []
            
            for del_idx, config in enumerate(smart_sign_list):
                num_zeros = len(np.where(np.asarray(config) == 0)[0])
                if num_zeros != 4:
                    del_idx_list.append(del_idx)
            
            for idx in sorted(del_idx_list, reverse = True):
                del smart_sign_list[idx]
            
            score_eta_list = []
            if len(smart_sign_list) < self.proc:
                self.proc = len(smart_sign_list)
                
            proc_idx = 0
            jobs = []
            best = None
            dict_ = {} # "proc_idx": (CA, eta)
            for idx, smart_sign in enumerate(smart_sign_list):
                
                proc_idx += 1
                jobs.append(Process(target = self._fit_child, args = (dict_, proc_idx, smart_sign, X, y, K_list, label)))
                if proc_idx == self.proc:
                    print("waiting for the {} jobs".format(self.proc))
                    start = time.time()
                    for p in jobs:
                        p.start()
                    
                    for p in jobs:
                        p.join()
                    print("\t.... complete. Time spent %s" % (time.time()-start))
                        
                    proc_idx = 0
                    jobs = []
                    if len(dict_) > 0:
                        print(dict_)
                    maxim = (0, 0) if best is None else best
                    for v in dict_.values():
                        if v[0] > maxim[0]:
                            maxim = v
                            
                    best = None if maxim == (0,0) else maxim
                    print("best: {}".format(best))
                        
            if len(jobs) != 0:
                for p in jobs:
                    p.start()
                for p in jobs:
                    p.join()

            if best is None:
                self.coef_ = self.estimator.coef(X, y) #return the not constrained problem's solution
                if self.verbose: print("Smart approach failed. No sparsity applied")

            else:
                
                self.coef_ = best[1]
                
            return self

        else: #TODO if needed
            raise Eception("Error you must not be here")
            for i in range(self.max_iter):
                self.estimator.fit()