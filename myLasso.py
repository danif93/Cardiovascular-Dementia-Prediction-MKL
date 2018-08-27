import numpy as np
import itertools
import operator 
import time
#from sklearn.base import BaseEstimator, RegressorMixin
#from copy import deepcopy
#from Utils import frobeniusInnerProduct


class Lasso:
    
    def __init__(self, alpha=1.0, max_iter=500, tol = 0.01, estimator = False, verbose = False):
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.estimator = estimator
        self.verbose = verbose
        
       
    def fit(self, X, y, label, K_list):
        
        if self.max_iter ==  0: #analytical solution
            #start = time.time()
            
            self.eta_length = X.shape[1]
            smart_config_list = []
            
            for i in range(self.eta_length):
                smart_config_list.append([0, 1,-1])
            
            smart_sign_list = []
            
            smart_sign_list += [list(elem) for elem in itertools.product(*smart_config_list)]
            
            del_idx_list = []
            
            for del_idx, config in enumerate(smart_sign_list):
                num_zeros = len(np.where(np.asarray(config) == 0)[0])
                if num_zeros != 5:# PREVIOUSLY num_zeros < 4 or num_zeros > 6:
                    del_idx_list.append(del_idx)
            
            for idx in sorted(del_idx_list, reverse = True):
                del smart_sign_list[idx]
            
            
            score_eta_list = []
            for idx, smart_sign in enumerate(smart_sign_list):
                
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
                    score_eta_list.append((self.estimator.externalScore(K_eta, np.outer(label,label)), real_eta))

            if score_eta_list == []:
                self.coef_ = self.estimator.coef(X, y) #return the not constrained problem's solution
                if self.verbose: print("Smart approach failed. No sparsity applied")

            else:
                
                #ca, self.coef_ = max(score_eta_list, key=operator.itemgetter(0))
                self.coef_ = max(score_eta_list, key=operator.itemgetter(0))[1]
                
            #print("\tone lasso analysis completed. Time spent %s" % (time.time()-start))
            #print("config chosen: {}".format(self.coef_))
            #print("CA: {}".format(ca))
            
            return self

        else: #TODO if needed
            raise Eception("Error you must not be here")
            for i in range(self.max_iter):
                self.estimator.fit()
           