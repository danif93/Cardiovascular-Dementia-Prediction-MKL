import numpy as np
import itertools
#from sklearn.base import BaseEstimator, RegressorMixin
#from copy import deepcopy
#from Utils import frobeniusInnerProduct


class Lasso:
    
    def __init__(self, alpha=1.0, max_iter=500, tol = 0.01, estimator = False):
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.estimator = estimator
        
       
    def fit(self, X, y):
        
        if self.max_iter ==  0: #analytical solution
            self.eta_length = X.shape[1]
            tmp_config_list = []
            """
            for i in range(self.eta_length):
                tmp_config_list.append([0, 1,-1])
            sign_matrix = np.asmatrix([list(elem) for elem in itertools.product(*tmp_config_list)])
            """
            smart_sign_list = []
            smart_config_list = [
                                 [[1,-1], [1,-1], [1,-1], [1,-1], [1,-1], [1,-1], [0], [0], [0]], 
                                 [[1,-1], [1,-1], [1,-1], [0], [0], [0], [1,-1], [1,-1], [1,-1]],
                                 [[0], [0], [0], [1,-1], [1,-1], [1,-1], [1,-1], [1,-1], [1,-1]],
                                 [[0], [1,-1], [1,-1], [0], [1,-1], [1,-1], [0], [1,-1], [1,-1]],
                                 [[1,-1], [0], [1,-1], [1,-1], [0], [1,-1], [1,-1], [0], [1,-1]],
                                 [[1,-1], [1,-1], [0], [1,-1], [1,-1], [0], [1,-1], [1,-1], [0]]
                                ] #TODO TO COMPLETE
            for smart_config in smart_sign_list:
                smart_sign_list += [list(elem) for elem in itertools.product(*smart_config)]
            
            for smart_sign in smart_sign_list:
                smart_sign = np.asarray(sorted(smart_sign, reverse = True))
                zero_idx = np.where(smart_sign_list == 0)
                C = np.multiply(np.ones(self.eta_length), sign_matrix[i,:])*alpha
                X_tmp = X
                X_tmp = np.delete(X, zero_idx, 0)
                X_tmp = np.delete(X, zero_idx, 1)
                y_tmp = y + C
                y_tmp = np.delete(y_tmp, zero_idx, 1)
                eta = estimator.coef(X_tmp, y_tmp)
                smart_sign = np.delete(smart_sign, zero_idx)
                if np.allclose(np.sign(eta), sign_matrix[i,:]):
                    real_eta = np.zeros(len(C))
                    eta_idx = 0
                    for idx, entry in enumerate(C):
                        if entry == 0:
                            real_eta[idx] = 0
                        else:
                            real_eta[idx] = eta[eta_idx]
                            eta_idx += 1
                            
                    return real_eta
            """
            for i in range(sign_matrix.shape[0]):
                if list(sign_matrix[i,:]) in smart_sign_list:
                    continue
                    
                X_tmp = X
                C = np.multiply(np.ones(self.max_iter), sign_matrix[i,:])*alpha
                eta = estimator.coef(X, y + C)
                if np.sign(eta) == sign_matrix[i,:]:
                    return eta
            """  
            
            return eta.coef(X, y) #return the not constrained problem's solution

        else: #TODO if needed
            for i in range(self.max_iter):
                estimator.fit()
            
      
"""

class Lasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, max_iter=500, tol = 0.01, fit_intercept=False, verbose = False): #for the moment we suppose the data are centered, otherwise not works
        self.alpha = alpha # 正則化項の係数
        self.max_iter = max_iter # 繰り返しの回数
        self.fit_intercept = fit_intercept # 切片(i.e., \beta_0)を用いるか
        self.coef_ = None # 回帰係数(i.e., \beta)保存用変数
        self.intercept_ = None # 切片保存用変数
        self.tol = tol
        self.verbose = verbose

    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0

    def fit(self, X_list, IK):
        if self.fit_intercept: #not works yet
            X = np.column_stack((np.ones(len(X)),X))

        beta = np.zeros(len(X_list))
        if self.fit_intercept: #not works yet
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:]))/(X.shape[0])

        for iteration in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = deepcopy(beta)
                tmp_beta[j] = 0.0
                X_eta = sum(beta_i * X_i for beta_i, X_i in zip(tmp_beta, X_list))
                r_j = IK - X_eta#y - np.dot(X, tmp_beta)
                
                if self.verbose: print("r_j norm: {}".format(np.linalg.norm(r_j)))
                if np.linalg.norm(r_j) < self.tol:
                    beta[j] = 0.0
                    break
                
                arg1 = frobeniusInnerProduct(X_list[j], r_j) #np.dot(X[:, j], r_j)
                arg2 = self.alpha*len(X_list) #X.shape[0]

                beta[j] = self._soft_thresholding_operator(arg1, arg2)/frobeniusInnerProduct(X_list[j], X_list[j])

                if self.fit_intercept:#not works yet
                    beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:]))/len(X_list) #(X.shape[0])

        if self.fit_intercept:#not works yet
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self

    def predict(self, X_list):
        y = sum(coef_i * X_i for coef_i, X_i in zip(self.coef_, X_list))
        if self.fit_intercept:#not works yet
            y += self.intercept_*np.ones(len(y))
        return y
        
"""