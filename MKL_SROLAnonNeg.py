# SIMPLE SROLA

import numpy as np
import KernelFile as kf

class MKL_simpleSrola:
    
    def __init__(self, Ktype_dict, C, tol = 0.01):
        
        #K_type_dict = dictionary [kernel_name] : param_value
        
        self.Ktype_dict = Ktype_dict
        self.C = C
        self.tol = tol
        
        
    def fit(self, Xtr_list, y):
        
        self._Xtr_list = Xtr_list
        self._y = y
        self._k_list = []
        for X in Xtr_list:
            # k_list_dataset = [kernel(X, Ktype, v) for v in value for Ktype, value in self.Ktype_dict.items()]                
            # self._k_list.append([kernel(X, Ktype, v) for v in value for Ktype, value in self.Ktype_dict.items()])

            k_list_dataset = []
            for Ktype, value in self.Ktype_dict.items():
                k_list_dataset.append(kf.kernel(X, Ktype, value))
                
            self._k_list.append(k_list_dataset)

        numK = len(Xtr_list)*len(self.Ktype_dict.keys())
        self.eta_ = np.ones(numK) / numK
            
        return self
    
    def _getKernels(self, X_list):
        
        K_list = []
        for dataset_index, X in enumerate(X_list):
            for k in self._k_list[dataset_index]:
                K_list.append(k.kernelMatrix(X))
                
        return K_list
            
    def learnEta(self, X_list):
        
        K_list = self._getKernels(X_list)
                
        self._learn(K_list) #TODO add other params if needed
        
        return self.eta_
    
    def _learn(self, K_list):
        
        while True:
            
            
            #TODO compute quadprog_solve_qp imputs
            
            p = np.empty((len(self._y)),len(self._y))
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    p[i,j] = self._y[i]*self._y[j]*sum([self.eta_[k_idx]*kern[i,j] for k_idx, kern in K_list])
            
            alpha_list = quadprog_solve_qp(p,) # TODO

            
            partial_derivatives = self._computeDerivatives(K_list, alpha_list)
            ref_eta_index = np.argmax(self.eta_)
            partial_derivatives = np.asarray(self._computeMagnitude(partial_derivatives, ref_eta_index))
            stepSize = self._chooseStepSize(partial_derivatives, alpha_list, K_list)
            self.eta_ += stepSize * partial_derivatives
            
            if self._converged(K_list, alpha_list):
                break
        
        
    def _computeDerivatives(self, K_list, alpha_list):
        
        partial_derivatives = []
        for K in K_list:
            grad = -0.5 * sum([alpha_i * alpha_j * K[i,j] for i, alpha_i in enumerate(alpha_list) for j, alpha_j in enumerate(alpha_list)])
            partial_derivatives.append(grad)
            
        return partial_derivatives
    
    
    def _computeMagnitude(self, partial_derivatives, ref_eta_index):
        
        magnitudes = []
        ref_eta = partial_derivatives[ref_eta_index]
        
        for index, d in enumerate(partial_derivatives):
            
            if d == 0 or d - ref_eta > 0 :
                magnitudes.append(0)
                
            elif d > 0 and index != ref_eta_index:
                magnitudes.append(ref_eta - d)
                
            elif index == ref_eta_index:
                val = 0
                for pd in partial_derivatives:
                    if pd > 0:
                        val += pd - ref_eta
                
            else:
                raise Excpetion("Unknown error, derivatives: {}, ref_index: {}".format(partial_derivatives, ref_eta_index))
                

    def _chooseStepSize(self, partial_derivatives, alpha_list, K_list):
        
        t_range = np.arange(0, 1, 0.1)
        
        f_values_list = []
        for t in t_range:
            f_values_list.append(self._f(self.eta_ + t * partial_derivatives, alpha_list, K_list))
            
        return t_range[np.argmax(f_values_list)]
            
            
    def _f(new_eta, alpha_list, K_list):
        
        val = 0
        for i, alpha_i in enumerate(alpha_list):
            for j, alpha_j in enumerate(alpha_list):
                for eta, K in zip(new_eta, K_list):
                    val += alpha_i * self._y[i] * alpha_j * self._y[j] * eta * K[i, j]

        val *= -0.5
        
        val += sum(alpha_list)
            
        return val
    
    
    def _coverged(self, K_list, alpha_list):
        
        dx = self._f(self.eta_, alpha_list, K_list) - sum(alpha_list)
        
        sx_list = []
        for K in K_list:
            sx_list.append(sum([alpha_i*alpha_j*self._y[i]*self._y[j]*K[i,j] for i, alpha_i in enumerate(alpha_list) for j, alpha_j in enumerate(alpha_list)]))
            
        sx = max(sx_list)            
                    
        
        if sx - dx <= self.tol:
            return True
        
        return False

        
    def predict(self, X_list, y):
        
        K_list = self._getKernels(X_list)
        
        y_pred =  0 #TODO prediction of y given eta and Kernel list -> y_pred
        
        score = self._score(y_pred, y)
        
        return y_pred, score
    
    #def _score(y_pred, y):
        
        # TODO score function
