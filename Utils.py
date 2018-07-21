from collections import Counter
import numpy as np
import math as mt
import quadprog as qp
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet

from multiprocessing import Process
from queue import Queue

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.grid_search import GridSearchCV

import KernelFile as kf

#----------------------------------------
# PREPROCESSING

def value_strategy(v, strategy):
    if strategy == 'min':
        return np.min(v)
    if strategy == 'mean':
        return np.mean(v)
    elif strategy == 'median':
        return np.median(v)
    elif strategy == 'most_frequent':
        return Counter(v).most_common(1)[0][0]


def imputing(data, strategy, axis, exception = [], floor = False):

    for i, e in enumerate(exception):
        exception[i] = np.where(data.columns == e)[0][0]

    for j in range(data.shape[axis]):

        if j in exception:
            continue

        indices = np.where(np.isnan(data.iloc[:,j]))[0] #tuple row, col

        if len(indices) == 0:
            #print("skipped {}".format(j))
            continue

        available = data.iloc[~indices, j]
        value = value_strategy(available, strategy)

        if floor:
            value = np.floor(value)

        data.iloc[indices, j] = value;
    return data


def oneHotEncoder(v):

    ohe = LabelBinarizer()
    enc = ohe.fit_transform(v)
    binary = []
    for r in enc:

        row = ''

        for c in r:
            row += str(c)

        binary.append(int(row))

    return np.asarray(binary)

# END PREPROCESSING

#-------------------------------------------

# GENERAL UTIL FUNCTIONS

# 0.5 x.T P x + q.T x
# s.t. G x <= h
#      Ax = b

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A != None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return qp.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def frobeniusInnerProduct(A, B):
    A = np.matrix.conjugate(A).ravel()
    B = B.ravel()
    return np.dot(A, B)


def normalization(X, norm = 'l2'):
    return normalize(X, norm = norm)


def getParamInterval(kernel):
    param = kernel.getParam()
    g_step = param/10
    k_type = kernel.getType()
    return np.arange(np.max([param-(g_step*8), g_step]), param+(g_step*8), g_step) if k_type=='gaussian' else np.arange(np.max([param-8,1]), param+8, 2)


def getKernelList(wrapper, kind='train_ds'):
    k_train_list = []
    for kernel_wrapp in wrapper:
        k_train_list.append(kernel_wrapp['kernel'].kernelMatrix(kernel_wrapp[kind]))
    return k_train_list


def kernelMatrixSum(kernel_list, weights, size):
    k_sumMat = np.zeros([size, size])
    # sum of all kernel matrix
    for kernel, w in zip(kernel_list, weights):
        k_sumMat += kernel*w
    return k_sumMat

# END GENERAL UTIL FUNCTIONS

#-------------------------------------------------------------------------------------------------

# USEFUL CLASSSES

class centeredKernelAlignment:

    def _centeredKernel(K): # K^c
        Nt = K.shape[0]
        Ns = K.shape[1]
        oneNt = np.ones((Nt)).reshape(-1,1)
        oneNs = np.ones((Ns)).reshape(-1,1)
        oneNt_mat = np.outer(oneNt, oneNt)
        oneNs_mat = np.outer(oneNs, oneNs)
        composite = np.outer(oneNt, oneNs)
        
        add1 = 1/Nt * np.dot(oneNt_mat, K)
        add2 = 1/Nt * np.dot(K, oneNs_mat)
        add3 = 1/(Nt*Nt) * np.dot(np.dot(oneNt.T, K), oneNs) * composite
        
        return K - add1 - add2 + add3


    def _kernelSimilarityMatrix(K_list): # M

        M = np.zeros((len(K_list), len(K_list)))

        for i, K1 in enumerate(K_list):
            for j, K2 in enumerate(K_list[i:]):

                s = frobeniusInnerProduct(K1, K2)
                M[i, i+j] = s

                if j != 0:
                    M[i+j, i] = s
        return M


    def _idealSimilarityVector(K_list, IK): # a

        a = np.zeros((len(K_list)))

        for i, K in enumerate(K_list):
            a[i] = frobeniusInnerProduct(K, IK)

        return a


    def computeEta(K_list, IK):

        K_c_list = [centeredKernelAlignment._centeredKernel(K) for K in K_list]

        M = centeredKernelAlignment._kernelSimilarityMatrix(K_c_list)

        a = centeredKernelAlignment._idealSimilarityVector(K_c_list, IK)

        num = np.dot(np.linalg.inv(M), a)

        return num / np.linalg.norm(num)


    def score(k1, k2):
        k1c = centeredKernelAlignment._centeredKernel(k1)
        k2c = centeredKernelAlignment._centeredKernel(k2)

        num = frobeniusInnerProduct(k1c, k2c)
        den = np.sqrt(frobeniusInnerProduct(k1c, k1c)*frobeniusInnerProduct(k2c, k2c))
        return num/den
    

# USEFUL CLASSSES

#-------------------------------------------

# SOLA AKW pg 22-23 (Similarity Optimizing Linear Approach with Arbitrary Kernel Weights)
# Cortes approach    
    
def parameterOptimization(k_dataset_wrapper, train_label, n_epoch=100, tol=0.01, verbose=False):
    
    idealKernel = train_label.reshape(-1,1).dot(train_label.reshape(-1,1).T)
    
    previousCA = -1
    
    k_train_list = getKernelList(k_dataset_wrapper, kind='train_ds')
    
    cka = centeredKernelAlignment
    
    weights = cka.computeEta(k_train_list, idealKernel)

    for i in range(n_epoch):
        k_train_list = getKernelList(k_dataset_wrapper, kind='train_ds')
        k_sumTrain = kernelMatrixSum(k_train_list, weights, len(train_label))
        currentCA = cka.cortesAlignment(k_sumTrain, idealKernel)
        if verbose: print('Epoch num {}; current CA is: {}'.format(i+1, currentCA))

        if previousCA>0 and currentCA>0 and np.abs(previousCA-currentCA)<tol: break
        else: previousCA = currentCA

        for kernel_idx in np.argsort(weights): # start optimizing the most impactful kernel parameter
            if verbose: print('\toptimizing {}'.format(kernel_idx))
            kernel = k_dataset_wrapper[kernel_idx]['kernel']

            if (kernel.getType()=='linear'): continue

            param_interval = getParamInterval(kernel)
            if verbose: print('\t\toptimizing over [{},{}]'.format(param_interval[0], param_interval[-1]))

            similarity_grid = np.zeros(len(param_interval))
            old_param = k_dataset_wrapper[kernel_idx]['kernel'].getParam()
            
            old_weightsConfig = np.zeros((len(param_interval)+1, len(k_dataset_wrapper)))
            old_weightsConfig[-1] = weights
             
            for p_idx, param in enumerate(param_interval):
                k_dataset_wrapper[kernel_idx]['kernel'].setParam(param)
                
                # update the weights with the new configuration
                k_train_list = getKernelList(k_dataset_wrapper, kind='train_ds')
                weights = cka.computeEta(k_train_list, idealKernel)
                
                old_weightsConfig[p_idx] = weights

                k_sumTrain = kernelMatrixSum(k_train_list, weights, len(train_label))

                similarity_grid[p_idx] = cka.cortesAlignment(k_sumTrain, idealKernel)

            selected = np.argmax(similarity_grid)

            if similarity_grid[selected] > currentCA:
                currentCA = similarity_grid[selected]
                k_dataset_wrapper[kernel_idx]['kernel'].setParam(param_interval[selected])
                weights = old_weightsConfig[selected]
                
                if verbose: print('\t\tselected {} with sim: {}'.format(param_interval[selected], currentCA))

            else:
                k_dataset_wrapper[kernel_idx]['kernel'].setParam(old_param)
                weights = old_weightsConfig[-1]
                if verbose: print('\t\tkept {} with sim: {}'.format(kernel.getParam(), currentCA))

    return k_dataset_wrapper

# END SOLA AKW

#-------------------------------------------------------

# SIMPLE SROLA

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
