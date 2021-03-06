from collections import Counter
import numpy as np
import math as mt
import pandas as pd
#import quadprog as qp

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
#from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet


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

def oneHotEncoder_v2(df, col_to_encode):
    
    for col in col_to_encode:
        c = df[col]
        diff_labels = np.unique(c)
        for i, dl in enumerate(diff_labels):
            new_c = np.ones(len(c))
            change_idx = np.where(c != dl)
            new_c[change_idx] = -1
            
            s = pd.DataFrame({col+str(i):new_c}, index = range(len(new_c)))
            df = df.add(s, fill_value = 0)
            
    df = df.drop(col_to_encode, axis = 1)
    
    return df
    
    


def centering(X, except_col = []): #to apply only on the training set

    mean = np.mean(X, axis = 0)
    Mean = np.empty(X.shape)
    for i in range(Mean.shape[0]):
        Mean[i,:] = mean
    
    X_c = X-Mean
    
    for col in except_col: #some columns maybe should not be centered
        X_c[:, col] = X[:, col]

    return mean, X_c


def centering_rescaling(X, except_col = []): #to apply only on the training set


    mean = np.mean(X, axis = 0)
    var = np.var(X, axis = 0)
    zero_idx = np.where(var == 0)
    var[zero_idx] = 1
    Mean = np.empty(X.shape)
    for i in range(Mean.shape[0]):
        Mean[i,:] = mean

    X_c = np.divide(X-Mean, var)
    
    for col in except_col: #some columns maybe should not be centered
        X_c[:, col] = X[:, col]

    return mean, var, X_c

def centering_v2(X, except_col = []): #to apply only on the training set


    mean = np.mean(X, axis = 0)
    Mean = np.empty(X.shape)
    for i in range(Mean.shape[0]):
        Mean[i,:] = mean
    
    X_c = X-Mean
    
    for col in except_col: #some columns maybe should not be centered
        X_c[:, col] = X[:, col]

    return mean, X_c

def centering_normalizing(X, except_col = []): #to apply only on the training set


    mean = np.mean(X, axis = 0)
    Mean = np.empty(X.shape)
    for i in range(Mean.shape[0]):
        Mean[i,:] = mean
    
    X_c = X-Mean
    
    for col in except_col: #some columns maybe should not be centered
        X_c[:, col] = X[:, col]
        
    return mean, normalize(X_c)



# END PREPROCESSING

#-------------------------------------------------------------------------------------------------

# GENERAL UTIL FUNCTIONS

def balanced_accuracy_score(y_test, y_pred):
    
    r1 = recall_score(y_test, y_pred)
    r2 = recall_score(-1*y_test, -1*y_pred)
    
    return (r1+r2)/2


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


def testConfigurations(estimator, y_train, y_test, config_list, train_list, test_list, kernel_types, lamb_list, sparsity, Ptype='classification', lock=None, fileToWrite=None, header='', normalize = False, verbose=False):
    
    # FIND THE BEST CONFIGURATIONS METRICS
    if Ptype == 'regression':
        n = np.linalg.norm(y_train)
        y_train /= n
        
    IK_tr = np.outer(y_train, y_train)
    IK_test = np.outer(y_test, y_train)

    for cl_idx, cl in enumerate(config_list):
        if len(cl) == 1:
            cl = cl[0]
        found_kWrap = kf.kernelWrapper(train_list, kernel_types, cl, normalize = normalize)
        # compute the list of kernels generated from the hyperparameter configuration at hand
        kernelMatrix_list = found_kWrap.kernelMatrix(train_list).kernelMatrix_list_

        # compute eta vector
        #---------------------------------------------
        #NEW CODE to modify for OLS
        lamb = lamb_list[cl_idx]
        if sparsity:
            eta = estimator.computeEta(kernelMatrix_list, IK_tr, y = y_train, sparsity = lamb, verbose = verbose)
        else:
            eta = estimator.computeEta(kernelMatrix_list, IK_tr, y = y_train, lamb = lamb, verbose = verbose)
        
        #----------------------------------------------
        #NEW CODE, FIXING THE POSSIBILITY TO HAVE NEGATIVE CA
        k_eta = np.zeros(kernelMatrix_list[0].shape)
        for eta_i, Ki in zip(eta, kernelMatrix_list):
            k_eta += eta_i * Ki

        score = estimator.score(k_eta, IK_test)
        if score < 0:
            score *= -1
            eta = -1*eta
        #---------------------------------------------

        # compute k_eta (approximation) for the validation set
        #kernelMatrix_list = found_kWrap.kernelMatrix(ds_test).kernelMatrix_list_
        #k_eta = np.zeros(kernelMatrix_list[0].shape)
        #for eta_i, Ki in zip(eta, kernelMatrix_list):
        #    k_eta += eta_i * Ki
        #performances = estimator.score(k_eta, IK_test)


        # compute performances estimation
        pred = found_kWrap.predict(test_list, eta, y_train, estimator, Ptype=Ptype)
        
        if Ptype=='classification':        
            #accuracy = accuracy_score(y_test, pred)
            accuracy = balanced_accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)
            
            if verbose:
                print("Perfomances computed for dictionary settings {}:".format(cl_idx+1))
                print("\tAccuracy: {}".format(accuracy))
                print("\tPrecision: {}".format(precision))
                print("\tRecall: {}".format(recall))
            
            if fileToWrite is not None:
                with lock:
                    with open(fileToWrite, "a") as myfile:
                        myfile.write(header)
                        myfile.write("Accuracy: {}\n".format(accuracy))
                        myfile.write("Precision: {}\n".format(precision))
                        myfile.write("Recall: {}\n".format(recall))
                        myfile.write("Lambda: {}\n".format(lamb))
                        myfile.write("Config: {}\n".format(cl))
                        myfile.write("Eta: {}\n\n".format(eta))
                    
            
        else:          
            meanErr = np.mean(np.abs(pred*n-y_test))
            varErr = np.var(np.abs(pred*n-y_test))
            
            if verbose:
                print("Perfomances computed for dictionary settings {}:".format(cl_idx+1))
                print("\tAverage error: {}".format(meanErr))
                print("\tError variance: {}".format(varErr))
                print("\tPred: {}".format(pred))
                print("\tPred_multiplied: {}".format(pred*n))
                
            if fileToWrite is not None:
                with lock:
                    with open(fileToWrite, "a") as myfile:
                        myfile.write(header)
                        myfile.write("Average error: {}\n".format(meanErr))
                        myfile.write("Error variance: {}\n".format(varErr))
                        myfile.write("Lambda: {}\n\n".format(lamb))


# END GENERAL UTIL FUNCTIONS

#-------------------------------------------------------------------------------------------------
