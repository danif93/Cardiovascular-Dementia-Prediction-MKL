from collections import Counter
import numpy as np
import math as mt
import pandas as pd
#import quadprog as qp
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet


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


def centering(df, except_col): #to apply only on the training set

    X = df.values

    mean = np.mean(X, axis = 0)
    Mean = np.empty(X.shape)
    for row in Mean:
        Mean[i,:] = mean

    X_c = pd.DataFrame(X-Mean, index = df.index, columns = df.columns)

    for col in except_col: #some columns maybe should not be centered
        X_c[col] = df[col]

    return mean, X_c,

# END PREPROCESSING

#-------------------------------------------------------------------------------------------------

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


# END GENERAL UTIL FUNCTIONS

#-------------------------------------------------------------------------------------------------
