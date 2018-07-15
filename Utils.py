from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelBinarizer

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