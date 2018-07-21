# SOLA AKW pg 22-23 (Similarity Optimizing Linear Approach with Arbitrary Kernel Weights)
# Interval Cortes approach

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
