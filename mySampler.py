import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import myGridSearch as mgs


class mySampler:
    def __init__(self, n_splits=3, test_size=.25):
        self._sampler = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        
    def sample(self, kernelDict_list, estimator, X_list, y, valid_fold = 3, verbose=False):
#kernelDict_list: [{'linear':[0],'polynomial':[2,5],'gaussian':[.1,.5]},{'linear':[0],'polynomial':[7,10],'gaussian':[.7,1]}, ...]

        global_best = []

        for split_idx, (train_idx, test_idx) in enumerate(self._sampler.split(X_list[0], y)):
            if verbose: print("{} split out of {} ...".format(split_idx+1, self._sampler.get_n_splits()))
            trainSet_list = [X[train_idx] for X in X_list]
            testSet_list = [X[test_idx] for X in X_list]
            trainLabel = y[train_idx]
            testLabel = y[test_idx]

            bestOverDict = []

            for d_idx, kernelDict in enumerate(kernelDict_list):
                if verbose: print("\tWorking on config {} of {}: {}".format(d_idx+1, len(kernelDict_list), kernelDict))

                gs = mgs.myGridSearchCV(estimator, kernelDict, fold = valid_fold).fit(trainSet_list, trainLabel)
                sel_CA, sel_kWrapp, weights = gs.transform(trainSet_list, verbose=False)
                sel_accuracy = accuracy_score(testLabel, sel_kWrapp.predict(testSet_list, weights, trainLabel))

                bestOverDict.append({"CA":sel_CA, "Accuracy":sel_accuracy, "config":sel_kWrapp})

            if verbose:
                print("\tResult of {}:".format(split_idx))
                for b in bestOverDict:
                    print(b["config"].printConfig())

            if verbose: print("\tMearging config of split {} ...".format(split_idx+1))

            best_kernel_dict = {}
            for kernel_dict_index, elem in enumerate(bestOverDict):
                kWrap = elem["config"]
                for K_type, param in kWrap.getConfig().items():
                    try:
                        best_kernel_dict[K_type].append(param)
                    except:
                        best_kernel_dict[K_type] = []
                        best_kernel_dict[K_type].append(param)

            for key in best_kernel_dict.keys():
                best_kernel_dict[key] = np.unique(best_kernel_dict[key])

            if verbose:
                print("\tMearging config of split {} completed. New kernel dict:".format(split_idx+1))
                for k, v in best_kernel_dict.items():
                    print("\t\t{} : {}".format(k,v))
            
            print("\tComputing performances using the merged dictionary")
            gs = mgs.myGridSearchCV(estimator, best_kernel_dict, fold = valid_fold).fit(trainSet_list, trainLabel)
            sel_CA, sel_kWrapp, weights = gs.transform(trainSet_list, verbose=True)
            sel_accuracy = accuracy_score(testLabel, sel_kWrapp.predict(testSet_list, weights, trainLabel))
            global_best.append({"CA":sel_CA, "Accuracy":sel_accuracy, "config":sel_kWrapp})

        return global_best
