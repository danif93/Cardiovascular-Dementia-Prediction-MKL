import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

import myGridSearch as mgs


class mySampler:
    def __init__(self, n_splits=3, test_size=.25, merging = False, sparsity = 0, normalize_kernels = False):
        self._sampler = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        self.merging = merging
        self.sparsity = sparsity
        self.normalize_kernels = normalize_kernels
        
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

                gs = mgs.myGridSearchCV(estimator, kernelDict, fold = valid_fold, sparsity = self.sparsity, normalize_kernels = self.normalize_kernels).fit(trainSet_list, trainLabel)
                sel_CA, sel_kWrapp, weights = gs.transform(trainSet_list, verbose = verbose) # it was false
                pred = sel_kWrapp.predict(testSet_list, weights, trainLabel)
                sel_accuracy = accuracy_score(testLabel, pred)
                precision = precision_score(testLabel, pred)
                recall = recall_score(testLabel, pred)
                #sel_accuracy = sel_kWrapp.accuracy(testLabel, test_pred = pred)
                #precision = sel_kWrapp.precision(testLabel, test_pred = pred)
                #recall = sel_kWrapp.recall(testLabel, test_pred = pred)

                bestOverDict.append({"CA":sel_CA, "Accuracy":sel_accuracy, "Precision":precision, "Recall":recall, "config":sel_kWrapp, "eta":weights})

            if verbose:
                print("\tResult of {}:".format(split_idx))
                for b in bestOverDict:
                    print("CA: {}".format(b["CA"]))
                    print(b["config"].printConfig())
                    print("eta vector: {}".format(b["eta"]))
                    
            if self.merging:

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
                gs = mgs.myGridSearchCV(estimator, best_kernel_dict, fold = valid_fold, sparsity = self.sparsity, normalize_kernels = self.normalize_kernels).fit(trainSet_list, trainLabel)
                sel_CA, sel_kWrapp, weights = gs.transform(trainSet_list, verbose= verbose) #it was true
                pred = sel_kWrapp.predict(testSet_list, weights, trainLabel)
                sel_accuracy = accuracy_score(testLabel, pred)
                precision = precision_score(testLabel, pred)
                recall = recall_score(testLabel, pred)
                global_best.append({"CA":sel_CA, "Accuracy":sel_accuracy, "Precision":precision, "Recall":recall, "config":sel_kWrapp, "eta":weights})
                
            else:
                global_best.append(bestOverDict)

        self.global_best_ = global_best
        return self
    
    
    
    def performancesFeatures(self):
        
        for c_idx, config in enumerate(self.global_best_[0]):
            print("statistics of configuration {}".format(c_idx))
            outcome_dict = {}
            outcome_dict['config'] = {}
            for res in self.global_best_: #one res per fold
                res = res[c_idx] #correct dictionary
                for key, value in res.items():
                    if key != 'config':
                        try:
                            outcome_dict[key].append(value)
                        except:
                            outcome_dict[key] = []
                            outcome_dict[key].append(value)
                    else:
                        config_dict = value.getConfig()
                        for k, v in config_dict.items():
                            try:
                                outcome_dict[key][k].append(v)
                            except:
                                outcome_dict[key][k] = []
                                outcome_dict[key][k].append(v)
                        
            for key, value in outcome_dict.items():
                if key != 'config':
                    if key != "eta":
                        outcome_dict[key] = (np.mean(value), np.var(value))
                    else:
                        outcome_dict[key] = (np.mean(value, axis = 0), np.var(value, axis = 0))
                    
                            
                    
                            
            print(outcome_dict)