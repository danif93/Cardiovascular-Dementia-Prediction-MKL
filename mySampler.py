import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import normalize

import myGridSearch as mgs
import Utils as ut
import time


class mySampler:
    def __init__(self, n_splits=3, test_size=.25, Ptype="classification", merging = False, sparsity=0, lamb=0, normalize_kernels=False, centering=False, normalizing=False):
        if Ptype=="classification":
            self._sampler = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        else:
            self._sampler = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        self.merging = merging
        self.sparsity = sparsity
        self.lamb = lamb
        self.normalize_kernels = normalize_kernels
        self.centering = centering
        self.normalizing = normalizing
        self.Ptype = Ptype
        
        
    def sample(self, kernelDict_list, estimator, X_list, y, valid_fold=3, verbose=False, exclusion_list = None, fileToWrite = None, header = ''):
        self.fileToWrite = fileToWrite
        self.header = header
        
        # exclusion_list in the form list of lists, one per dataset

        global_best = []

        for split_idx, (train_idx, test_idx) in enumerate(self._sampler.split(X_list[0], y)):
            if verbose: 
                print("\n{} split out of {} ...".format(split_idx+1, self._sampler.get_n_splits()))
                initTime = time.mktime(time.gmtime())
            trainSet_list = [X[train_idx] for X in X_list]
            testSet_list = [X[test_idx] for X in X_list]
            trainLabel = y[train_idx]
            testLabel = y[test_idx]
            
            
            #NEW CODE: managing labels for a correct regression (normalizing labels such that E[y^2] = 1)
            #if self.Ptype == 'regression':
                #testLabel /= np.linalg.norm(testLabel)
            
            
            # CENTERING AND NORMALIZING            
            if self.centering:
                if exclusion_list is not None:
                    scale_list = [ut.centering_normalizing(X, exc) for X, exc in zip(trainSet_list, exclusion_list)]
                else:
                    scale_list = [ut.centering_normalizing(X) for X in trainSet_list]
            
                trainSet_list = []
                new_ts = []
                for idx, (Xts, scale) in enumerate(zip(testSet_list, scale_list)):
                    new_Xts = Xts-scale[0]

                    if exclusion_list is not None and exclusion_list[idx] != []:
                        new_Xts[:, exclusion_list[idx]] = Xts[:, exclusion_list[idx]]

                    new_ts.append(new_Xts)
                    trainSet_list.append(scale[1])

                testSet_list = testSet_list
            
            if self.normalizing:
                for i in range(len(testSet_list)):
                    testSet_list[i] = normalize(testSet_list[i])
                    
                for i in range(len(trainSet_list)):
                    trainSet_list[i] = normalize(trainSet_list[i])            
            #------------------------------------------
 

            bestOverDict = []

            for d_idx, kernelDict in enumerate(kernelDict_list):
                if verbose: print("\tWorking on config {} of {}: {}".format(d_idx+1, len(kernelDict_list), kernelDict))

                gs = mgs.myGridSearchCV(estimator, kernelDict, fold=valid_fold, Ptype=self.Ptype, sparsity=self.sparsity,
                                        lamb=self.lamb, normalize_kernels=self.normalize_kernels).fit(trainSet_list, trainLabel)
                
                sel_CA, sel_kWrapp, weights = gs.transform(trainSet_list, verbose = verbose) # it was false
                
                #-------------------------------
                #NEW CODE: managing labels for a correct regression (normalizing labels such that E[y^2] = 1)
                trLabNorm = 1
                if self.Ptype == 'regression':
                    trLabNorm = np.linalg.norm(trainLabel)
                #-------------------------------
                
                pred = sel_kWrapp.predict(testSet_list, weights, trainLabel/trLabNorm, estimator, Ptype=self.Ptype)
                
                if self.Ptype == "classification":                   
                    sel_accuracy = accuracy_score(testLabel, pred)
                    precision = precision_score(testLabel, pred)
                    recall = recall_score(testLabel, pred)
                    bestOverDict.append({"CA":sel_CA, "Accuracy":sel_accuracy, "Precision":precision, "Recall":recall, "config":sel_kWrapp, "eta":weights})

                    if verbose:
                        print("\tResult of {}:".format(split_idx+1))
                        for b in bestOverDict:
                            print("CA: {}".format(b["CA"]))
                            print("Accuracy: {}".format(b["Accuracy"]))
                            print("Precision: {}".format(b["Precision"]))
                            print("Recall: {}".format(b["Recall"]))
                            print(b["config"].printConfig())
                            print("eta vector: {}\n".format(b["eta"]))
                        print("\n\tCompleted in {} minutes".format((time.mktime(time.gmtime())-initTime)/60))
                        
                else: # regression case                    
                    meanErr = np.mean(np.abs(pred*trLabNorm-testLabel))
                    varErr = np.var(np.abs(pred*trLabNorm-testLabel))
                    bestOverDict.append({"CA":sel_CA, "meanErr":meanErr, "varErr":varErr, "config":sel_kWrapp, "eta":weights})
                    
                    if verbose:
                        print("\tResult of {}:".format(split_idx+1))
                        for b in bestOverDict:
                            print("CA: {}".format(b["CA"]))
                            print("Average error: {}".format(b["meanErr"]))
                            print("Error variance: {}".format(b["varErr"]))
                            print(b["config"].printConfig())
                            print("eta vector: {}\n".format(b["eta"]))
                        print("\n\tCompleted in {} minutes".format((time.mktime(time.gmtime())-initTime)/60))
                            
            """       
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
                gs = mgs.myGridSearchCV(estimator, best_kernel_dict, fold = valid_fold, sparsity = self.sparsity, lamb = self.lamb, normalize_kernels = self.normalize_kernels).fit(trainSet_list, trainLabel)
                sel_CA, sel_kWrapp, weights = gs.transform(trainSet_list, verbose= verbose) #it was true
                pred = sel_kWrapp.predict(testSet_list, weights, trainLabel, estimator)
                sel_accuracy = accuracy_score(testLabel, pred)
                precision = precision_score(testLabel, pred)
                recall = recall_score(testLabel, pred)
                global_best.append({"CA":sel_CA, "Accuracy":sel_accuracy, "Precision":precision, "Recall":recall, "config":sel_kWrapp, "eta":weights})
                
            else:
            """
            
            global_best.append(bestOverDict)

        self.global_best_ = global_best
        return self
    
    
    def votingOverCA(self, ds_names, k_names):
                
        # INITIALISATION
        n_dict = len(self.global_best_[0])
        n_dataset = len(self.global_best_[0][0]['config'].config)
        try: 
            n_dictType = len(self.global_best_[0][0]['config'].config[0])
        except:
            n_dictType = n_dataset
            n_dataset = 1
        voting = []
        for d in range(n_dict):
            new_d = {}
            for ds_idx in range(n_dataset):
                new_d[ds_names[ds_idx]] = {}
                for dt_idx in range(n_dictType):
                    new_d[ds_names[ds_idx]][k_names[dt_idx]] = {}
            voting.append(new_d)
                
        # FILLING CONFIG FROM global_best_
        # global_best_: [#sampling:[#config_dict]]
        for sampling in self.global_best_: # running over the samples
            for dict_idx, config_dict in enumerate(sampling):
                for ds_idx, ds in enumerate(config_dict['config'].config):# running over the config_dict
                    try:
                        for dt_idx, dt in enumerate(ds):
                            try:
                                voting[dict_idx][ds_names[ds_idx]][k_names[dt_idx]][dt] += config_dict['CA']
                            except KeyError:
                                voting[dict_idx][ds_names[ds_idx]][k_names[dt_idx]][dt] = config_dict['CA']
                    except:
                        try:
                            voting[dict_idx][ds_names[0]][k_names[ds_idx]][ds] += config_dict['CA']
                        except KeyError:
                            voting[dict_idx][ds_names[0]][k_names[ds_idx]][ds] = config_dict['CA']
                                                       
        # RECOVERING CONFIG FROM voting
        winning_dict = []
        winning_list = []
        for config_dict in voting:
            new_c = {}
            new_c_l = []
            for ds_key in config_dict.keys():
                new_ds = {}
                new_ds_l = []
                for dt_key in config_dict[ds_key].keys():
                    max_v = max(config_dict[ds_key][dt_key], key=lambda key: config_dict[ds_key][dt_key][key])
                    new_ds[dt_key] = max_v
                    new_ds_l.append(max_v)
                new_c[ds_key] = new_ds
                new_c_l.append(new_ds_l)
            winning_dict.append(new_c)
            winning_list.append(new_c_l)
        
        return (winning_dict, winning_list)
    
    def performancesFeatures(self):
        
        for c_idx, config in enumerate(self.global_best_[0]):
            print("statistics of configuration {}".format(c_idx+1))
            outcome_dict = {}
            outcome_dict['config'] = {}
            for res in self.global_best_: #one res per sample
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
            
            if self.fileToWrite is not None:
                with open(self.fileToWrite, "a") as myfile:
                    myfile.write(header)
                    myfile.write("Outcome Dict: {}\n".format(outcome_dict))