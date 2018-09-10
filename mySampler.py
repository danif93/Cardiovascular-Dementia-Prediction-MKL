import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import normalize

import myGridSearch as mgs
import Utils as ut
import time

class mySampleWrapper:
    
    def __init__(self, lamb_list, n_splits=3, test_size=.25, Ptype="classification", merging = False, sparsity=False, normalize_kernels=False, centering=False, normalizing=False):

        self.samplers = []
        self.lamb_list = lamb_list
        self.sparsity = sparsity
        self.normalize_kernels = normalize_kernels

        
        for lamb in lamb_list:
            
            if sparsity == False:
                sampler = mySampler(n_splits, test_size, Ptype, merging, lamb = lamb, normalize_kernels=normalize_kernels, centering=centering, normalizing=normalizing)
                
                self.samplers.append(sampler)
                
            else:
                sampler = mySampler(n_splits, test_size, Ptype, merging, sparsity=lamb, normalize_kernels=normalize_kernels, centering=centering, normalizing=normalizing)
                
                self.samplers.append(sampler)
                
                
            
    def sample(self, kernelDict_list, estimator, X_list, y, valid_fold=3, verbose=False, exclusion_list = None):
        
        for sampler in self.samplers:
            sampler.sample(kernelDict_list, estimator, X_list, y, valid_fold, verbose, exclusion_list)
            
        return self
    
    
    def votingOverCA(self, ds_names, k_names):
        
        self.winning_sampler_list = [] 
        self.winning_dict_list = []
        self.winning_list_list = []
        self.winning_lamb_list = []
        
        for sampler in self.samplers:
            sampler.votingOverCA(ds_names, k_names)
            sampler.performancesFeatures()
            
        self.num_config = len(self.samplers[0].outcome_dict_list)
        
        for config_idx in range(self.num_config):
            ca_list = []
            for sampler in self.samplers:
                ca_list.append(sampler.outcome_dict_list[config_idx]['CA'][0])
            
            winner = np.argmax(ca_list)
            self.winning_sampler_list.append(winner)
            self.winning_dict_list.append(self.samplers[winner].winning_dict[config_idx])
            self.winning_list_list.append(self.samplers[winner].winning_list[config_idx])
            self.winning_lamb_list.append(self.lamb_list[winner])
        
        return self.winning_dict_list, self.winning_list_list, self.winning_lamb_list, self.sparsity
    
    
    def performancesFeatures(self, fileToWrite = None, header = '', lock = None):
        
        self.outcome_dict_list = []
        
        for config_idx in range(self.num_config):
            outcome_dict = self.samplers[self.winning_sampler_list[config_idx]].outcome_dict_list[config_idx]
            outcome_dict['lambda'] = self.winning_lamb_list[config_idx]
            self.outcome_dict_list.append(outcome_dict)
            
        
        if fileToWrite is not None and lock is not None:
            with lock:
                with open(fileToWrite, "a") as myfile:
                    myfile.write(header)
                    for outcome_dict in self.outcome_dict_list:
                        myfile.write("Outcome Dict: {}\n\n".format(outcome_dict))

        return self


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
        
    def sample(self, kernelDict_list, estimator, X_list, y, valid_fold=3, verbose=False, exclusion_list = None):
            
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
                    sel_accuracy = ut.balanced_accuracy_score(testLabel, pred)
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
        self.winning_dict = []
        self.winning_list = []
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
            self.winning_dict.append(new_c)
            self.winning_list.append(new_c_l)
        
        return (self.winning_dict, self.winning_list)
    
    def performancesFeatures(self, fileToWrite = None, header = '', lock = None):
        
        self.outcome_dict_list = []
        
        for c_idx, config in enumerate(self.global_best_[0]):
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
                        
            self.outcome_dict_list.append(outcome_dict)
            
            if fileToWrite is not None and lock is not None:
                with lock:
                    with open(fileToWrite, "a") as myfile:
                        myfile.write(header)
                        myfile.write("Outcome Dict: {}\n\n".format(outcome_dict))
                        
        return self
                        