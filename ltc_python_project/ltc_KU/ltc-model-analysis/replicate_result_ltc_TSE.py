# -*- coding: utf-8 -*-
import pandas as pd
import sys, os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

# apache_storm_dev_ku_LTC.csv
def calculate_eval_metrics_II(y_true,y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
        
    tp = fp = tn = fn = 0
    total_pos= total_neg = 0
    for i in range(len(y_true)):
        if((y_true[i] == 1) and (y_pred[i] == 1)):
            tp = tp + 1
        if((y_true[i] == 0) and (y_pred[i] == 1)):
            fp = fp + 1
        if((y_true[i] == 1) and (y_pred[i] == 0)):
            fn = fn + 1
        if((y_true[i] == 0) and (y_pred[i] == 0)):
            tn = tn + 1
                
        if(y_true[i] == 1):
            total_pos = total_pos + 1
        if(y_true[i] == -1):
            total_neg = total_neg + 1
    
    try:
        roc_value = roc_auc_score(y_true, y_pred)
    except Exception as exp:
        print(exp)
        roc_value = 0.0
        
    precision,recall,fscore,support = precision_recall_fscore_support(y_true,y_pred,average='binary')    
    far =   fp/max(1,(tp + tn))
        
    return precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value

def create_equal_size_winow(nfold, num_records, equal_window_size):
    windows = []
    start = 0
    for i in range (0,nfold):
        end = start + equal_window_size
        if i == nfold - 1:
            end = num_records
        print("{}/{} Start {} End {}".format(i, nfold, start, end))
        windows.append(list(range(start, end)))
        start = end
        
    return windows

def get_train_test_split_train_window(num_training_window, data_windows, nfold):
    train_index = []
    test_index = []
    for i in range(0, num_training_window):
        train_index = train_index + data_windows[i]
    
    for j in range(num_training_window, nfold):
        test_index = test_index + data_windows[j]
        
    return train_index, test_index

def get_test_train_xin_data(dev_sort_first_commit_pd,train_index, test_index, year_value):
    X_train = dev_sort_first_commit_pd.iloc[train_index[0]:train_index[len(train_index)-1], 3:65]
    X_test = dev_sort_first_commit_pd.iloc[test_index[0]:test_index[len(test_index)-1], 3:65]
    Y_train = dev_sort_first_commit_pd.iloc[train_index[0]:train_index[len(train_index)-1], 65:66] 
    Y_test = dev_sort_first_commit_pd.iloc[test_index[0]:test_index[len(test_index)-1], 65:66]    

    return X_train, Y_train, X_test, Y_test

def k_fold_stratified_windows_based_testing_xin_data(year):
    seed_value = 123
    nfold = 10
    result = []
   
    first_year_ltc = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/data_year_{}.csv".format(year)
    pd_data = pd.read_csv(first_year_ltc)   
        
    x = pd_data.iloc[ : , 3:65]
    y = pd_data.iloc[ : , 65:66] 
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
    fold = 0
    for train_index, test_index in skf.split(x, y):
        #print(len(train_index))
        #print(len(test_index))
        print("Fold {}".format(fold))
        fold = fold + 1
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
        #oversample =  imblearn.over_sampling.SMOTE(random_state=seed_value)
        #X, y = oversample.fit_resample(X_train, Y_train)
    
        rf_clf = RandomForestClassifier(random_state=seed_value, n_estimators = 100).fit(X_train, Y_train)
        rf_preds = rf_clf.predict(X_test)
        
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds)
        result.append([fold, precision, recall, fscore, tp, fp, tn, fn, roc_value, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
    result_pd = pd.DataFrame(result)
    result_pd.columns = ['fold','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    result_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/'
    result_pd.to_csv("{}k_fold_LTC_{}_years.csv".format(result_path, year), index = False) 
        
    