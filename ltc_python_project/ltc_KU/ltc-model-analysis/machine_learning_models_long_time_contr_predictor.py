# -*- coding: utf-8 -*-

import pandas as pd
import sys, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import random
from datetime import datetime
import imblearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.simplefilter('ignore') #we don't wanna see that
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
data_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/ku_feature_dev_profile/'

# apache_storm_dev_ku_LTC.csv
def calculate_eval_metrics_II(y_true,y_pred, y_pred_prob):
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
        roc_value = roc_auc_score(y_true, y_pred_prob)
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
    
def combine_all_project_data():
    project_list = ["apache_lucene","apache_wicket","apache_activemq",
		"jruby_jruby","caskdata_cdap","apache_hbase",
		"apache_hive","apache_storm","elastic_elasticsearch","apache_groovy"]
    pd_list = []
    for project_name in project_list:
        file_path = "{}{}_dev_ku_LTC.csv".format(data_path, project_name)
        pd_data = pd.read_csv(file_path)
        pd_list.append(pd_data)

    combined_pd = pd.concat(pd_list)
    return combined_pd


def get_test_train_data(dev_sort_first_commit_pd,train_index, test_index, year_value):
    X_train = dev_sort_first_commit_pd.iloc[train_index[0]:train_index[len(train_index)-1], 3:32]
    X_test = dev_sort_first_commit_pd.iloc[test_index[0]:test_index[len(test_index)-1], 3:32]
    Y_train = dev_sort_first_commit_pd.iloc[train_index[0]:train_index[len(train_index)-1], (32 + (year_value - 1)) : (33 + (year_value - 1))] 
    Y_test = dev_sort_first_commit_pd.iloc[test_index[0]:test_index[len(test_index)-1], (32 + (year_value - 1)) : (33 + (year_value - 1))]    

    return X_train, Y_train, X_test, Y_test


def windows_based_testing(year):
    seed_value = 123
    nfold = 10
    result = []
    for i in range (1,10):
        num_training_window = i
        pd_data = combine_all_project_data()
        pd_data['first_java_commit_date_time'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z' ) for x in pd_data['FirstJavaCommit']]
        # list(pd_data.columns)
        dev_sort_first_commit_pd = pd_data.sort_values(by='first_java_commit_date_time')
        
        num_records = dev_sort_first_commit_pd.shape[0]
        equal_window_size = int(num_records/nfold)
        data_windows = create_equal_size_winow(nfold, num_records, equal_window_size)
        train_index, test_index = get_train_test_split_train_window(num_training_window, data_windows, nfold)
    
        X_train, Y_train, X_test, Y_test = get_test_train_data(dev_sort_first_commit_pd, train_index, test_index, year)
        X = X_train
        y = Y_train
        #oversample =  imblearn.over_sampling.SMOTE(random_state=seed_value)
        #X, y = oversample.fit_resample(X_train, Y_train)
    
        rf_clf = RandomForestClassifier(random_state=seed_value, n_estimators = 10).fit(X, y)
        rf_preds = rf_clf.predict(X_test)
        predicted_proba = rf_clf.predict_proba(X_test)[:,1]
        
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds, predicted_proba)
        result.append([num_training_window, precision, recall, fscore, tp, fp, tn, fn, roc_value, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
    result_pd = pd.DataFrame(result)
    result_pd.columns = ['training_window','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    result_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/results/'
    result_pd.to_csv("{}training_window_1_9_LTC_{}_years.csv".format(result_path, year), index = False) 


def k_fold_stratified_windows_based_testing(year_value):
    seed_value = 123
    nfold = 10
    result = []
   
       
    pd_data = combine_all_project_data()
    pd_data['first_java_commit_date_time'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z' ) for x in pd_data['FirstJavaCommit']]
    # list(pd_data.columns)
    dev_sort_first_commit_pd = pd_data.sort_values(by='first_java_commit_date_time')
        
    x = dev_sort_first_commit_pd.iloc[ : , 3:32]
    y = dev_sort_first_commit_pd.iloc[ : , (32 + (year_value - 1)) : (33 + (year_value - 1))] 
    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
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
    
        rf_clf = RandomForestClassifier(random_state=seed_value, n_estimators = 10).fit(X_train, Y_train)
        rf_preds = rf_clf.predict(X_test)
        predicted_proba = rf_clf.predict_proba(X_test)[:,1]
        
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds, predicted_proba)
        result.append([fold, precision, recall, fscore, tp, fp, tn, fn, roc_value, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
    result_pd = pd.DataFrame(result)
    result_pd.columns = ['fold','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    result_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/results/'
    result_pd.to_csv("{}k_fold_LTC_{}_years_updated_June_17.csv".format(result_path, year_value), index = False) 
    
    
def predict_long_time_contributor():
    seed_value = 123
    nfold = 10
    num_training_window = 6
    #file_path = "{}{}_dev_ku_LTC.csv".format(data_path, project_name)
    #pd_data = pd.read_csv(file_path)
    
    pd_data = combine_all_project_data()
    pd_data['first_java_commit_date_time'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z' ) for x in pd_data['FirstJavaCommit']]
    # list(pd_data.columns)
    dev_sort_first_commit_pd = pd_data.sort_values(by='first_java_commit_date_time')

    num_records = dev_sort_first_commit_pd.shape[0]
    equal_window_size = int(num_records/nfold)
    data_windows = create_equal_size_winow(nfold, num_records, equal_window_size)
    train_index, test_index = get_train_test_split_train_window(num_training_window, data_windows, nfold)
    
    X_train, Y_train, X_test, Y_test = get_test_train_data(dev_sort_first_commit_pd, train_index, test_index, 3)
    X = X_train
    y = Y_train
    #oversample =  imblearn.over_sampling.SMOTE(random_state=seed_value)
    #X, y = oversample.fit_resample(X_train, Y_train)
    
    
    
    rf_clf = RandomForestClassifier(random_state=seed_value).fit(X, y)
    rf_preds = rf_clf.predict(X_test)
    predicted_proba = rf_clf.predict_proba(X_test)[:,1]
    Y_test.columns = ['target']
    
    precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds, predicted_proba)
    #precision,recall,fscore,support = precision_recall_fscore_support(Y_test,rf_preds,average='binary')    

if __name__ == "__main__":
    #windows_based_testing(1)
    #windows_based_testing(2)
    #windows_based_testing(3)
    for i in range(1,4):
        k_fold_stratified_windows_based_testing(i)
    print("Program finishes successfully")