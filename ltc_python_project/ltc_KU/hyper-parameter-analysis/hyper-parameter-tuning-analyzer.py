#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:49:40 2024

@author: ahsan
"""

import pandas as pd
import os.path
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import xgboost as xgb
import time

import pickle

import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)

from autospearman_kla_python_impl import AutoSpearman

import warnings
warnings.simplefilter('ignore') #we don't wanna see that

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']


data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'

# KU feature dimension index
dev_exp_features_ind = list(range(4,32))
cur_proj_features_ind = list(range(95,123))
prev_dev_exp_features_ind = list(range(124,152))
col_exp_features_ind = list(range(153,181))
prev_proj_features_ind = list(range(182,210))

#Xin feature dimension index
xin_features_ind = list(range(33,94))

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"
KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)

#save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 
save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/hyper-parameter-tune-cheaper-model/" 

param_grid_random_forest = {
        'n_estimators' : [10, 50, 100, 200], # number of tree in the forest
        'max_depth': [None, 5, 10] # maximum depth of the tree
    
    }

param_grid_decision_tree = {
     'criterion' : ['gini', 'entropy', 'log_loss'],
     'max_depth': [None, 5, 10], # maximum depth of the tree
     'ccp_alpha' : [0.0001, 0.001, 0.01, 0.1, 0.5] # complexity
    }

param_grid_knn = {
        'n_neighbors' : [1, 5, 9, 13, 17, 20]
    }

param_grid_naive_bayes = {
        'var_smoothing' : [1e-5, 1e-9, 1e-11, 1e-15]
    }

param_grid_svm = {
        'C': [0.1, 0.5, 1, 10],  
        'kernel': ['rbf', 'linear','sigmoid']
    }

param_grid_xgboost = {
    'n_estimators' : [10, 50, 100, 200],
    'max_depth': [None, 5, 10],
    'learning_rate': [0.1, 0.01, 0.001]
    }

param_lgbm_classifier = {
    'n_estimators' : [10, 50, 100, 200],
    'num_leaves' : [10, 30, 50, 100],
    'learning_rate': [0.1, 0.01, 0.001]
        
    }
  

def get_target_label(year_value, pd_full, target_name):
    if year_value == 1:
        Y = pd_full[['LTC_Developer_Cat_Year_One']]
    elif year_value == 2:
        Y = pd_full[['LTC_Developer_Cat_Year_Two']]
    elif year_value == 3:
        Y = pd_full[['LTC_Developer_Cat_Year_Three']]
    

    if target_name == t_name[1]:
        Y = pd_full[['is_ltc']]
        
    return Y


#Calculate Evaluaation Metrics
def calculate_eval_metrics_II(y_true, y_pred, y_pred_prob):
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





def get_feature_names(col_list):
    dev_exp_ku_features = [col_list[i] for i in dev_exp_features_ind] 
    cur_proj_ku_features = [col_list[i] for i in cur_proj_features_ind]
    prev_dev_exp_features = [col_list[i] for i in prev_dev_exp_features_ind]
    col_exp_features = [col_list[i] for i in col_exp_features_ind]
    prev_proj_features = [col_list[i] for i in prev_proj_features_ind]
    
    xin_features = [col_list[i] for i in xin_features_ind]
    
    return dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features, col_exp_features,prev_proj_features, prev_proj_features, xin_features
    


def hyper_parameter_model_analysis(X, y, model_name, year_value, boot_id):
    
    start_time = time.time()
    rf = RandomForestClassifier(random_state = boot_id)
    grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_random_forest, scoring='roc_auc', cv = 10, n_jobs = 30)
    grid_search_rf.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_rf_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_rf, open(filename_clf, 'wb'))
    
    
    start_time = time.time()
    clf = DecisionTreeClassifier(random_state=boot_id)
    grid_search_clf = GridSearchCV(estimator = clf, param_grid = param_grid_decision_tree, scoring='roc_auc', cv = 10, n_jobs = 30)
    grid_search_clf.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_dtree_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_clf, open(filename_clf, 'wb'))

    start_time = time.time()
    neigh = KNeighborsClassifier()
    grid_search_neigh = GridSearchCV(estimator = neigh, param_grid = param_grid_knn, scoring='roc_auc', cv = 10, n_jobs = 30)
    grid_search_neigh.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_knn_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_neigh, open(filename_clf, 'wb'))


    start_time = time.time()
    naive = GaussianNB()
    grid_search_naive = GridSearchCV(estimator = naive, param_grid = param_grid_naive_bayes, scoring='roc_auc', cv = 10, n_jobs = 30)
    grid_search_naive.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_NB_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_naive, open(filename_clf, 'wb'))
    
    '''
    start_time = time.time()
    svm_cls = SVC(cache_size = 20000)
    grid_search_svm_cls = GridSearchCV(estimator = svm_cls, param_grid = param_grid_svm, scoring='roc_auc', cv = 10, n_jobs = 30, verbose = 3)
    grid_search_svm_cls.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_SVM_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_svm_cls, open(filename_clf, 'wb'))
    '''
    
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(random_state=boot_id)
    grid_search_xgb = GridSearchCV(estimator = xgb_model, param_grid = param_grid_xgboost, scoring='roc_auc', cv = 10, n_jobs = 30)
    grid_search_xgb.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_XGB_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_xgb, open(filename_clf, 'wb'))
    
    start_time = time.time()
    lgm = LGBMClassifier(random_state=boot_id)
    grid_search_lgm = GridSearchCV(estimator = lgm, param_grid = param_lgbm_classifier, scoring='roc_auc', cv = 10, n_jobs = 30)
    grid_search_lgm.fit(X, y)
    #print(" YEAR 1 --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
    filename_clf = "{}hyper-parameter-tuning/{}_LGBM_{}_{}.pkl".format(save_model_dir,model_name,boot_id,year_value)
    pickle.dump(grid_search_lgm, open(filename_clf, 'wb'))
    
    print(" YEAR {} Boot {} --- {:.4f} minutes ---".format(year_value, boot_id, (time.time() - start_time)/60.0))
    
    


def out_of_bootstrap_running_with_hyperparameter_tuning(model_name, X, Y, start_boot, boot_lim, year_value):
    result = []
    sample_iterations = list(range(start_boot, boot_lim + 1,1))
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]
    for boot_id in sample_iterations:
        boot_index_data= year_value_boot_data[year_value_boot_data['boot_id'] == boot_id]
        start_time = time.time()
        print('{} bootstrap = {}'.format(model_name,boot_id), end = " ")
        np.random.seed(boot_id)
        train_index = boot_index_data[boot_index_data['index_type']=='train_index']['index_value'].tolist()
        test_index = boot_index_data[boot_index_data['index_type']=='test_index']['index_value'].tolist()
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        hyper_parameter_model_analysis(X_train, Y_train, model_name, year_value, boot_id)
        

def analyze_hyper_parameter(year_value, t_name):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    col_list = list(pd_full.columns)
    
    
    dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features,\
        col_exp_features,prev_proj_features, prev_proj_features, xin_features = get_feature_names(col_list)

    all_ku_features =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features\
                        + prev_proj_features

    combo_models_features = all_ku_features +  xin_features
    
    ku_cur_features = dev_exp_ku_features\
                        + cur_proj_ku_features
    ku_cur_xin_all_features = ku_cur_features + xin_features
    
    ku_dev_exp_xin_all_featues = dev_exp_ku_features + xin_features
    
    
    # Autospearman removing correlated features
    X = pd_full[ku_dev_exp_xin_all_featues]
    auto_feature = AutoSpearman(X_train = X, verbose=False)
    X = X[auto_feature]
    Y = get_target_label(year_value, pd_full, t_name)
    out_of_bootstrap_running_with_hyperparameter_tuning(KU_DEV_EXP_XIN_AUTO, X, Y, 1, 100, year_value)
    
    '''
    # Autospearman removing correlated features
    X = pd_full[ku_cur_xin_all_features]
    auto_feature = AutoSpearman(X_train = X, verbose=False)
    X = X[auto_feature]
    Y = get_target_label(year_value, pd_full, t_name)
    out_of_bootstrap_running_with_hyperparameter_tuning(KU_CUR_XIN_AUTO, X, Y, 51, 59, year_value)
    '''


def hyper_param_with_test(year_value, t_name):
    
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    col_list = list(pd_full.columns)
    
    dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features,\
        col_exp_features,prev_proj_features, prev_proj_features, xin_features = get_feature_names(col_list)

    all_ku_features =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features\
                        + prev_proj_features

    combo_models_features = all_ku_features +  xin_features
    
    ku_cur_features = dev_exp_ku_features\
                        + cur_proj_ku_features
    ku_cur_xin_all_features = ku_cur_features + xin_features
    
    ku_dev_exp_xin_all_features = dev_exp_ku_features + xin_features
    
    #MODEL = KU_ALL_XIN_ALL_AUTO
    #MODEL = KU_CUR_XIN_AUTO
    MODEL = KU_DEV_EXP_XIN_AUTO
    
    X = pd_full[ku_dev_exp_xin_all_features]
    
    
    #X = pd_full[combo_models_features]
    #X = X[feature_names]
    Y = get_target_label(year_value, pd_full, t_name)

    algo_model = ['rf', 'dtree', 'knn', 'NB', 'XGB', 'LGBM']
    boot_lim = 100
    sample_iterations = list(range(1, boot_lim + 1,1))
    
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]

    for boot_id in sample_iterations:
        print("WORKING {} YEAR {} BOOT {}".format(MODEL, year_value, boot_id))
        boot_index_data= year_value_boot_data[year_value_boot_data['boot_id'] == boot_id]
        train_index = boot_index_data[boot_index_data['index_type']=='train_index']['index_value'].tolist()
        test_index = boot_index_data[boot_index_data['index_type']=='test_index']['index_value'].tolist()
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        filename_clf = "{}hyper-parameter-tuning/{}_{}_{}_{}.pkl".format(save_model_dir,MODEL, 'rf', boot_id, year_value)
        loaded_model = pickle.load(open(filename_clf, 'rb'))
        feature_names = list(loaded_model.feature_names_in_)
    
        
        for m in algo_model:
            #print(f"{m}")
            filename_clf = "{}hyper-parameter-tuning/{}_{}_{}_{}.pkl".format(save_model_dir,MODEL, m, boot_id,year_value)
            loaded_model = pickle.load(open(filename_clf, 'rb'))
            X_test = X_test[feature_names]
            preds = loaded_model.predict(X_test)
            predicted_proba = loaded_model.predict_proba(X_test)[:,1]
            precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],preds, predicted_proba)
       
            result_dict = {
                'precision' : precision,
                'recall' : recall,
                'fscore' : fscore,
                'support' : support,
                'far' : far,
                'tp' : tp,
                'fp' : fp,
                'tn' : tn,
                'fn' : fn,
                'auc' : roc_value,
                'total_train_inst' : total_train_inst, 
                'total_test_inst': total_test_inst, 
                'train_positive' : train_positive, 
                'train_negative': train_negative, 
                'test_positive' : test_positive, 
                'test_negative' : test_negative
            }
            loaded_model.model_result = result_dict
            fname = "{}hyper-parameter-tuning-with-test/{}_{}_{}_{}.pkl".format(save_model_dir,MODEL, m, boot_id,year_value)
            pickle.dump(loaded_model, open(fname, 'wb'))

def hyper_param_model_result_analysis():
    for year_value in [1,2,3]:
        hyper_param_with_test(year_value, t_name[0])
    
def main():
    #for year_value in [1, 2, 3]:
    #analyze_hyper_parameter(2, t_name[0])
    hyper_param_model_result_analysis()
    #hyper_param_with_test(3, t_name[0])
    print("main function..")

if __name__ == "__main__":
    main()
    print("Program finishes successfully.")