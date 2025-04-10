# -*- coding: utf-8 -*-
import pandas as pd
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
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

from lightgbm import LGBMClassifier
import time


import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)

from autospearman_kla_python_impl import AutoSpearman

import warnings
warnings.simplefilter('ignore') #we don't wanna see that

def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x

#Calculate Evaluaation Metrics
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


pd_list = []
t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']

def out_of_bootstrap_running(target_name, feature_name, X, Y, year_value = 1, file_name = 'bootstrap_result'):
    result = []
    indices = list(range(0,X.shape[0] ,1))
    sample_iterations = list(range(1,20 + 1,1))
    
    for i in sample_iterations:
        print('{} bootstrap = {}'.format(feature_name,i))
        np.random.seed(i)
        train_index = list(np.random.choice(indices,len(indices)))
        test_index = np.setdiff1d(indices, train_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        rf_clf = RandomForestClassifier(random_state=i, n_estimators = 100).fit(X_train, Y_train)
        rf_preds = rf_clf.predict(X_test)
        predicted_proba = rf_clf.predict_proba(X_test)[:,1]
        
        logistic_clf = LogisticRegression(random_state=i).fit(X_train, Y_train)
        logistic_preds = logistic_clf.predict(X_test)
        logistic_pred_prob = logistic_clf.predict_proba(X_test)[: ,1]
        
        lgbm_clf = LGBMClassifier(objective='binary', random_state=5).fit(X_train, Y_train)
        lgbm_preds = lgbm_clf.predict(X_test)
        lgbm_pred_prob = lgbm_clf.predict_proba(X_test)[: ,1]
        
        
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds, predicted_proba)
        precision_log, recall_log, fscore_log, support_log, far_log, tp_log, fp_log, tn_log, fn_log, roc_value_log = calculate_eval_metrics_II(Y_test['target'],logistic_preds, logistic_pred_prob)
        precision_lgbm, recall_lgbm, fscore_lgbm, support_lgbm, far_lgbm, tp_lgbm, fp_lgbm, tn_lgbm, fn_lgbm, roc_value_lgbm = calculate_eval_metrics_II(Y_test['target'],lgbm_preds, lgbm_pred_prob)
        
        
        result.append([year_value,target_name, feature_name,"RandomForest", i, precision, recall, fscore, tp, fp, tn, fn, roc_value, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
        result.append([year_value, target_name, feature_name, "LogisticRegression",i, precision_log, recall_log, fscore_log, tp_log, fp_log, tn_log, fn_log, roc_value_log, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
        result.append([year_value, target_name, feature_name, "LightGBM",i, precision_lgbm, recall_lgbm, fscore_lgbm, tp_lgbm, fp_lgbm, tn_lgbm, fn_lgbm, roc_value_lgbm, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
        
    result_pd = pd.DataFrame(result)
    result_pd.columns = ['ltc_year','ltc_approach','features','classifier','bootstrap','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    
    return result_pd


def bootstrap_model_run(year_value,target_name):
    path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/full_result_updated_label_ltc{}.csv'.format(year_value)
    
    pd_full = pd.read_csv(path)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]

    Y = None
    if year_value == 1:
        Y = pd_full[['LTC_Developer_Cat_Year_One']]
    elif year_value == 2:
        Y = pd_full[['LTC_Developer_Cat_Year_Two']]
    elif year_value == 3:
        Y = pd_full[['LTC_Developer_Cat_Year_Three']]
    

    if target_name == t_name[1]:
        Y = pd_full[['is_ltc']]

    output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/run-ku-model-autospearman/"
    
    X = pd_full[ku_features_count]
    sp_ku_count_features = AutoSpearman(X_train = X, verbose=False)
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','KU_COUNT', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_count_boostrap.csv"))

    X = pd_full[ku_features_norm]
    sp_ku_norm_features = AutoSpearman(X_train = X, verbose=False)
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','KU_NORM', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_norm_boostrap.csv"))
    
    X = pd_full[xin_features + ku_features_count]
    sp_xin_ku_count_features = AutoSpearman(X_train = X, verbose=False)
    result_xin_features = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES_KU_COUNT', X, Y, year_value = year_value, file_name = 'xin_features_boostrap')
    result_xin_features.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_count_xin_features_boostrap.csv"))

    
    X = pd_full[xin_features]
    sp_xin_features = AutoSpearman(X_train = X, verbose=False)
    result_xin_features = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES', X, Y, year_value = year_value, file_name = 'xin_features_boostrap')
    result_xin_features.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"xin_features_boostrap.csv"))

    '''
    X = pd_full[xin_features + ku_full_features]
    sp_xin_ku_full_features = AutoSpearman(X_train = X, verbose=False)
    result_xin_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES_KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ku_full_xin_features_boostrap')
    result_xin_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_full_xin_features_boostrap.csv"))

    X = pd_full[ku_full_features]
    sp_ku_full_features = AutoSpearman(X_train = X, verbose=False)
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_full_boostrap.csv"))

    X = pd_full[xin_features + ku_features_norm]
    sp_xin_ku_norm_features = AutoSpearman(X_train = X, verbose=False)
    result_xin_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES_KU_NORM', X, Y, year_value = year_value, file_name = 'ku_full_xin_features_boostrap')
    result_xin_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_norm_xin_features_boostrap.csv"))
    '''

def bootstrap_analysis():
    start_time = time.time()
    bootstrap_model_run(1,t_name[0])
    print(" YEAR 1 --- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    bootstrap_model_run(2,t_name[0])
    print(" YEAR 2 --- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    bootstrap_model_run(3,t_name[0])
    print(" YEAR 3 --- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    bootstrap_analysis()