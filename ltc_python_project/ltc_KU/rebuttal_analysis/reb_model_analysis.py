#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:30:15 2024

@author: ahsan
"""
import shap
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
import imblearn
from imblearn.over_sampling import BorderlineSMOTE
import pickle

import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)

from autospearman_kla_python_impl import AutoSpearman

from imblearn.pipeline import Pipeline

import warnings
warnings.simplefilter('ignore') #we don't wanna see that

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']

shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/shap/"

data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'

# KU feature dimension index
dev_exp_features_ind = list(range(4,33))
cur_proj_features_ind = list(range(95,124))
prev_dev_exp_features_ind = list(range(124,153))
col_exp_features_ind = list(range(153,182))
prev_proj_features_ind = list(range(182,211))


prev_dev_exp_features_sum_ind = list(range(211,240))
col_exp_features_sum_ind = list(range(240,269))
prev_proj_features_sum_ind = list(range(269,298))

#Xin feature dimension index
xin_features_ind = list(range(33,94))

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index_30_jan_2025.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/models/" 

KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"

KU_OTHER_EXP_MED_AUTO    = "prev_exp_med_prof_auto"
KU_COMMUNITY_MED_AUTO    = "com_exp_med_prof_auto"
PREV_PROJ_PROF_MED_AUTO  = "prev_proj_med_prof_auto"

KU_OTHER_EXP_SUM_AUTO    = "prev_exp_sum_prof_auto"
KU_COMMUNITY_SUM_AUTO    = "com_exp_sum_prof_auto"
PREV_PROJ_PROF_SUM_AUTO  = "prev_proj_sum_prof_auto"

KU_ALL_MED_AUTO          = "ku_all_dim_med_prof_auto"
KU_ALL_SUM_AUTO          = "ku_all_dim_sum_prof_auto"

KU_ALL_MED_SUM_AUTO = "ku_all_dim_med_sum_prof_auto"

XIN_FEAT_AUTO        = "xin_feature_auto"

KU_ALL_MED_XIN_ALL_AUTO  = "ku_all_mid_xin_all_auto"
KU_ALL_SUM_XIN_ALL_AUTO  = "ku_all_sum_xin_all_auto"

KU_ALL_MED_SUM_XIN_ALL_AUTO = "ku_all_mid_sum_xin_all_auto"

KU_CUR_AUTO = "ku_present_proj_exp_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"

KULTC_SUM_AUTO = "kultc_sum_feat_auto"
KULTC_SUM_XIN_AUTO = "kultc_sum_xin_feat_auto"

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

def get_feature_names(col_list):
    dev_exp_ku_features = [col_list[i] for i in dev_exp_features_ind] 
    cur_proj_ku_features = [col_list[i] for i in cur_proj_features_ind]
    prev_dev_exp_features_med = [col_list[i] for i in prev_dev_exp_features_ind]
    col_exp_features_med = [col_list[i] for i in col_exp_features_ind]
    prev_proj_features_med = [col_list[i] for i in prev_proj_features_ind]
    
    prev_dev_exp_features_sum = [col_list[i] for i in prev_dev_exp_features_sum_ind]
    col_exp_features_sum = [col_list[i] for i in col_exp_features_sum_ind]
    prev_proj_features_sum = [col_list[i] for i in prev_proj_features_sum_ind]
    
    xin_features = [col_list[i] for i in xin_features_ind]
    
    return dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features_med, col_exp_features_med,prev_proj_features_med, prev_dev_exp_features_sum, col_exp_features_sum, prev_proj_features_sum, xin_features


def default_random_forest(boot_id, X_train, Y_train, X_test, Y_test):
    
    np.random.seed(boot_id)
    rf_clf = RandomForestClassifier(
        random_state=boot_id, n_estimators=100, n_jobs=30).fit(X_train, Y_train)
    rf_preds = rf_clf.predict(X_test)
    predicted_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_clf.feature_names = list(X_train.columns.values)

    Y_test.columns = ['target']
    Y_train.columns = ['target']
    total_train_inst = len(Y_train)
    total_test_inst = len(Y_test)
    train_positive = Y_train.target.sum()
    train_negative = len(Y_train) - train_positive
    test_positive = Y_test.target.sum()
    test_negative = len(Y_test) - test_positive

    precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(
        Y_test['target'], rf_preds, predicted_proba)

    result_dict = {
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'support': support,
        'far': far,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'auc': roc_value,
        'total_train_inst': total_train_inst,
        'total_test_inst': total_test_inst,
        'train_positive': train_positive,
        'train_negative': train_negative,
        'test_positive': test_positive,
        'test_negative': test_negative
    }
    rf_clf.model_result = result_dict
    
    return rf_clf


def smote_class_balacing(boot_id, X_train, Y_train, X_test, Y_test):
    pipeline = Pipeline([
        ('smote', BorderlineSMOTE(random_state=boot_id)),
        ('classifier', RandomForestClassifier(random_state=boot_id))
        ])
    
    sm = BorderlineSMOTE(random_state=boot_id, k_neighbors = 5, m_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, Y_train)
    rf_clf = default_random_forest(boot_id,X_res, y_res, X_test, Y_test)
    return rf_clf
    

def out_of_bootstrap_running(feature_name, X, Y, boot_lim, year_value, file_name):
    result = []
    sample_iterations = list(range(1,boot_lim + 1,1))
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]
    for boot_id in sample_iterations:
        boot_index_data= year_value_boot_data[year_value_boot_data['boot_id'] == boot_id]
        start_time = time.time()
        print('{} bootstrap = {}'.format(feature_name,boot_id))
        np.random.seed(boot_id)
        train_index = boot_index_data[boot_index_data['index_type']=='train_index']['index_value'].tolist()
        test_index = boot_index_data[boot_index_data['index_type']=='test_index']['index_value'].tolist()
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        
        rf_clf = default_random_forest(boot_id, X_train, Y_train, X_test, Y_test)
        rf_clf_smote = smote_class_balacing(boot_id, X_train, Y_train, X_test, Y_test)

        filename_rf = "{}feature-models/rf_{}_{}_{}.pkl".format(save_model_dir, file_name,boot_id,year_value)
        pickle.dump(rf_clf, open(filename_rf, 'wb'))
        
        
def full_model_running(feature_name, X, Y, year_value, file_name):
    np.random.seed(1)
    rf_clf = RandomForestClassifier(
        random_state=1, n_estimators=100, n_jobs=30).fit(X, Y)
    rf_clf.feature_names = list(X.columns.values)

    filename_rf = "{}feature-models/rf_{}_full_{}.pkl".format(save_model_dir, file_name,year_value)
    pickle.dump(rf_clf, open(filename_rf, 'wb'))
    
    explainer = shap.Explainer(rf_clf.predict, X, seed = 1)
    shap_values = explainer(X)
    shap_values.feature_names = rf_clf.feature_names
    
    shap_output_file = "{}{}_shap_full_{}.pkl".format(shap_value_output_dir,file_name,year_value)
    pickle.dump(shap_values, open(shap_output_file, 'wb'))
    
    
def run_auto_model(pd_full, model_feature_auto, year_value, Y):
    for model_name, feature_list in model_feature_auto.items():
        X = pd_full[feature_list]
        auto_feature = AutoSpearman(X_train = X, verbose=False)
        X = X[auto_feature]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    
def run_all_feature_model(pd_full, model_with_all_features, year_value, Y):
    for model_name, feature_list in model_with_all_features.items():
        X = pd_full[feature_list]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    
    
def run_full_model(pd_full, model_list, model_feature_list, year_value, Y):
    # FULL MODEL
    for model_name in model_list:
        feature_list = model_feature_list[model_name]
        X = pd_full[feature_list]
        auto_feature = AutoSpearman(X_train = X, verbose=False)
        X = X[auto_feature]
        full_model_running(model_name, X, Y, year_value, model_name)
        

def analyze_model_different_features(year_value, t_name):
    
    path = "{}ltc_data_full_year_{}_30_jan_2025_rem_mult_proj.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features_med, \
        col_exp_features_med,prev_proj_features_med, prev_dev_exp_features_sum, \
            col_exp_features_sum, prev_proj_features_sum, xin_features = get_feature_names(col_list)
    
    #dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features,\
    #    col_exp_features,prev_proj_features, xin_features = get_feature_names(col_list)

    all_ku_features_med =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features_med\
                        + col_exp_features_med\
                        + prev_proj_features_med
    all_ku_features_sum =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features_sum\
                        + col_exp_features_sum\
                        + prev_proj_features_sum
                        
    all_ku_features_med_sum =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features_med\
                        + col_exp_features_med\
                        + prev_proj_features_med\
                        + prev_dev_exp_features_sum\
                        + col_exp_features_sum\
                        + prev_proj_features_sum                     
                        
    combo_models_features_med = all_ku_features_med +  xin_features
    combo_models_features_sum = all_ku_features_sum +  xin_features
    combo_models_features_med_sum = all_ku_features_med_sum +  xin_features

    ku_dev_exp_xin_all_features = dev_exp_ku_features + xin_features
    
    ku_dev_exp_xin_all_features = dev_exp_ku_features + xin_features
    
    model_feature_auto = {
        KU_DEV_EXP_AUTO     :  dev_exp_ku_features,
        PROJECT_PROF_AUTO    : cur_proj_ku_features,
        KU_OTHER_EXP_MED_AUTO    : prev_dev_exp_features_med,
        KU_COMMUNITY_MED_AUTO    : col_exp_features_med,
        PREV_PROJ_PROF_MED_AUTO  : prev_proj_features_med,
        KU_OTHER_EXP_SUM_AUTO    : prev_dev_exp_features_sum,
        KU_COMMUNITY_SUM_AUTO    : col_exp_features_sum,
        PREV_PROJ_PROF_SUM_AUTO  : prev_proj_features_sum,
        KU_ALL_MED_AUTO          : all_ku_features_med,
        KU_ALL_SUM_AUTO          : all_ku_features_sum,
        KU_ALL_MED_SUM_AUTO      : all_ku_features_med_sum,
        XIN_FEAT_AUTO            :  xin_features,
        KU_ALL_MED_XIN_ALL_AUTO  : combo_models_features_med,
        KU_ALL_SUM_XIN_ALL_AUTO  : combo_models_features_sum,
        KU_ALL_MED_SUM_XIN_ALL_AUTO: combo_models_features_med_sum,
        KU_DEV_EXP_XIN_AUTO : ku_dev_exp_xin_all_features
        }
    
    #model_list = [KU_ALL_MED_AUTO, KU_ALL_SUM_AUTO, XIN_FEAT_AUTO,
    #              KU_ALL_MED_XIN_ALL_AUTO,KU_ALL_SUM_XIN_ALL_AUTO,
    #              KU_ALL_MED_SUM_AUTO,KU_ALL_MED_SUM_XIN_ALL_AUTO,
    #              KU_DEV_EXP_XIN_AUTO]
    
    model_list = [KU_ALL_SUM_XIN_ALL_AUTO]
    
    Y = get_target_label(year_value, pd_full, t_name)
    
    #KULTC_SUM_AUTO = "kultc_sum_feat_auto"
    #KULTC_SUM_XIN_AUTO = "kultc_sum_xin_feat_auto"

    #model_list = [KU_DEV_EXP_XIN_AUTO]
    
    #model_list = [KULTC_SUM_AUTO, KULTC_SUM_XIN_AUTO]
    
    #run_full_model(pd_full, full_model, model_feature_auto, year_value, Y)
    
    # Autospearman removing correlated features
    '''
    for model_name, feature_list in model_feature_auto.items():
        X = pd_full[feature_list]
        auto_feature = AutoSpearman(X_train = X, verbose=False)
        X = X[auto_feature]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    
    
    for model_name, feature_list in model_with_all_features.items():
        X = pd_full[feature_list]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    '''
    
    for model_name in model_list:
        feature_list = model_feature_auto[model_name]
        X = pd_full[feature_list]
        auto_feature = AutoSpearman(X_train = X, verbose=False)
        X = X[auto_feature]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    
    
def main():
    
    #for year_value in [1, 2, 3]:
    start_time = time.time()
    analyze_model_different_features(1, t_name[0])
    print(" YEAR 1 --- {:.4f} seconds ---".format(time.time() - start_time))
    
    
    start_time = time.time()
    analyze_model_different_features(2, t_name[0])
    print(" YEAR 2 --- {:.4f} seconds ---".format(time.time() - start_time)) 
    
    
    start_time = time.time()
    analyze_model_different_features(3, t_name[0])
    print(" YEAR 3 --- {:.4f} seconds ---".format(time.time() - start_time))
    
    
    print("main function..")

if __name__ == "__main__":
    main()
    print("Program finishes successfully.")
