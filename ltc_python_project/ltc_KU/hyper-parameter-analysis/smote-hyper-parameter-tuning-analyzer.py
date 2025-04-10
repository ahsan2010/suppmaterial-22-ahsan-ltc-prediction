#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:02:06 2024

@author: ahsan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:30:15 2024

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
import imblearn
from imblearn.over_sampling import BorderlineSMOTE
import pickle

import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)

from autospearman_kla_python_impl import AutoSpearman

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

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

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 


KU_DEV_EXP      = "dev_exp_prof"
KU_OTHER_EXP    = "prev_exp_prof"
KU_COMMUNITY    = "com_exp_prof"
PROJECT_PROF    = "proj_prof"
PREV_PROJ_PROF  = "prev_proj_prof"
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_MINUS_EXP = "ku_all_minus_dev_exp"
KU_ALL_MINUS_PREV_EXP = "ku_all_minus_prev_exp"
KU_ALL_MINUS_COMM_EXP = "ku_all_minus_comm_exp"
KU_ALL_MINUS_PROJ_PROF = "ku_all_minus_proj_prof"
KU_ALL_MINUS_PREV_PROJ = "ku_all_minus_prev_proj_prof"

KU_CUR = "ku_present_proj_exp"
KU_CUR_XIN = "ku_present_proj_exp_xin_all"


KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
KU_OTHER_EXP_AUTO    = "prev_exp_prof_auto"
KU_COMMUNITY_AUTO    = "com_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"
PREV_PROJ_PROF_AUTO  = "prev_proj_prof_auto"
KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

KU_ALL_MINUS_EXP_AUTO = "ku_all_minus_dev_exp_auto"
KU_ALL_MINUS_PREV_EXP_AUTO = "ku_all_minus_prev_exp_auto"
KU_ALL_MINUS_COMM_EXP_AUTO = "ku_all_minus_comm_exp_auto"
KU_ALL_MINUS_PROJ_PROF_AUTO = "ku_all_minus_proj_prof_auto"
KU_ALL_MINUS_PREV_PROJ_AUTO = "ku_all_minus_prev_proj_prof_auto"

KU_CUR_AUTO = "ku_present_proj_exp_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"

parameters = {
    'smote__k_neighbors': [2, 3, 5, 7, 10, 15],
    'smote__m_neighbors': [2, 3, 5, 7, 10, 15, 20],
    'smote__kind' : ['borderline-1', 'borderline-2'] 
}

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
    prev_dev_exp_features = [col_list[i] for i in prev_dev_exp_features_ind]
    col_exp_features = [col_list[i] for i in col_exp_features_ind]
    prev_proj_features = [col_list[i] for i in prev_proj_features_ind]
    
    xin_features = [col_list[i] for i in xin_features_ind]
    
    return dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features, col_exp_features,prev_proj_features, xin_features


def default_random_forest(year_value, boot_id, X_train, Y_train, X_test, Y_test):
    
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


def smote_tomek(year_value, boot_id, X_train, Y_train, X_test, Y_test):
    smt = SMOTETomek(random_state=boot_id)
    X_res, y_res = smt.fit_resample(X_train, Y_train)
    rf_clf = default_random_forest(year_value, boot_id, X_res, y_res, X_test, Y_test)
    return rf_clf

def smote_class_balacing(year_value, boot_id, X_train, Y_train, X_test, Y_test):
    start_time = time.time()
    pipeline = Pipeline([
        ('smote', BorderlineSMOTE(random_state=boot_id)),
        ('classifier', RandomForestClassifier(random_state=boot_id, n_estimators=100))
        ])
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy', n_jobs=30)
    grid_search.fit(X_train, Y_train)
    rf_preds = grid_search.predict(X_test)
    predicted_proba = grid_search.predict_proba(X_test)[:, 1]
    grid_search.feature_names = list(X_train.columns.values)
    
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
    grid_search.model_result = result_dict
    print("SMOTE: YEAR {}  BOOT {} --- {:.4f} seconds ---".format(year_value, boot_id, time.time() - start_time))
    
    return grid_search


    

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
        
        rf_clf = default_random_forest(year_value, boot_id, X_train, Y_train, X_test, Y_test)
        #rf_clf_smote = smote_class_balacing(year_value, boot_id, X_train, Y_train, X_test, Y_test)
        rf_clf_smote_tomek = smote_tomek(year_value, boot_id, X_train, Y_train, X_test, Y_test)

        print("Result: {} Year {} Boot {} D: {:.2} S: {:.2}".format(feature_name, year_value, boot_id, rf_clf.model_result['auc'],rf_clf_smote_tomek.model_result['auc']))
        #filename_rf = "{}feature-models/rf_somte_{}_{}_{}.pkl".format(save_model_dir, file_name,boot_id,year_value)
        filename_rf = "{}feature-models/rf_somte_tomek_{}_{}_{}.pkl".format(save_model_dir, file_name,boot_id,year_value)
        
        pickle.dump(rf_clf_smote_tomek, open(filename_rf, 'wb'))

def analyze_model_different_features(year_value, t_name):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    col_list = list(pd_full.columns)
    
    
    dev_exp_ku_features, cur_proj_ku_features, prev_dev_exp_features,\
        col_exp_features,prev_proj_features, xin_features = get_feature_names(col_list)

    all_ku_features =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features\
                        + prev_proj_features
                        
    ku_all_minus_exp = cur_proj_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features\
                        + prev_proj_features
                        
    ku_all_minus_prev_exp =     dev_exp_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features\
                        + prev_proj_features
    
    ku_all_minus_col =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features\
                        + prev_proj_features
                        
    ku_all_minus_proj_prof =     dev_exp_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features\
                        + prev_proj_features
                        
    ku_all_minus_prev_proj_prof =     dev_exp_ku_features\
                        + cur_proj_ku_features\
                        + prev_dev_exp_features\
                        + col_exp_features
    
    ku_cur_features = dev_exp_ku_features\
                        + cur_proj_ku_features
    ku_cur_xin_all_features = ku_cur_features + xin_features
    
    combo_models_features = all_ku_features +  xin_features

    
    model_with_all_features = {
       KU_DEV_EXP : dev_exp_ku_features,
       KU_OTHER_EXP : prev_dev_exp_features,
       KU_COMMUNITY : col_exp_features,
       PROJECT_PROF : cur_proj_ku_features,
       PREV_PROJ_PROF : prev_proj_features,
       KU_ALL : all_ku_features,
       XIN_FEAT: xin_features,
       KU_ALL_XIN_ALL: combo_models_features,
       KU_ALL_MINUS_EXP : ku_all_minus_exp,
       KU_ALL_MINUS_PREV_EXP: ku_all_minus_prev_exp,
       KU_ALL_MINUS_COMM_EXP: ku_all_minus_col,
       KU_ALL_MINUS_PROJ_PROF : ku_all_minus_proj_prof,
       KU_ALL_MINUS_PREV_PROJ: ku_all_minus_prev_proj_prof,
       KU_CUR : ku_cur_features,
       KU_CUR_XIN : ku_cur_xin_all_features
       }

    model_feature_auto = {
        KU_DEV_EXP_AUTO : dev_exp_ku_features,
        KU_OTHER_EXP_AUTO : prev_dev_exp_features,
        KU_COMMUNITY_AUTO : col_exp_features,
        PROJECT_PROF_AUTO : cur_proj_ku_features,
        PREV_PROJ_PROF_AUTO : prev_proj_features,
        KU_ALL_AUTO : all_ku_features,
        XIN_FEAT_AUTO : xin_features,
        KU_ALL_XIN_ALL_AUTO : combo_models_features,
        KU_ALL_MINUS_EXP_AUTO : ku_all_minus_exp,
        KU_ALL_MINUS_PREV_EXP_AUTO: ku_all_minus_prev_exp,
        KU_ALL_MINUS_COMM_EXP_AUTO: ku_all_minus_col,
        KU_ALL_MINUS_PROJ_PROF_AUTO : ku_all_minus_proj_prof,
        KU_ALL_MINUS_PREV_PROJ_AUTO: ku_all_minus_prev_proj_prof,
        KU_CUR_AUTO : ku_cur_features,
        KU_CUR_XIN_AUTO : ku_cur_xin_all_features
        }
    
    Y = get_target_label(year_value, pd_full, t_name)
    # Autospearman removing correlated features
    
    model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO, KU_ALL_XIN_ALL_AUTO]
    
    for model_name in model_list:
        feature_list = model_feature_auto[model_name]
        X = pd_full[feature_list]
        auto_feature = AutoSpearman(X_train = X, verbose=False)
        X = X[auto_feature]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    
    '''
    for model_name, feature_list in model_with_all_features.items():
        X = pd_full[feature_list]
        out_of_bootstrap_running(model_name, X, Y, 100, year_value, model_name)
    '''
    
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
    
    

if __name__ == "__main__":
    main()
    print("Program finishes successfully.")
