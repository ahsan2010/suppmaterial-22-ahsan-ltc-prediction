# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

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

import pickle

import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)

from autospearman_kla_python_impl import AutoSpearman

import warnings
warnings.simplefilter('ignore') #we don't wanna see that


dev_month_ku_expertise_file_dir             = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/'
project_ku_profile_dir                      = "/home/local/SAIL/ahsan/LTC_Project/project-ku-profile/"
dev_other_project_experience_ku_profile_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
community_ku_profile_dir                    = "/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/"    
other_project_ku_profile_dir                = "/home/local/SAIL/ahsan/LTC_Project/other-project-ku-profile/"

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"

save_model_dir              = "/home/local/SAIL/ahsan/LTC_Project/save-model-ltc-single-data-file/" 
#save_model_dir              = "/home/local/SAIL/ahsan/LTC_Project/saved-model-ltc/"
rf_model_output_path        =  save_model_dir + "random-forest/"
lr_model_output_path        =  save_model_dir + "logistic-regression/" 
lgbm_model_output_path      =  save_model_dir + "light-gbm/"

KU_DEV_EXP      = "dev_exp_prof"
KU_OTHER_EXP    = "prev_exp_prof"
KU_COMMUNITY    = "com_exp_prof"
PROJECT_PROF    = "proj_prof"
PREV_PROJ_PROF  = "prev_proj_prof"
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
KU_OTHER_EXP_AUTO    = "prev_exp_prof_auto"
KU_COMMUNITY_AUTO    = "com_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"
PREV_PROJ_PROF_AUTO  = "prev_proj_prof_auto"
KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

STACK_KU_ALL = "stack_ku_all_dim"
STACK_KU_ALL_XIN = "stack_ku_all_dim_with_xin"

STACK_KU_ALL_AUTO = "stack_ku_all_dim_auto"
STACK_KU_ALL_XIN_AUTO = "stack_ku_all_dim_with_xin_auto"


model_list = [KU_DEV_EXP,
              KU_OTHER_EXP,
              KU_COMMUNITY,
              PROJECT_PROF,
              PREV_PROJ_PROF,
              KU_ALL,
              XIN_FEAT,
              KU_ALL_XIN_ALL]

model_list_auto = [KU_DEV_EXP_AUTO,
              KU_OTHER_EXP_AUTO,
              KU_COMMUNITY_AUTO,
              PROJECT_PROF_AUTO,
              PREV_PROJ_PROF_AUTO,
              KU_ALL_AUTO,
              XIN_FEAT_AUTO,
              KU_ALL_XIN_ALL_AUTO]

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)


stacked_model_list = [STACK_KU_ALL, STACK_KU_ALL_XIN]
stacked_model_list_auto = [STACK_KU_ALL_AUTO, STACK_KU_ALL_XIN_AUTO]

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


def get_train_test_prob (feature, X_train, X_test, Y_train, classifier_name, is_auto, boot_id):
    X_train = X_train[feature]
    X_test = X_test[feature]
    if is_auto:
        auto_feature = AutoSpearman(X_train = X_train, verbose=False)
        X_train = X_train[auto_feature]
        X_test = X_test[auto_feature]
    
    
    if classifier_name == "RF":
        np.random.seed(boot_id)
        rf_clf = RandomForestClassifier(random_state=boot_id, n_estimators = 100, n_jobs=20).fit(X_train, Y_train)
        rf_preds = rf_clf.predict(X_test)
        rf_predicted_proba_train = rf_clf.predict_proba(X_train)[:,1]
        rf_predicted_proba_test = rf_clf.predict_proba(X_test)[:,1]
        
        return rf_predicted_proba_train, rf_predicted_proba_test
    
    if classifier_name == "LR":
        np.random.seed(boot_id)
        logistic_clf = LogisticRegression(random_state=boot_id).fit(X_train, Y_train)
        logistic_preds = logistic_clf.predict(X_test)
        lr_predicted_proba_train = logistic_clf.predict_proba(X_train)[:,1]
        lr_predicted_proba_test = logistic_clf.predict_proba(X_test)[:,1]
        
        return lr_predicted_proba_train, lr_predicted_proba_test
    
    if classifier_name == "LGBM":
        np.random.seed(boot_id)
        lgbm_clf = LGBMClassifier(objective='binary', random_state=boot_id, n_jobs=20).fit(X_train, Y_train)
        lgbm_preds = lgbm_clf.predict(X_test)
        lgbm_predicted_proba_train = lgbm_clf.predict_proba(X_train)[:,1]
        lgbm_predicted_proba_test = lgbm_clf.predict_proba(X_test)[:,1]
        
        return lgbm_predicted_proba_train, lgbm_predicted_proba_test
    
    return None, None
        
def out_of_bootstrap_running(target_name, feature_list, selected_feature, X, Y,  boot_lim, year_value, file_name, is_auto):
    result = []
    sample_iterations = list(range(1,boot_lim + 1,1))
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]
    for boot_id in sample_iterations:
        boot_index_data= year_value_boot_data[year_value_boot_data['boot_id'] == boot_id]
        start_time = time.time()
        print('{} bootstrap = {}'.format(file_name,boot_id), end = " ")
        np.random.seed(boot_id)
        train_index = boot_index_data[boot_index_data['index_type']=='train_index']['index_value'].tolist()
        test_index = boot_index_data[boot_index_data['index_type']=='test_index']['index_value'].tolist()
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        lr_train, lr_test = {}, {}
        rf_train, rf_test = {}, {}
        lgbm_train, lgbm_test = {}, {}
        
        for feature_name in selected_feature:
            feature = feature_list[feature_name]
            rf_train_prob, rf_test_prob = get_train_test_prob (feature, X_train, X_test, Y_train, "RF", is_auto, boot_id)
            lr_train_prob, lr_test_prob = get_train_test_prob (feature, X_train, X_test, Y_train, "LR", is_auto, boot_id)
            lgbm_train_prob, lgbm_test_prob = get_train_test_prob (feature, X_train, X_test, Y_train, "LGBM", is_auto, boot_id)
            
            rf_train[feature_name] =  (list(rf_train_prob))
            rf_test[feature_name] =  (list(rf_test_prob))
            
            lr_train[feature_name] = (list(lr_train_prob))
            lr_test[feature_name] = (list(lr_test_prob))
            
            lgbm_train[feature_name] = (list(lgbm_train_prob))
            lgbm_test[feature_name] = (list(lgbm_test_prob))
            
        
        pd_rf_train = pd.DataFrame(rf_train)
        pd_lr_train = pd.DataFrame(lr_train)
        pd_lgbm_train = pd.DataFrame(lgbm_train)
        
        pd_rf_test = pd.DataFrame(rf_test)
        pd_lr_test = pd.DataFrame(lr_test)
        pd_lgbm_test = pd.DataFrame(lgbm_test)
        
            
        
        np.random.seed(boot_id)
        rf_clf = RandomForestClassifier(random_state=boot_id, n_estimators = 100, n_jobs=20).fit(pd_rf_train, Y_train)
        rf_preds = rf_clf.predict(pd_rf_test)
        rf_predicted_proba_train = rf_clf.predict_proba(pd_rf_train)[:,1]
        rf_predicted_proba_test = rf_clf.predict_proba(pd_rf_test)[:,1]
        
        np.random.seed(boot_id)
        logistic_clf = LogisticRegression(random_state=boot_id).fit(pd_lr_train, Y_train)
        logistic_preds = logistic_clf.predict(pd_lr_test)
        lr_predicted_proba_train = logistic_clf.predict_proba(pd_lr_train)[:,1]
        lr_predicted_proba_test = logistic_clf.predict_proba(pd_lr_test)[:,1]
        
        np.random.seed(boot_id)
        lgbm_clf = LGBMClassifier(objective='binary', random_state=boot_id, n_jobs=20).fit(pd_lgbm_train, Y_train)
        lgbm_preds = lgbm_clf.predict(pd_lgbm_test)
        lgbm_predicted_proba_train = lgbm_clf.predict_proba(pd_lgbm_train)[:,1]
        lgbm_predicted_proba_test = lgbm_clf.predict_proba(pd_lgbm_test)[:,1]
        
        
        
        
        
        filename_clf = "{}rf_{}_{}_{}.pkl".format(rf_model_output_path,file_name,boot_id,year_value)
        filename_lr = "{}lr_{}_{}_{}.pkl".format(lr_model_output_path,file_name,boot_id,year_value)
        filename_lgbm = "{}lgbm_{}_{}_{}.pkl".format(lgbm_model_output_path,file_name,boot_id,year_value)
        
        #pickle.dump(rf_clf, open(filename_clf, 'wb'))
        #pickle.dump(logistic_clf, open(filename_lr, 'wb'))
        #pickle.dump(lgbm_clf, open(filename_lgbm, 'wb'))
        
        
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds, rf_predicted_proba_test)
        precision_log, recall_log, fscore_log, support_log, far_log, tp_log, fp_log, tn_log, fn_log, roc_value_log = calculate_eval_metrics_II(Y_test['target'],logistic_preds, lr_predicted_proba_test)
        precision_lgbm, recall_lgbm, fscore_lgbm, support_lgbm, far_lgbm, tp_lgbm, fp_lgbm, tn_lgbm, fn_lgbm, roc_value_lgbm = calculate_eval_metrics_II(Y_test['target'],lgbm_preds, lgbm_predicted_proba_test)
        
        result.append([year_value,target_name, file_name,"RandomForest", boot_id, precision, recall, fscore, tp, fp, tn, fn, roc_value, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
        result.append([year_value, target_name, file_name, "LogisticRegression",boot_id, precision_log, recall_log, fscore_log, tp_log, fp_log, tn_log, fn_log, roc_value_log, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
        result.append([year_value, target_name, file_name, "LightGBM",boot_id, precision_lgbm, recall_lgbm, fscore_lgbm, tp_lgbm, fp_lgbm, tn_lgbm, fn_lgbm, roc_value_lgbm, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
        
        print(" Time: {:.4f} seconds".format(time.time() - start_time))
        
    result_pd = pd.DataFrame(result)
    result_pd.columns = ['ltc_year','ltc_approach','features','classifier','bootstrap','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    
    return result_pd

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


def bootstrap_index_generation():
    output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"
    year_value_list = [1,2,3]
    pd_list = []
    for year_value in year_value_list:
        path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, year_value)
        X = pd.read_csv(path)
        indices = list(range(0,X.shape[0] ,1))
        sample_iterations = list(range(1,100 + 1,1))
        print("Working LTC Year {}".format(year_value))
        
        for i in sample_iterations:
            start_time = time.time()
            np.random.seed(i)
            train_index = list(np.random.choice(indices,len(indices)))
            test_index = np.setdiff1d(indices, train_index)
            
            d_train = pd.DataFrame(train_index, columns=['index_value'])
            d_train['boot_id'] = i
            d_train['ltc_year'] = year_value
            d_train['index_type'] = 'train_index'
            
            d_test = pd.DataFrame(test_index, columns=['index_value'])
            d_test['boot_id'] = i
            d_test['ltc_year'] = year_value
            d_test['index_type'] = 'test_index'
            
            d_merge = pd.concat([d_train, d_test], axis = 0)
            pd_list.append(d_merge)
            
        print("Finished")
        
        
    pd_final_combine_index = pd.concat(pd_list, axis = 0)
    pd_final_combine_index.to_csv("{}bootstrap_index.csv".format(output_path), index=False)
    
def bootstrap_model_run(year_value,target_name):
    path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, year_value)
    pd_full = pd.read_csv(path)
    
    col_list = list(pd_full.columns)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]
    
    ku_proj_dim_count_col = ["proj_dim_" + x for x in ku_features_count]
    ku_prev_exp_count_col = ["prev_exp_" + x for x in ku_features_count]
    ku_comm_exp_count_col = ["comm_exp_" + x for x in ku_features_count]
    ku_other_proj_dim_count_col =["other_proj_dim_" + x for x in ku_features_count]
    
    all_ku_count_feature = ku_features_count + ku_proj_dim_count_col + ku_prev_exp_count_col + ku_comm_exp_count_col + ku_other_proj_dim_count_col
    all_ku_all_xin = all_ku_count_feature + xin_features
    
    path_proj_dim = "{}proj-ku-dim-ltc-{}.csv".format(project_ku_profile_dir, year_value)
    path_prev_dev_exp = "{}other_project_exp_ku_count_norm_label_ltc{}.csv".format(dev_other_project_experience_ku_profile_dir, year_value)
    path_comm_exp = "{}community_ku_count_norm_label_ltc{}.csv".format(community_ku_profile_dir,year_value)
    prev_proj_dim = "{}other-proj-ku-dim-ltc-{}.csv".format(other_project_ku_profile_dir, year_value)

    pd_proj_dim         = pd.read_csv(path_proj_dim)
    pd_prev_dev_exp     = pd.read_csv(path_prev_dev_exp)
    pd_comm_exp         = pd.read_csv(path_comm_exp)
    pd_prev_proj_dim    = pd.read_csv(prev_proj_dim)


    pd_proj_dim_selected = pd_proj_dim[col_list[0:3] + ku_features_count]
    pd_proj_dim_selected.columns = col_list[0:3] + ku_proj_dim_count_col
    
    pd_prev_exp_selected = pd_prev_dev_exp[col_list[0:3] + ku_features_count]
    pd_prev_exp_selected.columns = col_list[0:3] + ku_prev_exp_count_col
    
    pd_comm_exp_selected = pd_comm_exp[col_list[0:3] + ku_features_count]
    pd_comm_exp_selected.columns = col_list[0:3] + ku_comm_exp_count_col
    
    pd_prev_proj_selected = pd_prev_proj_dim[col_list[0:3] + ku_features_count]
    pd_prev_proj_selected.columns = col_list[0:3] + ku_other_proj_dim_count_col
    
    pd_full_merged = pd.concat([pd_full[col_list[0:3] + ku_features_count + xin_features].reset_index(drop=True),
                                    pd_proj_dim_selected[ku_proj_dim_count_col].reset_index(drop=True),
                                    pd_prev_exp_selected[ku_prev_exp_count_col].reset_index(drop=True),
                                    pd_comm_exp_selected[ku_comm_exp_count_col].reset_index(drop=True),
                                    pd_prev_proj_selected[ku_other_proj_dim_count_col].reset_index(drop=True),
                                    ], axis = 1)
    
    Y = get_target_label(year_value, pd_full, target_name)
    

    output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/stack-model/"
    
    feature_list = {KU_DEV_EXP      : ku_features_count,
                    KU_OTHER_EXP    : ku_prev_exp_count_col,
                    KU_COMMUNITY    : ku_comm_exp_count_col,
                    PROJECT_PROF    : ku_proj_dim_count_col,
                    PREV_PROJ_PROF  : ku_other_proj_dim_count_col,
                    KU_ALL          : all_ku_count_feature,
                    XIN_FEAT        : xin_features,
                    KU_ALL_XIN_ALL  : all_ku_all_xin}
    
    
    selected_feature = {STACK_KU_ALL : [KU_DEV_EXP, KU_OTHER_EXP, KU_COMMUNITY, PROJECT_PROF, PREV_PROJ_PROF],
                        STACK_KU_ALL_XIN: [KU_DEV_EXP, KU_OTHER_EXP, KU_COMMUNITY, PROJECT_PROF, PREV_PROJ_PROF, XIN_FEAT]}
    
    
    for i in range(len(stacked_model_list)):
        #print(i)
        model = stacked_model_list[i]
        X = pd_full_merged
        result_ku = out_of_bootstrap_running('Ahsan_Java_Commit', feature_list, selected_feature [model], X, Y, 50,  year_value, model, False)
        result_ku.to_csv('{}ltc_{}_{}_stack.csv'.format(output_path, year_value, model))
    
    '''
    for i in range(len(stacked_model_list_auto)):
        #print(i)
        model = stacked_model_list_auto[i]
        stack_model = stacked_model_list[i]
        X = pd_full_merged
        result_ku = out_of_bootstrap_running('Ahsan_Java_Commit', feature_list, selected_feature[stack_model], X, Y, 50,  year_value, model, True)
        result_ku.to_csv('{}ltc_{}_{}_stack_auto.csv'.format(output_path, year_value, stacked_model_list_auto[i]))
    '''


    
def bootstrap_analysis():
    '''
    start_time = time.time()
    bootstrap_model_run(1,t_name[0])
    print(" YEAR 1 --- {:.4f} seconds ---".format(time.time() - start_time))
    '''
    
    start_time = time.time()
    bootstrap_model_run(2,t_name[0])
    print(" YEAR 2 --- {:.4f} seconds ---".format(time.time() - start_time))
    
    
    start_time = time.time()
    bootstrap_model_run(3,t_name[0])
    print(" YEAR 3 --- {:.4f} seconds ---".format(time.time() - start_time))
    

if __name__ == "__main__":
    bootstrap_analysis()
