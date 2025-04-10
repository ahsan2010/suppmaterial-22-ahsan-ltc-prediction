import pandas as pd
import os.path
import numpy as np
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
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import time

def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x

# Models other than the stacked models
def k_fold_stratified_windows_based_testing(target_name, feature_name, X, Y, year_value = 1, file_name = 'result'):
    seed_value = 123
    nfold = 10
    result = []
    print("{} = {}".format(year_value, file_name))
    x = X
    y = Y
    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
    skf = StratifiedKFold(n_splits=nfold, shuffle=False)
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
        result.append([year_value, target_name, feature_name, fold, precision, recall, fscore, tp, fp, tn, fn, roc_value, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])
    result_pd = pd.DataFrame(result)
    result_pd.columns = ['ltc_year','ltc_approach','features','fold','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    #result_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/results/'
    
    #result_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/'
    #result_pd.to_csv("{}{}_{}_years.csv".format(result_path, file_name, year_value), index = False) 
    return result_pd

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
# Analysis all models other than the Stacked Models
def analysis_model(year_value):
    #path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/full_result_ltc_{}.csv'.format(year_value)
    path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_result_ltc_{}.csv'.format(year_value)
    
    pd_full = pd.read_csv(path)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]

    ahsan_ltc_1_col = pd_full.iloc[: , 32:33]
    ahsan_ltc_2_col = pd_full.iloc[: , 33:34]
    ahsan_ltc_3_col = pd_full.iloc[: , 34:35]

    #xin_ltc_col = pd_full.iloc[ : , 129:130]
    
    Y = None
    if year_value == 1:
        Y = pd_full[['LTC_Developer_Cat_Year_One']]
    elif year_value == 2:
        Y = pd_full[['LTC_Developer_Cat_Year_Two']]
    elif year_value == 3:
        Y = pd_full[['LTC_Developer_Cat_Year_Three']]
    
    # only KU daata
    X = pd_full[ku_features_count]
    pd_list.append(k_fold_stratified_windows_based_testing('Ahsan_Java_Commit','KU_COUNT', X, Y, year_value = year_value, file_name = 'ahsan_target_ku_features_count'))


    X = pd_full[ku_features_norm]
    pd_list.append(k_fold_stratified_windows_based_testing('Ahsan_Java_Commit','KU_NORM', X, Y, year_value = year_value, file_name = 'ahsan_target_ku_features_norm'))

    X = pd_full[ku_full_features]
    pd_list.append(k_fold_stratified_windows_based_testing('Ahsan_Java_Commit', 'KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ahsan_target_ku_feature_count_norm'))

    
    X = pd_full[xin_features]
    pd_list.append(k_fold_stratified_windows_based_testing('Ahsan_Java_Commit', 'XIN_FEATURES', X, Y, year_value = year_value, file_name = 'ahsan_target_xin_feature'))
    
    X = pd_full[xin_features + ku_full_features]
    pd_list.append(k_fold_stratified_windows_based_testing('Ahsan_Java_Commit', 'XIN_FEATURES_KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ahsan_target_xin_feature_ku_norm_count'))
    
    
    X = pd_full[ku_features_count]
    Y1_Xin = pd_full[['is_ltc']]
    pd_list.append(k_fold_stratified_windows_based_testing('Xin_Any_Commit','KU_COUNT', X, Y1_Xin, year_value = year_value, file_name = 'xin_target_ku_features_count'))

    X = pd_full[ku_features_norm]
    Y1_Xin = pd_full[['is_ltc']]
    pd_list.append(k_fold_stratified_windows_based_testing('Xin_Any_Commit', 'KU_NORM', X, Y1_Xin, year_value = year_value, file_name = 'xin_target_ku_features_norm'))


    X = pd_full[ku_full_features]
    Y1_Xin = pd_full[['is_ltc']]
    pd_list.append(k_fold_stratified_windows_based_testing('Xin_Any_Commit','KU_COUNT_NORM', X, Y1_Xin, year_value = year_value, file_name = 'xin_target_ku_feature_count_norm'))


    # only xin data
    X = pd_full[xin_features]
    Y1_Xin = pd_full[['is_ltc']]
    pd_list.append(k_fold_stratified_windows_based_testing('Xin_Any_Commit','XIN_FEATURES', X, Y1_Xin, year_value = year_value, file_name = 'xin_features_xin_target'))
    

    # ku + xin data
    # only xin data
    X = pd_full[xin_features + ku_full_features]
    Y1_Xin = pd_full[['is_ltc']]
    pd_list.append(k_fold_stratified_windows_based_testing('Xin_Any_Commit','XIN_FEATURES_KU_COUNT_NORM', X, Y1_Xin, year_value = year_value, file_name = 'xin_features_ku_count_norm_features_xin_target'))

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']
# Generate and analysis of the stacked models.
def stack_model_implement(year_value, target_name):
    
    path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_result_ltc_{}.csv'.format(year_value)
    
    pd_full = pd.read_csv(path)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]

    ahsan_ltc_1_col = pd_full.iloc[: , 32:33]
    ahsan_ltc_2_col = pd_full.iloc[: , 33:34]
    ahsan_ltc_3_col = pd_full.iloc[: , 34:35]

    xin_ltc_col = pd_full.iloc[ : , 129:130]
    
    Y = None
    if year_value == 1:
        Y = pd_full[['LTC_Developer_Cat_Year_One']]
    elif year_value == 2:
        Y = pd_full[['LTC_Developer_Cat_Year_Two']]
    elif year_value == 3:
        Y = pd_full[['LTC_Developer_Cat_Year_Three']]
    

    if target_name == t_name[1]:
        Y = pd_full[['is_ltc']]

    
    X_Xin = pd_full[xin_features] #62 features
    X_KU = pd_full[ku_full_features] #58 features
    seed_value = 123
    nfold = 10
    result = []
    print("{} = {}".format(year_value, path))
    x = X_KU
    y = Y
    skf = StratifiedKFold(n_splits=nfold, shuffle=False)
    fold = 0
    #train_index, test_index = next(iter(skf.split(x,y)))
    for train_index, test_index in skf.split(x, y):
        print("Fold {}".format(fold))
        print('Train {} Test {}'.format(len(train_index), len(test_index)))
        fold = fold + 1
        
        X_train, X_test = X_KU.iloc[train_index], X_KU.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        
        total_train_inst = len(Y_train)
        total_test_inst = len(Y_test)
        train_positive = Y_train.target.sum()
        train_negative = len(Y_train) - train_positive
        test_positive = Y_test.target.sum()
        test_negative = len(Y_test) - test_positive
        
        
        # First model with KU features for the stacked models
        rf_clf_ku = RandomForestClassifier(random_state=seed_value, n_estimators = 100).fit(X_train, Y_train)
        rf_preds_ku = rf_clf_ku.predict(X_train)
        predicted_prob_ku = rf_clf_ku.predict_proba(X_train)[:,1]
        rf_preds_ku_test = rf_clf_ku.predict(X_test)
        predicted_prob_ku_test = rf_clf_ku.predict_proba(X_test)[:,1]
        
        X_train_Xin, X_test_Xin = X_Xin.iloc[train_index], X_Xin.iloc[test_index]
        
        rf_clf_xin = RandomForestClassifier(random_state=seed_value, n_estimators = 100).fit(X_train_Xin, Y_train)
        rf_preds_xin = rf_clf_xin.predict(X_train_Xin)
        predicted_prob_xin = rf_clf_xin.predict_proba(X_train_Xin)[:,1]
        rf_preds_xin_test = rf_clf_xin.predict(X_test_Xin)
        predicted_prob_xin_test = rf_clf_xin.predict_proba(X_test_Xin)[:,1]
        precision_xin, recall_xin, fscore_xin, support_xin, far_xin, tp_xin, fp_xin, tn_xin, fn_xin, auc_xin = calculate_eval_metrics_II(Y_test['target'],rf_preds_xin_test, predicted_prob_xin_test)
        print('Test Xin feature AUC {}'.format(auc_xin))
        
        precision, recall, fscore, support, far, tp, fp, tn, fn, auc = calculate_eval_metrics_II(Y_train['target'],rf_preds_ku, predicted_prob_ku)
        print('Train KU AUC {}'.format(auc))
        
        precision_full, recall_full, fscore_full, support_full, far_full, tp_full, fp_full, tn_full, fn_full, auc_full = calculate_eval_metrics_II(Y_train['target'],rf_preds_xin, predicted_prob_xin)
        print('Train Xin features AUC {}'.format(auc_full))
        
        precision_full, recall_full, fscore_full, support_full, far_full, tp_full, fp_full, tn_full, fn_full, auc_full = calculate_eval_metrics_II(Y_test['target'],rf_preds_ku_test, predicted_prob_ku_test)
        print('Test KU features AUC {}'.format(auc_full))
        
        #Generate train dataset for the stacked model (Xin pred + ku_pred)
        pd_stacked_train = pd.DataFrame(zip(predicted_prob_ku, predicted_prob_xin))
        pd_stacked_train.columns = ['ku_pred', 'xin_pred']

        #Generate test dataset for the stacked model (Xin pred + ku_pred)
        pd_stacked_test = pd.DataFrame(zip(predicted_prob_ku_test, predicted_prob_xin_test))
        pd_stacked_test.columns = ['ku_pred', 'xin_pred']
        
        # Traain and test the stacked models
        rf_clf_stacked = RandomForestClassifier(random_state=seed_value, n_estimators = 100).fit(pd_stacked_train, Y_train)
        rf_preds_stacked = rf_clf_stacked.predict(pd_stacked_test)
        predicted_prob_stacked = rf_clf_stacked.predict_proba(pd_stacked_test)[:,1]
        precision_st, recall_st, fscore_st, support_st, far_st, tp_st, fp_st, tn_st, fn_st, auc_st = calculate_eval_metrics_II(Y_test['target'],rf_preds_stacked, predicted_prob_stacked)
        print('Test Stacked KU and Xin pred AUC {}'.format(auc_st))

        result.append([year_value, target_name, 'STACKED_XIN_PRED_PROB_KU_PRED_PROB', fold, precision_st, recall_st, fscore_st, tp_st, fp_st, tn_st, fn_st, auc_st, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])

        # Gustavo! This is the stacked models we discussed 
        #Generate train dataset for the stacked model (Xin feature + ku_pred)
        pd_stacked_xin_feature_ku_pred_train = pd.DataFrame(list(X_train_Xin.values))
        pd_stacked_xin_feature_ku_pred_train.columns = list(X_train_Xin)
        pd_stacked_xin_feature_ku_pred_train ['ku_pred'] = pd_stacked_train['ku_pred']
        
        #Generate test dataset for the stacked model (Xin feature + ku_pred)
        pd_stacked_xin_feature_ku_pred_test = pd.DataFrame(list(X_test_Xin.values))
        pd_stacked_xin_feature_ku_pred_test.columns = list(X_test_Xin)
        pd_stacked_xin_feature_ku_pred_test['ku_pred'] = pd_stacked_test['ku_pred']
        
        rf_clf_xin_feature_ku_pred_stacked = RandomForestClassifier(random_state=seed_value, n_estimators = 100).fit(pd_stacked_xin_feature_ku_pred_train, Y_train)
        rf_preds_xin_feature_ku_pred_stacked = rf_clf_xin_feature_ku_pred_stacked.predict(pd_stacked_xin_feature_ku_pred_test)
        predicted_prob_xin_feature_ku_pred_stacked = rf_clf_xin_feature_ku_pred_stacked.predict_proba(pd_stacked_xin_feature_ku_pred_test)[:,1]
        
        precision_xin_st, recall_xin_st, fscore_xin_st, support_xin_st, far_xin_st, tp_xin_st, fp_xin_st, tn_xin_st, fn_xin_st, auc_xin_st = calculate_eval_metrics_II(Y_test['target'],rf_preds_xin_feature_ku_pred_stacked, predicted_prob_xin_feature_ku_pred_stacked)
        print('Test Xin Feature + KU pred AUC {}'.format(auc_xin_st))
        result.append([year_value, target_name, 'STACKED_XIN_FEATURE_KU_PRED_PROB', fold, precision_xin_st, recall_xin_st, fscore_xin_st, tp_xin_st, fp_xin_st, tn_xin_st, fn_xin_st, auc_xin_st, total_train_inst, total_test_inst, train_positive, train_negative, test_positive, test_negative])

    result_pd = pd.DataFrame(result)
    result_pd.columns = ['ltc_year','ltc_approach','features','fold','precision','recall', 'fscore', 'true_positive','false_positive', 'true_negative', 'false_negative', 'auc', 'total_train_inst', 'total_test_inst', 'train_LTC', 'train_NON_LTC', 'test_LTC', 'test_NON_LTC']
    return result_pd




def out_of_bootstrap_running(target_name, feature_name, X, Y, year_value = 1, file_name = 'bootstrap_result'):
    import numpy as np
    
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
    #path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_result_ltc_{}.csv'.format(year_value)
    #path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/full_result_updated_label_ltc{}.csv'.format(year_value)
    path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/full_result_updated_label_ltc{}.csv'.format(year_value)
    
    
    pd_full = pd.read_csv(path)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]

    ahsan_ltc_1_col = pd_full.iloc[: , 32:33]
    ahsan_ltc_2_col = pd_full.iloc[: , 33:34]
    ahsan_ltc_3_col = pd_full.iloc[: , 34:35]

    xin_ltc_col = pd_full.iloc[ : , 129:130]
    
    Y = None
    if year_value == 1:
        Y = pd_full[['LTC_Developer_Cat_Year_One']]
    elif year_value == 2:
        Y = pd_full[['LTC_Developer_Cat_Year_Two']]
    elif year_value == 3:
        Y = pd_full[['LTC_Developer_Cat_Year_Three']]
    

    if target_name == t_name[1]:
        Y = pd_full[['is_ltc']]

    xin_top_feature = [
            'month_user_commits',
            'user_history_followers',
            'before_repo_watchers',
            'user_age',
            'before_repo_contributor_mean'
            ]
    
    #output_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/Result/'
    output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/Result/"
    
    X = pd_full[xin_features]
    result_xin_features = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES', X, Y, year_value = year_value, file_name = 'xin_features_boostrap')
    result_xin_features.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"xin_features_boostrap.csv"))

    X = pd_full[xin_features + ku_full_features]
    result_xin_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES_KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ku_full_xin_features_boostrap')
    result_xin_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_full_xin_features_boostrap.csv"))

    X = pd_full[ku_full_features]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_full_boostrap.csv"))

    X = pd_full[xin_top_feature]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_SIMPLIFIED', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"xin_simp_boostrap.csv"))

    X = pd_full[xin_top_feature + ku_features_count]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_SIMPLIFIED_KU_COUNT', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_count_xin_simp_boostrap.csv"))

    X = pd_full[xin_top_feature + ku_features_norm]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_SIMPLIFIED_KU_NORM', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_norm_xin_simp_boostrap.csv"))

    X = pd_full[xin_top_feature + ku_full_features]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_SIMPLIFIED_KU_COUNT_NORM', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_full_xin_simp_boostrap.csv"))

    X = pd_full[ku_features_count]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','KU_COUNT', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_count_boostrap.csv"))

    X = pd_full[ku_features_norm]
    result_ku = out_of_bootstrap_running('Ahsan_Java_Commit','KU_NORM', X, Y, year_value = year_value, file_name = 'ku_full_boostrap')
    result_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_norm_boostrap.csv"))
    
    X = pd_full[xin_features + ku_features_count]
    result_xin_features = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES_KU_COUNT', X, Y, year_value = year_value, file_name = 'xin_features_boostrap')
    result_xin_features.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_count_xin_features_boostrap.csv"))

    X = pd_full[xin_features + ku_features_norm]
    result_xin_ku = out_of_bootstrap_running('Ahsan_Java_Commit','XIN_FEATURES_KU_NORM', X, Y, year_value = year_value, file_name = 'ku_full_xin_features_boostrap')
    result_xin_ku.to_csv('{}ltc_{}_{}'.format(output_path,year_value,"ku_norm_xin_features_boostrap.csv"))

def prev_model_analysis():
    #mapped_developer_project_full_data(1)
    analysis_model(1)
    analysis_model(2)
    analysis_model(3)
    #Analysis of the stacked models.
    for target_name in t_name:
        pd_list.append(stack_model_implement(1, target_name))
        pd_list.append(stack_model_implement(2, target_name))
        pd_list.append(stack_model_implement(3, target_name))
    pd_result_full = pd.concat(pd_list)
    pd_result_full.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_results_ltc_different_models_no_shuffle_update_Aug_04_2022.csv' ,index = False)
    
def get_top_ranked_features(feature_imp_values,feature_names):
    
    feature_list = []
    for x in feature_names:
        imp_value = np.median(feature_imp_values[x])
        #print(imp_value)
        feature_list.append([x,imp_value])
    pd_top_feature = pd.DataFrame(feature_list)
    pd_top_feature.columns = ['feature','imp_value']
    pd_top_feature=pd_top_feature.sort_values('imp_value',ascending=False)
    
    return pd_top_feature
    

def find_feature_imp_xin_model(year_value = 1):
    #year_value = 1
    path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/data_year_{}.csv'.format(year_value)
    pd_full = pd.read_csv(path)
    pd_full = pd_full[pd_full['language'] != 'Java']
    xin_features = list(pd_full.columns)[3:65]
    X = pd_full[xin_features]
    Y = pd_full[['is_ltc']]
    
    indices = list(range(0,X.shape[0] ,1))
    sample_iterations = list(range(1,10 + 1,1))
    
    feature_names = xin_features
    feature_imp_values = {}
    for x in feature_names:
        feature_imp_values[x] = []
    
    for i in sample_iterations:
        print('bootstrap = {}'.format(i))
        np.random.seed(i)
        train_index = list(np.random.choice(indices,len(indices)))
        test_index = np.setdiff1d(indices, train_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        rf_clf = RandomForestClassifier(random_state=i, n_estimators = 100).fit(X_train, Y_train)
        rf_preds = rf_clf.predict(X_test)
        predicted_proba = rf_clf.predict_proba(X_test)[:,1]
        Y_test.columns = ['target']
        Y_train.columns = ['target']
        precision, recall, fscore, support, far, tp, fp, tn, fn, roc_value = calculate_eval_metrics_II(Y_test['target'],rf_preds, predicted_proba)
        
        importances = rf_clf.feature_importances_
        for i in range(len(feature_names)):
            feature_name = feature_names[i]
            feature_imp_values[feature_name].append(importances[i])
        
    ranked_feature = get_top_ranked_features(feature_imp_values,feature_names)
    ranked_feature.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/feature_imp_xin/feature_imp_{}.csv'.format(year_value),index = False)

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

def xin_feature_analysis():
    find_feature_imp_xin_model(2)
    find_feature_imp_xin_model(3)

if __name__ == "__main__":
    #xin_feature_analysis()
    #bootstrap_analysis()
    #bootstrap_model_run(2)
    #bootstrap_model_run(3)