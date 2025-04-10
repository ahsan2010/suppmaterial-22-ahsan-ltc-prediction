#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:26:04 2024

@author: ahsan
"""
import shap
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

from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import get_context

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


save_model_dir              = "/home/local/SAIL/ahsan/LTC_Project/save-model-dec-15/" 
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

sk_rank_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/sk-input-data/"

shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/single_model_whole_data_trained/"
#shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/ku-all-dim-all-year/"
#shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
#shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/ku-all-dim-all-year/"
bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']

path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, 1)
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


def get_feature_list():
    path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, 1)
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
    
    feature_list = [ku_features_count,
                    ku_prev_exp_count_col,
                    ku_comm_exp_count_col,
                    ku_proj_dim_count_col,
                    ku_other_proj_dim_count_col,
                    all_ku_count_feature,
                    xin_features,
                    all_ku_all_xin]
    
    return feature_list

def get_data(year_value,target_name):
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
    
    return pd_full_merged, Y, all_ku_count_feature, all_ku_all_xin, ku_prev_exp_count_col, ku_other_proj_dim_count_col


def child_thread_shap_analysis(file_name, year_value):
    
    X, Y,all_ku_count_feature, all_ku_all_xin, _, _ = get_data(year_value, t_name[0])
    
    if file_name == KU_ALL_AUTO:
        features = all_ku_count_feature
    elif file_name == KU_ALL_XIN_ALL_AUTO:
        features = all_ku_all_xin
    
    Y.columns = ['target']
    total_inst = len(Y)
    total_test = len(Y)
    positive = Y.target.sum()
    negative = len(Y) - positive
    
    start_time = time.time()
   
    X = X[features]
    auto_feature = AutoSpearman(X_train = X, verbose=False)
    X = X[auto_feature]
    np.random.seed(1)
    rf_clf = RandomForestClassifier(random_state=1, n_estimators = 100, n_jobs=20).fit(X, Y)
    rf_preds = rf_clf.predict(X)
    predicted_proba = rf_clf.predict_proba(X)[:,1]
    rf_clf.feature_names = list(X.columns.values)
          
    model_output_file = "{}{}_model_{}.pkl".format(shap_value_output_dir,file_name,year_value)
    pickle.dump(rf_clf, open(model_output_file, 'wb'))
    
    explainer = shap.Explainer(rf_clf.predict, X, seed = 1)
    shap_values = explainer(X)
    shap_values.feature_names = rf_clf.feature_names
    
    shap_output_file = "{}{}_shap_{}.pkl".format(shap_value_output_dir,file_name,year_value)
    pickle.dump(shap_values, open(shap_output_file, 'wb'))
    
    print("Complete [{}] Year [{}] Time: {:.4f} seconds".format(file_name, year_value,time.time() - start_time))

def multi_run_wrapper(args):
    file_name = args[0]
    year_value = args[1]
    child_thread_shap_analysis(file_name, year_value)
    
def parallel_shap_feature_importance(year_value_list, file_name):
    
    start_time = time.time()

    with get_context("spawn").Pool(10) as p:
        p.map(multi_run_wrapper,[[file_name, year_value] for year_value in year_value_list])
        p.close()
        p.join()
        
    print("Done All [{}] Time: {:.4f} seconds".format(file_name,time.time() - start_time))

def get_feature_name_list(boot_lim,shap_value_output_dir_ku,file_name,year_value):
    feature_list = []
    for boot_id in range(1, boot_lim + 1):
        print("Working boot_id [{}]".format(boot_id))
        shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_output_dir_ku,file_name,year_value,boot_id)
        load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        feature_list.extend(load_shap_value.feature_names)
    feature_list = list(set(feature_list))
    return feature_list

def get_KU_feature_name(f):
    if f in ku_features_count:
        return ("KU-CUR-PROJ-EXP:" + f)
    elif f in ku_proj_dim_count_col:
        ss = f[len("proj_dim_"):]
        return ("KU-CUR-PROJ-PROF:"+ss)
    elif f in ku_prev_exp_count_col:
        ss = f[len("prev_exp_"):]
        return ("KU-PREV-PROJ-EXP:"+ss)
    elif f in ku_comm_exp_count_col:
        ss = f[len("comm_exp_"):]
        return ("KU-CUR-PROJ-COLLAB-EXP:"+ss)
    elif f in ku_other_proj_dim_count_col:
        ss = f[len("other_proj_dim_"):]
        return ("KU-PREV-PROJ-PROF:"+ss)
   
    return (f)

def generate_single_boot_feature_ranking(load_shap_value, feature_list):
    feature_shap_rank = {}
    for f in feature_list:
        feature_shap_rank[f] = []
    for row_ind in range(len(load_shap_value.values)):
        feature_name = load_shap_value.feature_names
        feature_data = np.abs(load_shap_value.values[row_ind, :])
        pd_feature = pd.DataFrame({'feature_name':feature_name,'shap_value': feature_data})
        pd_feature = pd_feature.sort_values('shap_value', ascending = False)
        pd_feature['rank'] = range(1,len(load_shap_value.feature_names) + 1)
        
        for d in zip(pd_feature['feature_name'].values, pd_feature['rank'].values):
            feature_shap_rank[d[0]].append(d[1])
    
    return feature_shap_rank

def generate_sk_reank_input_data_per_year(file_name, year_value):
    
    sk_rank_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/sk-rank-data/'
    feature_shap_all_boot = {}
    
    shap_model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/single_model_whole_data_trained/'
    model_output_file = "{}{}_model_{}.pkl".format(shap_model_path,file_name,year_value)
    loaded_model = pickle.load(open(model_output_file, 'rb'))
    
    feature_list = loaded_model.feature_names
    
    shap_output_file = "{}{}_shap_{}.pkl".format(shap_value_output_dir,file_name,year_value)
    load_shap_value = pickle.load(open(shap_output_file, 'rb'))
            
    for f in feature_list:
        feature_shap_all_boot[f] = []
    
    feature_shap_rank = generate_single_boot_feature_ranking(load_shap_value, feature_list)
    sk_output_file = "{}{}_sk_input_year_{}.txt".format(sk_rank_dir,file_name,year_value)
    with open(sk_output_file, 'w') as fp:
        for f in feature_list:
            rank_string = ' '.join([str(x) for x in feature_shap_rank[f]])
            fp.write(get_KU_feature_name(f))
            fp.write(' ')
            fp.write(rank_string)
            fp.write('\n')

    print("[Finish] file = {} year = {}".format(file_name, year_value))


def generate_sk_rank_input_data():
    file_name_list = [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO]
    year_list = [1,2,3]
    
    for file_name in file_name_list:
        for year_value in year_list:
            generate_sk_reank_input_data_per_year(file_name, year_value)

    
def shap_figure_generate_prev_no_prev_experience():
    
    file_name_list = [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO]
    year_list = [1,2,3]
    
    
    for file_name in file_name_list:
        for year_value in year_list:
            X, Y,all_ku_count_feature, all_ku_all_xin, ku_prev_exp, ku_prev_proj = get_data(year_value, t_name[0])
            
            X['prev_exp'] = X[ku_prev_exp].sum(axis = 1)
            X['ku_prev_proj'] = X[ku_prev_proj].sum(axis = 1)
            
            X['prev_exp-ku_proj_proj'] = (X['prev_exp'] > 0) | (X['ku_prev_proj'] > 0)
            X['no_prev_exp'] = ~((X['prev_exp'] > 0) | (X['ku_prev_proj'] > 0))
        
        
            shap_model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/single_model_whole_data_trained/'
            model_output_file = "{}{}_model_{}.pkl".format(shap_model_path,file_name,year_value)
            loaded_model = pickle.load(open(model_output_file, 'rb'))
        
            feature = loaded_model.feature_names
            feature_name_list = []
            for f in feature:
                feature_name_list.append(get_KU_feature_name(f))
            
            shap_output_file = "{}{}_shap_{}.pkl".format(shap_value_output_dir,file_name,year_value)
            load_shap_value = pickle.load(open(shap_output_file, 'rb'))
            
            t = list(X['prev_exp-ku_proj_proj'])
            tt = list(X['no_prev_exp'])
            X_with_exp = np.where(t)[0]
            X_without_exp = np.where(tt)[0]
            
            data_exp = X[X['prev_exp-ku_proj_proj']==True]
            X_exp = data_exp[loaded_model.feature_names]
            exp_shap =  load_shap_value[X_with_exp]
            fig_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/figures/'
            #figure_file_name = "{}{}_{}_single_whole_with_experience.pdf".format(fig_path,file_name, year_value)
            #shap.summary_plot(exp_shap, X_exp, feature_names = feature_name_list, plot_size =(8,10),show=False)
            
            figure_file_name = "{}{}_{}_single_whole_with_experience_all_feature.pdf".format(fig_path,file_name, year_value)
            shap.summary_plot(exp_shap, X_exp, feature_names = feature_name_list, plot_size =(10,14),show=False, max_display = 30)

            plt.savefig(fname = figure_file_name)
            plt.clf() 
            
            data_no_exp = X[X['no_prev_exp']==True]
            X_no_exp = data_no_exp[loaded_model.feature_names]
            no_exp_shap =  load_shap_value[X_without_exp]
            fig_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/figures/'
            #figure_file_name = "{}{}_{}_single_whole_without_experience.pdf".format(fig_path,file_name, year_value)
            #shap.summary_plot(no_exp_shap, X_no_exp, feature_names = feature_name_list, plot_size =(8,10),show=False)
            
            figure_file_name = "{}{}_{}_single_whole_without_experience_all_feature.pdf".format(fig_path,file_name, year_value)
            shap.summary_plot(no_exp_shap, X_no_exp, feature_names = feature_name_list, plot_size =(10,14),show=False, max_display = 30)

            plt.savefig(fname = figure_file_name)
            plt.clf() 
            
            
            
def shap_figure_generate():
    
    file_name_list = [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO]
    year_list = [1,2,3]
    
    for file_name in file_name_list:
        for year_value in year_list:
            X, Y,all_ku_count_feature, all_ku_all_xin,_,_ = get_data(year_value, t_name[0])
            
            shap_model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/single_model_whole_data_trained/'
            model_output_file = "{}{}_model_{}.pkl".format(shap_model_path,file_name,year_value)
            loaded_model = pickle.load(open(model_output_file, 'rb'))
        
            feature = loaded_model.feature_names
            feature_name_list = []
            for f in feature:
                feature_name_list.append(get_KU_feature_name(f))
            
            shap_output_file = "{}{}_shap_{}.pkl".format(shap_value_output_dir,file_name,year_value)
            load_shap_value = pickle.load(open(shap_output_file, 'rb'))
                
            X = X[loaded_model.feature_names]
            fig_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/figures/'
            figure_file_name = "{}{}_{}_single_whole_all_feature.pdf".format(fig_path,file_name, year_value)
            #shap.summary_plot(load_shap_value, X, feature_names = feature_name_list, plot_size =(8,10),show=False, max_display = len(feature_name_list))
            shap.summary_plot(load_shap_value, X, feature_names = feature_name_list, plot_size =(10,14),show=False, max_display = 30)

            plt.savefig(fname = figure_file_name)
            plt.clf() 

def shap_analysis():
    year_value_list = [1, 2, 3]
    parallel_shap_feature_importance (year_value_list, KU_ALL_AUTO)
    #parallel_shap_feature_importance (year_value_list, KU_ALL_XIN_ALL_AUTO)
    


def manual_study_devleoper_enhance_skills():
    year_value = 3

    path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, 1)
    pd_full = pd.read_csv(path)
    
    
    X, Y,all_ku_count_feature, all_ku_all_xin,_,_ = get_data(1, t_name[0])
    
    
    Y1 = get_target_label(1, pd_full, t_name[0])
    Y2 = get_target_label(2, pd_full, t_name[0])
    Y3 = get_target_label(3, pd_full, t_name[0])
    
    

    dev_prev_localization = X[X['prev_exp_localization'] > 0]    
    dev_prev_localization ['Y1'] = Y1.iloc[list(dev_prev_localization.index)]
    dev_prev_localization ['Y2'] = Y2.iloc[list(dev_prev_localization.index)]
    dev_prev_localization ['Y3'] = Y3.iloc[list(dev_prev_localization.index)]
    
    dev_prev_localization[dev_prev_localization['Y1']==1].shape
    dev_prev_localization[dev_prev_localization['Y2']==1].shape
    dev_prev_localization[dev_prev_localization['Y3']==1].shape
    dev_prev_localization[dev_prev_localization['Y']==1].shape
    
    dev_prev_localization[dev_prev_localization['proj_dim_localization']>0].shape
    dev_prev_localization[dev_prev_localization['localization']>0].shape
    dev_prev_localization[(dev_prev_localization['localization']>0) & (dev_prev_localization['Y3']==1)].shape

def main():
    #shap_analysis()
    #shap_figure_generate()
    #generate_sk_rank_input_data()
    shap_figure_generate_prev_no_prev_experience()
    print('Program finishes successfully')

if __name__ == "__main__":
    main()