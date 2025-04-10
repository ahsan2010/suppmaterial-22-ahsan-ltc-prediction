#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:10:17 2023

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

shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/re-run-shap-values-march-7-2024/xin-ku-all-year/"
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
    
    return pd_full_merged, Y


def child_thread_shap_analysis(boot_id, file_name, year_value, X, Y, features):
    start_time = time.time()
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]
    filename_clf = "{}rf_{}_{}_{}.pkl".format(rf_model_output_path,file_name,boot_id,year_value)
    loaded_model = pickle.load(open(filename_clf, 'rb'))
    boot_index_data = year_value_boot_data[year_value_boot_data['boot_id'] == boot_id]
    train_index = boot_index_data[boot_index_data['index_type']=='train_index']['index_value'].tolist()
    test_index = boot_index_data[boot_index_data['index_type']=='test_index']['index_value'].tolist()
        
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    prediction = loaded_model.predict(X_test)    
    explainer = shap.Explainer(loaded_model.predict, X_test, seed = 1)
    shap_values = explainer(X_test)
    shap_values.feature_names = features
    
    shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_output_dir,file_name,year_value,boot_id)
    pickle.dump(shap_values, open(shap_output_file, 'wb'))
        
    print("Complete [{}] boot_id [{}] Time: {:.4f} seconds".format(file_name, boot_id,time.time() - start_time))

def multi_run_wrapper(args):
    boot_id = args[0]
    file_name = args[1]
    year_value = args[2]
    X = args[3]
    Y = args[4]
    features = args[5]
    child_thread_shap_analysis(boot_id, file_name, year_value, X, Y, features)
    
def parallel_shap_feature_importance(boot_lim, year_value, file_name):
    
    start_time = time.time()
    
    sample_iterations = list(range(1,boot_lim + 1,1))
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]
    X, Y = get_data(year_value, t_name[0])
    
    filename_clf = "{}rf_{}_{}_{}.pkl".format(rf_model_output_path,file_name,1,year_value)
    loaded_model = pickle.load(open(filename_clf, 'rb'))
    features = loaded_model.feature_names
    X = X[features]
    
    
    with get_context("spawn").Pool(20) as p:
        p.map(multi_run_wrapper,[[boot_id,file_name,year_value, X, Y, features] for boot_id in sample_iterations])
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


def generate_single_boot_feature_ranking(load_shap_value, feature_list):
    feature_avg_list = []
    for feature_ind in range(len(load_shap_value.feature_names)):
        feature_name = load_shap_value.feature_names[feature_ind]
        feature_data = load_shap_value.values[: , feature_ind]
        median_feature = np.average(np.abs(feature_data))
        feature_avg_list.append([feature_name, median_feature])
    
    pd_feature = pd.DataFrame(feature_avg_list)
    pd_feature.columns = ['feature_name', 'shap_value']
    
    for f in feature_list:
        if f not in pd_feature['feature_name'].values:
            pd_feature.loc[len(pd_feature.index)] = [f, 0.0]
    pd_feature = pd_feature.sort_values('shap_value', ascending = False)
    pd_feature['rank'] = range(1,len(load_shap_value.feature_names) + 1)
    
    return pd_feature

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
        return ("KU-REV-PROJ-PROF:"+ss)
   
    return (f)

def generate_sk_reank_input_data_per_year(boot_lim, shap_value_dir, file_name, year_value):
    
    feature_list = get_feature_name_list(boot_lim, shap_value_dir, file_name, year_value)
    feature_shap_all_boot = {}
    
    for f in feature_list:
        feature_shap_all_boot[f] = []
    
    for boot_id in range(1, boot_lim + 1):
        print("Working boot_id [{}/{}]".format(boot_id, boot_lim))
        shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_dir,file_name,year_value,boot_id)
        load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        pd_feature = generate_single_boot_feature_ranking(load_shap_value, feature_list)
    
        for d in zip(pd_feature['feature_name'].values, pd_feature['rank'].values):
            feature_shap_all_boot[d[0]].append(d[1])
    
    
    sk_output_file = "{}{}_sk_input_year_{}.txt".format(sk_rank_dir,file_name,year_value)
    with open(sk_output_file, 'w') as fp:
        for f in feature_list:
            rank_string = ' '.join([str(x) for x in feature_shap_all_boot[f]])
            fp.write(get_KU_feature_name(f))
            fp.write(' ')
            fp.write(rank_string)
            fp.write('\n')
    
    print("[Finish] file = {} year = {}".format(file_name, year_value))

def testing_shap_feature_importance(boot_lim, year_value, file_name):
    
    sample_iterations = list(range(1,boot_lim + 1,1))
    year_value_boot_data = pd_bootstrap_index[pd_bootstrap_index['ltc_year'] == year_value]
    X, Y = get_data(year_value, t_name[0])
    
    filename_clf = "{}rf_{}_{}_{}.pkl".format(rf_model_output_path,file_name,1,year_value)
    loaded_model = pickle.load(open(filename_clf, 'rb'))
    features = loaded_model.feature_names
    X = X[features]
    
    
    with get_context("spawn").Pool(20) as p:
        p.map(multi_run_wrapper,[[boot_id,file_name,year_value, X, Y, features] for boot_id in sample_iterations])
        p.close()
        p.join()
        
    for boot_id in sample_iterations:
        start_time = time.time()
        filename_clf = "{}rf_{}_{}_{}.pkl".format(rf_model_output_path,file_name,boot_id,year_value)
        loaded_model = pickle.load(open(filename_clf, 'rb'))
        boot_index_data = year_value_boot_data[year_value_boot_data['boot_id'] == boot_id]
        train_index = boot_index_data[boot_index_data['index_type']=='train_index']['index_value'].tolist()
        test_index = boot_index_data[boot_index_data['index_type']=='test_index']['index_value'].tolist()
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        explainer = shap.Explainer(loaded_model.predict, X_test)
        shap_values = explainer(X_test)
        shap_values.feature_names = features
        
        shap_output_file = "{}{}_shap_{}.csv".format(shap_value_output_dir,file_name,boot_id)
        pickle.dump(shap_values, open(shap_output_file, 'wb'))

        #load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        
        print("Complete [{}] boot_id [{}] Time: {:.4f} seconds".format(file_name, boot_id,time.time() - start_time))


def generate_sk_rank_data(file_name, shap_value_dir):
    year_values = [1,2,3]
    boot_lim = 100
    for year_value in year_values:
        generate_sk_reank_input_data_per_year(boot_lim, shap_value_dir, file_name, year_value)

def run_ku_all_auto():
    parallel_shap_feature_importance(100, 1, KU_ALL_AUTO)
    parallel_shap_feature_importance(100, 2, KU_ALL_AUTO)
    parallel_shap_feature_importance(100, 3, KU_ALL_AUTO)
    
def run_xin_ku_all_auto():
    parallel_shap_feature_importance(100, 1, KU_ALL_XIN_ALL_AUTO)
    parallel_shap_feature_importance(100, 2, KU_ALL_XIN_ALL_AUTO)
    parallel_shap_feature_importance(100, 3, KU_ALL_XIN_ALL_AUTO)

def main():
    shap_value_output_dir_combine = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
    shap_value_output_dir_ku = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/ku-all-dim-all-year/"
    
    #generate_sk_rank_data(KU_ALL_AUTO, shap_value_output_dir_ku)
    #generate_sk_rank_data(KU_ALL_XIN_ALL_AUTO, shap_value_output_dir_combine)
    
    #run_ku_all_auto()
    run_xin_ku_all_auto()
    
    print('Program finishes successfully')

if __name__ == "__main__":
    main()