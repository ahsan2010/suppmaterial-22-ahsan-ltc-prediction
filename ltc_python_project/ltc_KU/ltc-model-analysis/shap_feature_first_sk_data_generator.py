#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 01:03:56 2024

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
dev_month_ku_expertise_file_dir             = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/'

sk_rank_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/first_sk_input/"

shap_value_output_dir_ku_all = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/ku_all_dim/"
shap_value_output_dir_ku_xin_all = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/ku_xin_all_dim/"
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
        print("Working {} boot_id [{}/{}]".format(file_name, boot_id, boot_lim))
        shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_dir,file_name,year_value,boot_id)
        load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        feature_shap_rank = generate_single_boot_feature_ranking(load_shap_value, feature_list)
    
        sk_output_file = "{}{}/{}_sk_input_year_{}_{}.txt".format(sk_rank_dir,file_name,file_name,year_value,boot_id)
        with open(sk_output_file, 'w') as fp:
            for f in feature_list:
                rank_string = ' '.join([str(x) for x in feature_shap_rank[f]])
                fp.write(get_KU_feature_name(f))
                fp.write(' ')
                fp.write(rank_string)
                fp.write('\n')
    
    print("[Finish] file = {} year = {}".format(file_name, year_value))

def generate_sk_rank_data(file_name, shap_value_dir):
    year_values = [1,2,3]
    boot_lim = 100
    for year_value in year_values:
        generate_sk_reank_input_data_per_year(boot_lim, shap_value_dir, file_name, year_value)
        print("Done Year: {} {}".format(file_name, year_value))

def main():
    shap_value_output_dir_combine = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
    shap_value_output_dir_ku = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/ku-all-dim-all-year/"
    
    generate_sk_rank_data(KU_ALL_AUTO, shap_value_output_dir_ku)
    generate_sk_rank_data(KU_ALL_XIN_ALL_AUTO, shap_value_output_dir_combine)
    
    
    print('Program finishes successfully')

if __name__ == "__main__":
    main()