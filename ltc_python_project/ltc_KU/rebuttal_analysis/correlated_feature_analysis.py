#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:41:46 2025

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

dev_exp_features_ind = list(range(3,32))
cur_proj_features_ind = list(range(94,123))
prev_dev_exp_features_ind = list(range(123,152))
col_exp_features_ind = list(range(152,181))
prev_proj_features_ind = list(range(181,210))

#Xin feature dimension index
xin_dev_prof = list(range(33,40))
xin_repo_prof = list(range(40,59))
xin_month_repo_prof = list(range(59,77))
xin_dev_act_prof = list(range(77,89))
xin_collab_prof = list(range(89,94))

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

def get_feature_names(col_list):
    dev_exp_ku_features = [col_list[i] for i in dev_exp_features_ind] 
    cur_proj_ku_features = [col_list[i] for i in cur_proj_features_ind]
    prev_dev_exp_features = [col_list[i] for i in prev_dev_exp_features_ind]
    col_exp_features = [col_list[i] for i in col_exp_features_ind]
    prev_proj_features = [col_list[i] for i in prev_proj_features_ind]
    
    xin_dev_prof_features = [col_list[i] for i in xin_dev_prof]
    xin_repo_prof_features = [col_list[i] for i in xin_repo_prof]
    xin_month_repo_prof_features = [col_list[i] for i in xin_month_repo_prof]
    xin_dev_act_prof_features = [col_list[i] for i in xin_dev_act_prof]
    xin_collab_prof_features = [col_list[i] for i in xin_collab_prof]

    ku_feature_dimension_list ={
        'KULTC_DEV_EXP' : dev_exp_ku_features,
        'KULTC_PROJ' : cur_proj_ku_features,
        'KULTC_DEV_PREV_EXP' : prev_dev_exp_features,
        'KULTC_COLLAB_EXP' : col_exp_features,
        'KULTC_PREV_PROJ' : prev_proj_features
        }
    
    xin_feature_dimension_list = {
        'BAOLTC_DEV_PPROF'  : xin_dev_prof_features,
        'BAOLTC_REPO_PROF' : xin_repo_prof_features,
        'BAOLTC_DEV_ACT' : xin_dev_act_prof_features,
        'BAOLTC_REPO_ACT' : xin_month_repo_prof_features,
        'BAOLTC_COLLAB_NET' : xin_collab_prof_features
        }
    
    feature_dimension_list = {
        'KULTC_DEV_EXP' : dev_exp_ku_features,
        'KULTC_PROJ' : cur_proj_ku_features,
        'KULTC_DEV_PREV_EXP' : prev_dev_exp_features,
        'KULTC_COLLAB_EXP' : col_exp_features,
        'KULTC_PREV_PROJ' : prev_proj_features,
        'BAOLTC_DEV_PPROF'  : xin_dev_prof_features,
        'BAOLTC_REPO_PROF' : xin_repo_prof_features,
        'BAOLTC_DEV_ACT' : xin_dev_act_prof_features,
        'BAOLTC_REPO_ACT' : xin_month_repo_prof_features,
        'BAOLTC_COLLAB_NET' : xin_collab_prof_features
        }
    
    map_feature_dimension = {}
    for key,value in feature_dimension_list.items():
        f_list = value
        for f in f_list:
            map_feature_dimension[f] = key
    
    return ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension

def analyze_correlated_features():
    year_value = 1
    path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, year_value)
    pd_full = pd.read_csv(path)
    
    col_list = list(pd_full.columns)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]
    
    xin_dev_dim = xin_features[0:8]
    xin_repo_dim = xin_features[9:27]
    xin_dev_month_act_dim = xin_features[28:45]
    xin_repo_month_act_dim = xin_features[46:57]
    collab_network_dim = xin_features[58:62]
    
    ku_proj_dim_count_col = ["proj_dim_" + x for x in ku_features_count]
    ku_prev_exp_count_col = ["prev_exp_" + x for x in ku_features_count]
    ku_comm_exp_count_col = ["comm_exp_" + x for x in ku_features_count]
    ku_other_proj_dim_count_col =["other_proj_dim_" + x for x in ku_features_count]
    
    all_ku_count_feature = ku_features_count + ku_proj_dim_count_col + ku_prev_exp_count_col + ku_comm_exp_count_col + ku_other_proj_dim_count_col
    all_ku_all_xin = all_ku_count_feature + xin_features
    
    ku_cur_dev_exp = ku_features_count
    ku_xin_dev = all_ku_count_feature + xin_dev_dim
    ku_xin_repo = all_ku_count_feature + xin_repo_dim
    ku_xin_dev_month_act = all_ku_count_feature + xin_dev_month_act_dim
    ku_xin_repo_month_act = all_ku_count_feature + xin_repo_month_act_dim
    ku_xin_collab = all_ku_count_feature + collab_network_dim
    ku_xin_all = all_ku_all_xin
    
    
    
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
    
    col_names = list(pd_full_merged)
    ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension = get_feature_names(col_names)
    
    feature_list = {
        
        KU_ALL_AUTO          : all_ku_count_feature,
        XIN_FEAT_AUTO        : xin_features,
        KU_ALL_XIN_ALL_AUTO  : all_ku_all_xin
        }
    model_name = KU_ALL_AUTO
    print(model_name)
    feat = feature_list[model_name]
    X = pd_full_merged [feat]
    auto_feature = AutoSpearman(X_train = X, verbose=False)
    
    pd_feature = pd.DataFrame(auto_feature, columns = ["feature"])
    pd_feature['dimension'] = [map_feature_dimension[x] for x in pd_feature['feature']]
    pd_feature=pd_feature.sort_values('dimension')
   
    pd_feature.loc[pd_feature['dimension']=='KULTC_DEV_EXP','feature'] = 'dev_exp_' + pd_feature.loc[pd_feature['dimension']=='KULTC_DEV_EXP','feature']
    pd_feature.loc[pd_feature['dimension']=='KULTC_PREV_PROJ','feature'] = pd_feature.loc[pd_feature['dimension']=='KULTC_PREV_PROJ','feature'].str.replace('other_proj','prev_proj')
    pd_feature.loc[pd_feature['dimension']=='KULTC_COLLAB_EXP','feature'] = pd_feature.loc[pd_feature['dimension']=='KULTC_COLLAB_EXP','feature'].str.replace('comm_','collab_')

    
    ROOT = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/'
    f = '{}feature_after_auto_{}.csv'.format(ROOT,model_name)
    pd_feature.to_csv(f, index = False)
    

def fixing_corr_table_from_csv():
    year_value = 1
    path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, year_value)
    pd_full = pd.read_csv(path)
    
    col_list = list(pd_full.columns)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]
    
    xin_dev_dim = xin_features[0:8]
    xin_repo_dim = xin_features[9:27]
    xin_dev_month_act_dim = xin_features[28:45]
    xin_repo_month_act_dim = xin_features[46:57]
    collab_network_dim = xin_features[58:62]
    
    ku_proj_dim_count_col = ["proj_dim_" + x for x in ku_features_count]
    ku_prev_exp_count_col = ["prev_exp_" + x for x in ku_features_count]
    ku_comm_exp_count_col = ["comm_exp_" + x for x in ku_features_count]
    ku_other_proj_dim_count_col =["other_proj_dim_" + x for x in ku_features_count]
    
    all_ku_count_feature = ku_features_count + ku_proj_dim_count_col + ku_prev_exp_count_col + ku_comm_exp_count_col + ku_other_proj_dim_count_col
    all_ku_all_xin = all_ku_count_feature + xin_features
    
    ku_cur_dev_exp = ku_features_count
    ku_xin_dev = all_ku_count_feature + xin_dev_dim
    ku_xin_repo = all_ku_count_feature + xin_repo_dim
    ku_xin_dev_month_act = all_ku_count_feature + xin_dev_month_act_dim
    ku_xin_repo_month_act = all_ku_count_feature + xin_repo_month_act_dim
    ku_xin_collab = all_ku_count_feature + collab_network_dim
    ku_xin_all = all_ku_all_xin
    
    
    
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
    
    
    col_names = list(pd_full_merged)
    ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension = get_feature_names(col_names)
    
    
    
    cor_ku_all = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/cor_ku_all.csv'
    pd_ku_all = pd.read_csv(cor_ku_all)
    
    pd_ku_all['dimension'] = [map_feature_dimension[x] for x in pd_ku_all['excluded_feat']]
    
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat1'] = 'dev_exp_' + pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat1']
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat1'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat1'].str.replace('other_proj','prev_proj')
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat1'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat1'].str.replace('comm_','collab_')

    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat2'] = 'dev_exp_' + pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat2']
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat2'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat2'].str.replace('other_proj','prev_proj')
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat2'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat2'].str.replace('comm_','collab_')

    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','excluded_feat'] = 'dev_exp_' + pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','excluded_feat']
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','excluded_feat'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','excluded_feat'].str.replace('other_proj','prev_proj')
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','excluded_feat'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','excluded_feat'].str.replace('comm_','collab_')

    pd_ku_all.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/cor_ku_all_table.csv', index = False)
    
    
    cor_ku_xin_all = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/cor_ku_all_xin_all.csv'
    pd_ku_all = pd.read_csv(cor_ku_xin_all)
    pd_ku_all['dimension'] = [map_feature_dimension[x] for x in pd_ku_all['excluded_feat']]
    
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat1'] = 'dev_exp_' + pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat1']
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat1'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat1'].str.replace('other_proj','prev_proj')
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat1'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat1'].str.replace('comm_','collab_')

    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat2'] = 'dev_exp_' + pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','feat2']
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat2'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','feat2'].str.replace('other_proj','prev_proj')
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat2'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','feat2'].str.replace('comm_','collab_')

    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','excluded_feat'] = 'dev_exp_' + pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_DEV_EXP','excluded_feat']
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','excluded_feat'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_PREV_PROJ','excluded_feat'].str.replace('other_proj','prev_proj')
    pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','excluded_feat'] = pd_ku_all.loc[pd_ku_all['dimension']=='KULTC_COLLAB_EXP','excluded_feat'].str.replace('comm_','collab_')
     
    pd_ku_all.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/cor_ku_all_xin_all_table.csv', index = False)
    
    
def distribution_analysis():
    year_value = 1
    path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, year_value)
    pd_full = pd.read_csv(path)
    
    col_list = list(pd_full.columns)
    ku_features_count = list(pd_full.columns)[3:32]
    ku_features_norm = list(pd_full.columns)[35:64]
    ku_full_features = ku_features_count + ku_features_norm
    xin_features = list(pd_full.columns)[67:129]
    
    xin_dev_dim = xin_features[0:8]
    xin_repo_dim = xin_features[9:27]
    xin_dev_month_act_dim = xin_features[28:45]
    xin_repo_month_act_dim = xin_features[46:57]
    collab_network_dim = xin_features[58:62]
    
    ku_proj_dim_count_col = ["proj_dim_" + x for x in ku_features_count]
    ku_prev_exp_count_col = ["prev_exp_" + x for x in ku_features_count]
    ku_comm_exp_count_col = ["comm_exp_" + x for x in ku_features_count]
    ku_other_proj_dim_count_col =["other_proj_dim_" + x for x in ku_features_count]
    
    all_ku_count_feature = ku_features_count + ku_proj_dim_count_col + ku_prev_exp_count_col + ku_comm_exp_count_col + ku_other_proj_dim_count_col
    all_ku_all_xin = all_ku_count_feature + xin_features
    
    ku_cur_dev_exp = ku_features_count
    ku_xin_dev = all_ku_count_feature + xin_dev_dim
    ku_xin_repo = all_ku_count_feature + xin_repo_dim
    ku_xin_dev_month_act = all_ku_count_feature + xin_dev_month_act_dim
    ku_xin_repo_month_act = all_ku_count_feature + xin_repo_month_act_dim
    ku_xin_collab = all_ku_count_feature + collab_network_dim
    ku_xin_all = all_ku_all_xin
    
    
    
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
    
    sel_col_prev_exp = list(pd_prev_exp_selected.columns)[4:34]
    sel_col_prev = list(pd_prev_proj_selected.columns)[4:34]
    sel_col_comm = list(pd_comm_exp_selected.columns)[4:34]
    
    pd_prev_exp_data = pd_prev_exp_selected[sel_col_prev_exp]
    pd_prev_proj_data = pd_prev_proj_selected[sel_col_prev]
    pd_comm_data = pd_comm_exp_selected[sel_col_comm]

if __name__ == "__main__":
    start_time = time.time()
    analyze_correlated_features()
    print("PROGRAM FINISHES SUCCESSFULLY")
    print(" Total Time: --- {:.4f} minutes ---".format((time.time() - start_time)/60.0))
