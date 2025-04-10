#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:22:59 2024

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

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']


sk_input_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature-analysis/sk-rank-input/'
data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'


KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"
KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

# KU feature dimension index
dev_exp_features_ind = list(range(4,32))
cur_proj_features_ind = list(range(95,123))
prev_dev_exp_features_ind = list(range(124,152))
col_exp_features_ind = list(range(153,181))
prev_proj_features_ind = list(range(182,210))

#Xin feature dimension index
xin_dev_prof = list(range(33,40))
xin_repo_prof = list(range(40,59))
xin_month_repo_prof = list(range(59,77))
xin_dev_act_prof = list(range(77,89))
xin_collab_prof = list(range(89,94))

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)
shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/shap/"

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 

full_model = [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO, KU_CUR_XIN_AUTO]

ku_dimension_list = [
    'KU_DEV_EXP',
    'KU_CUR_PROJ',
    'KU_DEV_PREV_EXP',
    'KU_COLLAB_EXP',
    'KU_PREV_PROJ'
    ]

xin_dimension_list = [
    'BAO_DEV_PPROF',
    'BAO_REPO_PROF',
    'BAO_DEV_ACT',
    'BAO_REPO_ACT',
    'BAO_COLLAB_NET'
    ]


def map_feature_to_dimension(pd_feature,model_dimension_list, map_feature_dimension):
    feature_list = list(pd_feature['feature_name'])
    dimension_list = []
    for f in feature_list:
        dimension_list.append(map_feature_dimension[f])
    return dimension_list

def generate_single_boot_feature_ranking(load_shap_value, model_dimension_list, map_feature_dimension):
    feature_list = load_shap_value.feature_names
    feature_shap_rank = {}
    for d in model_dimension_list:
        feature_shap_rank[d] = []
    for row_ind in range(len(load_shap_value.values)):
        feature_name = load_shap_value.feature_names
        feature_data = np.abs(load_shap_value.values[row_ind, :])
        pd_feature = pd.DataFrame({'feature_name':feature_name,'shap_value': feature_data})
        dimension_list = map_feature_to_dimension(pd_feature,  model_dimension_list, map_feature_dimension)
        pd_feature['dimension'] = dimension_list
        pd_sum = pd_feature.groupby('dimension').sum().reset_index()
        for d in model_dimension_list:
            if d in list(pd_sum['dimension']):
                v = pd_sum[pd_sum['dimension'] == d].shap_value.values[0]
                feature_shap_rank[d].append(v)
    
    return feature_shap_rank
    

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
        'KU_DEV_EXP' : dev_exp_ku_features,
        'KU_CUR_PROJ' : cur_proj_ku_features,
        'KU_DEV_PREV_EXP' : prev_dev_exp_features,
        'KU_COLLAB_EXP' : col_exp_features,
        'KU_PREV_PROJ' : prev_proj_features
        }
    
    xin_feature_dimension_list = {
        'BAO_DEV_PPROF'  : xin_dev_prof_features,
        'BAO_REPO_PROF' : xin_repo_prof_features,
        'BAO_DEV_ACT' : xin_dev_act_prof_features,
        'BAO_REPO_ACT' : xin_month_repo_prof_features,
        'BAO_COLLAB_NET' : xin_collab_prof_features
        }
    
    feature_dimension_list = {
        'KU_DEV_EXP' : dev_exp_ku_features,
        'KU_CUR_PROJ' : cur_proj_ku_features,
        'KU_DEV_PREV_EXP' : prev_dev_exp_features,
        'KU_COLLAB_EXP' : col_exp_features,
        'KU_PREV_PROJ' : prev_proj_features,
        'BAO_DEV_PPROF'  : xin_dev_prof_features,
        'BAO_REPO_PROF' : xin_repo_prof_features,
        'BAO_DEV_ACT' : xin_dev_act_prof_features,
        'BAO_REPO_ACT' : xin_month_repo_prof_features,
        'BAO_COLLAB_NET' : xin_collab_prof_features
        }
    
    map_feature_dimension = {}
    for key,value in feature_dimension_list.items():
        f_list = value
        for f in f_list:
            map_feature_dimension[f] = key
    
    return ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension

def shap_featue_analysis(year_value):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension = get_feature_names(col_list)
    model_dimension_list = {
        XIN_FEAT_AUTO : ['BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ'],
        KU_ALL_XIN_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_CUR_XIN_AUTO : ['KU_DEV_EXP','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET']
        }
    
    
    for m in [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO, KU_CUR_XIN_AUTO]:
        shap_file = "{}{}_shap_full_{}.pkl".format(shap_value_output_dir,m,year_value)
        load_shap_value = pickle.load(open(shap_file, 'rb'))
        feature_shap_value = generate_single_boot_feature_ranking(load_shap_value, model_dimension_list[m], map_feature_dimension)
   
        sk_output_file = "{}{}_sk_input_year_{}.txt".format(sk_input_dir,m,year_value)
        with open(sk_output_file, 'w') as fp:
            for f in feature_shap_value.keys():
                rank_string = ' '.join([str(x) for x in feature_shap_value[f]])
                fp.write(f)
                fp.write(' ')
                fp.write(rank_string)
                fp.write('\n')

        print("[Finish] file = {} year = {}".format(m, year_value))



def main():
    for year_value in [1,2,3]:
        shap_featue_analysis(year_value)
    print("main function..")

if __name__ == "__main__":
    main()
    print("Program finishes successfully.")