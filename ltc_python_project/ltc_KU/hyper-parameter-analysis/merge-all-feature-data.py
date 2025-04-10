#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:42:05 2024

@author: ahsan
"""

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

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']


data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'

def data_merge_all_features(year_value,target_name):
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
                                    pd_full[col_list[32:35] + [col_list[129]]].reset_index(drop=True) ], axis = 1)
    
    output_path = "{}ltc_data_full_year_{}_30_jan_2025.csv".format(data_dir, year_value)
    #output_path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full_merged.to_csv(output_path, index = False)
   
def merge_all_dimension_data():
    start_time = time.time()
    data_merge_all_features(1,t_name[0])
    print(" YEAR 1 --- {:.4f} seconds ---".format(time.time() - start_time))    
    
    start_time = time.time()
    data_merge_all_features(2,t_name[0])
    print(" YEAR 2 --- {:.4f} seconds ---".format(time.time() - start_time))
    
    start_time = time.time()
    data_merge_all_features(3,t_name[0])
    print(" YEAR 3 --- {:.4f} seconds ---".format(time.time() - start_time))


if __name__ == "__main__":
    merge_all_dimension_data()
    print("Program finishes successfully.")