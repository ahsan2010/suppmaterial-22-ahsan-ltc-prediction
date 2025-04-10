#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:01:47 2025

@author: ahsan
"""

import pandas as pd
import numpy as np

dev_month_ku_expertise_file_dir  = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/'
project_ku_profile_dir                      = "/home/local/SAIL/ahsan/LTC_Project/project-ku-profile/"
dev_other_project_experience_ku_profile_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
community_ku_profile_dir                    = "/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/"    
other_project_ku_profile_dir                = "/home/local/SAIL/ahsan/LTC_Project/other-project-ku-profile/"

def get_target_label(year_value, pd_full):
    if year_value == 1:
        Y = pd_full[['LTC_Developer_Cat_Year_One']]
    elif year_value == 2:
        Y = pd_full[['LTC_Developer_Cat_Year_Two']]
    elif year_value == 3:
        Y = pd_full[['LTC_Developer_Cat_Year_Three']]
        
    return Y

def get_full_feature_data(year_value):
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
    
    return pd_full, pd_full_merged, ku_features_count,ku_proj_dim_count_col,ku_prev_exp_count_col, ku_comm_exp_count_col, ku_other_proj_dim_count_col

def analyze_year_dev_expertise_values(year_value):
    
    pd_full, pd_full_merged, cur_exp_col, cur_proj_col, prev_exp_col, comm_exp_col, prev_proj_col = get_full_feature_data(year_value)
    Y = get_target_label(year_value, pd_full)
    pd_full_merged['is_LTC'] = Y
    
    pd_full_merged['tot_cur_exp'] = pd_full_merged[cur_exp_col].sum(axis=1)
    pd_full_merged['tot_prev_exp']  = pd_full_merged[prev_exp_col].sum(axis=1)
    
    pd_full_merged_no_prev_exp = pd_full_merged[pd_full_merged['tot_prev_exp'] == 0.0]
    
    pd_full_cur_exp  = pd_full_merged_no_prev_exp[cur_exp_col+['is_LTC','tot_cur_exp']]
    
    pd_full_prev_exp = pd_full_merged[prev_exp_col + ['is_LTC','tot_prev_exp']]
    
    ltc_cur_exp = pd_full_cur_exp[pd_full_cur_exp['is_LTC']==1]
    non_ltc_cur_exp = pd_full_cur_exp[pd_full_cur_exp['is_LTC']==0]
    
    ltc_prev_exp = pd_full_prev_exp[pd_full_prev_exp['is_LTC']==1]
    non_ltc_prev_exp = pd_full_prev_exp[pd_full_prev_exp['is_LTC']==0]
    
    
    
    pass