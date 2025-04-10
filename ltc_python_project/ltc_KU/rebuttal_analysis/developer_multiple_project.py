#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:22:00 2025

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


def study_developer_working_multiple_projects():
    dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/'
    year_value = 1
    pd_full, pd_full_merged, _,_,_, _, _  = get_full_feature_data(year_value)
    duplicates = pd_full[pd_full.duplicated(subset=['dev_name'], keep=False)]
    duplicates = duplicates.drop_duplicates(subset=['Project_Name','dev_name'], keep=False)
    f = '{}duplicate_dev_project.csv'.format(dir)
    duplicates.to_csv(f,index=False)
    user_involve_multiple_project = list(set(duplicates['dev_name']))
    frequency = duplicates['dev_name'].value_counts()
    pass

def study_deveolopers_prev_projects():
    dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/rebuttal/'
    year_value = 1
    pd_full, pd_full_merged, _,_,_, _, _  = get_full_feature_data(year_value)
    
    dev_name_list = list(set(pd_full['dev_name']))
    
    prev_proj_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/selected_other_projects.csv'
    pd_prev_proj = pd.read_csv(prev_proj_file)
    
    pd_count_prev_proj = pd_prev_proj.groupby('dev_name')['repo_url'].nunique().reset_index()
    
    prev_proj_dataset = pd_count_prev_proj[pd_count_prev_proj['dev_name'].isin(dev_name_list)]
    
    
    p = pd_full.merge(prev_proj_dataset, how = 'left', on = 'dev_name')
    
    p_prev_proj_col = p[['dev_name','repo_url']]
    
    p_prev_proj_col['repo_url'] = p_prev_proj_col['repo_url'].fillna(0)
    p_prev_proj_col['repo_url'].describe()

    