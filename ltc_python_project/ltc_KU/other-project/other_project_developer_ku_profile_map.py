#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:38:40 2023

@author: ahsan
"""
import numpy as np
import pandas as pd
import time
import collections

other_project_parent = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/mrege_parent_other_proj.csv"
ku_profile_other_project_normalized = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/normalized_ku_full_ltc.csv"
ku_profile_other_project_count = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/count_ku_full_ltc.csv"

temp_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"

LTC_PATH_DIR = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"

LTC_ONE_FILE = "full_result_updated_label_ltc1"
LTC_TWO_FILE = "full_result_updated_label_ltc2"
LTC_THREE_FILE = "full_result_updated_label_ltc3"

ltc_file_list = [LTC_ONE_FILE, LTC_TWO_FILE, LTC_THREE_FILE]


def merge_ku_profiles():
    pd_other_parent = pd.read_csv(other_project_parent)
    pd_other_parent['parent_repo'] = [x.replace("/","-") for x in pd_other_parent['parent_repo']]
    
    path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
    normalized_list = list()
    count_list = list()
    
    project_list = pd_other_parent['parent_repo'].tolist()
    project_list = set(project_list)
    project_list = list(project_list)
    total = 0
    missing_normalized = 0
    missing_count = 0
    for p in project_list:
        total = total + 1
        print("Complete {} [{}/{}]".format(p, total, len(project_list)))
        norm_path = "{}normalized/{}_dev_normalized_ku_LTC.csv".format(path, p)
        count_path = "{}withoutNormalized/{}_dev_ku_LTC.csv".format(path, p)
        try:
            pd_norm = pd.read_csv(norm_path)
            normalized_list.append(pd_norm)
        except Exception as e:
            #print('Problem in path {}'.format(norm_path))
            #print(str(e))
            missing_normalized = missing_normalized + 1
            
        try:
            pd_count = pd.read_csv(count_path)
            count_list.append(pd_count)
        except Exception as e:
            #print('Problem in path {}'.format(count_path))
            #print(str(e))
            missing_count = missing_count + 1
        
    normalized_pd = pd.concat(normalized_list)
    count_pd = pd.concat(count_list)
    
    normalized_pd.to_csv("{}normalized_ku_full_ltc.csv".format(path),index = False)
    count_pd.to_csv("{}count_ku_full_ltc.csv".format(path),index = False)
    
    print(f"Missing count = {missing_count}/{len(project_list)}")
    print(f"Missing normalized = {missing_normalized}/{len(project_list)}")
    
    
def run_mapping(pd_other_parent, pd_knowledge, file_path):
    pd_temp = pd_other_parent
    pd_temp = pd_temp.rename(columns={'dev_name_x': 'Developer_Name'}) 
    merged_ku_profile_dev_name = pd.merge(pd_temp,pd_knowledge, how = 'inner', on = ['Developer_Name','Project_Name'])
    
    pd_other_parent = pd_other_parent.rename(columns={'Developer_Name': 'dev_name_x'}) 
    pd_other_parent['dev_name_x'] = pd_other_parent['user_login']
    pd_other_parent = pd_other_parent.rename(columns={'dev_name_x': 'Developer_Name'}) 
    merged_ku_profile_dev_login = pd.merge(pd_other_parent,pd_knowledge, how = 'inner', on = ['Developer_Name','Project_Name'])

    
    pd_knowledge_final = pd.concat([merged_ku_profile_dev_name,merged_ku_profile_dev_login])
    pd_knowledge_final.to_csv(file_path, index = False)
    return pd_knowledge_final
    

def find_map_value(x,map_data):
    if x in map_data:
        return map_data[x]
    return x

def map_developer():
    pd_other_parent = pd.read_csv(other_project_parent)
    pd_other_parent['parent_repo'] = [x.replace("/","-") for x in pd_other_parent['parent_repo']]
    pd_other_parent = pd_other_parent.rename(columns={'parent_repo': 'Project_Name'}) 
    
    pd_ku_profile_normalized = pd.read_csv(ku_profile_other_project_normalized)
    col = list(pd_ku_profile_normalized)
    norm_feature_list = ['norm_' + x for x in col[3:32]]
    norm_feature_list = col[0:3] + norm_feature_list 
    pd_ku_profile_normalized.columns = norm_feature_list
    
    pd_ku_profile_count = pd.read_csv(ku_profile_other_project_count)

    
    root = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
    output_file_count = "{}dev_ku_profile_count.csv".format(root)
    output_file_normalized = "{}dev_ku_profile_normalized.csv".format(root)
    output_file_count_normalized = "{}dev_ku_profile_count_norm.csv".format(root)
    
    pd_ku_count_final = run_mapping(pd_other_parent, pd_ku_profile_count, output_file_count)
    pd_ku_norm_final = run_mapping(pd_other_parent, pd_ku_profile_normalized, output_file_normalized)
    pd_ku_count_norm_final = pd.merge(pd_ku_count_final, pd_ku_norm_final, how = "inner", on = list(pd_ku_count_final.columns)[0:12])

    
    login_dev_map_file = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/user_map_result_full_ku.csv"
    pd_login_dev_map_file = pd.read_csv(login_dev_map_file)
    pd_login_dev_map_file = pd_login_dev_map_file[['login', 'dev_name']]
    pd_login_dev_map_file = pd_login_dev_map_file.dropna()
    
    map_data = {}
    temp_data = pd_login_dev_map_file.to_dict(orient = 'records')
    for d in temp_data:
        map_data[d['login']] = d['dev_name']
    
    pd_ku_count_final['Developer_Name'] = [ find_map_value(x, map_data) for x in pd_ku_count_final['user_login']]
    pd_ku_norm_final['Developer_Name'] = [ find_map_value(x, map_data) for x in pd_ku_norm_final['user_login']]
    pd_ku_count_norm_final['Developer_Name'] = [ find_map_value(x, map_data) for x in pd_ku_count_norm_final['user_login']]

    pd_ku_count_norm_final.to_csv(output_file_count_normalized, index = False)


def my_dev_merge_function_investigate(first_df, second_df):
    left_df_user_list = first_df['dev_name'].to_list()
    matched_user_list =[]
    record_find_per_match = []
    for ind in range(len(left_df_user_list)):
        user = left_df_user_list[ind]
        if (user in second_df['dev_name'].to_list()):
            matched_user_list.append(user)
            matched_records = second_df[second_df['dev_name'] == user]
            record_find_per_match.append(matched_records.shape[0])
            if (matched_records.shape[0] == 42):
                print(matched_records['dev_name'])
                print(first_df.loc[ind])
    
    counter = collections.Counter(record_find_per_match)
    unique_matched_user = set(matched_user_list)

def create_developer_feature_with_ltc():
    output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
    root = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
    file_count_normalized = "{}dev_ku_profile_count_norm.csv".format(root)
    
    
    pd_other_proj_ku = pd.read_csv(file_count_normalized)
    column_list = list(pd_other_proj_ku.columns)
    selected_columns = [column_list[10]] + [column_list[1]] + column_list[11:] 
    
    pd_other_proj_ku_selected = pd_other_proj_ku[selected_columns]
    
    group_by_column = ['Developer_Name']
    summary_columns = list(selected_columns)[3:]
    agg_dict = {}
    
    
    for col in summary_columns:
        #agg_dict[col] = np.median
        agg_dict[col] = np.sum
    
    df = pd_other_proj_ku_selected.groupby(group_by_column).agg(
            agg_dict
            ).reset_index()
    
    final_pd = df.drop_duplicates()
    final_pd = final_pd.rename(columns={'Developer_Name':'dev_name'})

    for i in range(len(ltc_file_list)):
        ltc_file_path = "{}{}.csv".format(LTC_PATH_DIR,ltc_file_list[i])
        pd_ltc = pd.read_csv(ltc_file_path)
        column_list = list(pd_ltc.columns)
        target = column_list[32:35]
        #selected_columns = column_list[0:2] + column_list[4:32] + column_list [35:64] + column_list [66:129] + target
        selected_columns = column_list[0:3] + column_list [66:129] + target

        pd_selected_col = pd_ltc[selected_columns]
        
        #my_dev_merge_function_investigate(pd_selected_col, final_pd)
        
        pd_ku_feature_full = pd.merge(pd_selected_col, final_pd, how = 'left', on = column_list[1:2])
        
        nan_rows = pd_ku_feature_full.isna().any(axis=1)
        nan_data = pd_ku_feature_full[nan_rows.to_list()]
        print(nan_data.shape[0])
        pd_ku_feature_full = pd_ku_feature_full.fillna(0)
    
        pd_ku_feature_full.to_csv("{}other_project_exp_ku_count_norm_label_ltc{}.csv".format(output_dir, i + 1), index = False)

        
def calculate_skewness():
    
    output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
    root = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/other-projects/"
    file_count_normalized = "{}dev_ku_profile_count_norm.csv".format(root)
    
    
    pd_other_proj_ku = pd.read_csv(file_count_normalized)
    column_list = list(pd_other_proj_ku.columns)
    selected_columns = [column_list[10]] + [column_list[1]] + column_list[11:] 
    
    pd_other_proj_ku_selected = pd_other_proj_ku[selected_columns]
    
    group_by_column = ['Developer_Name']
    summary_columns = list(selected_columns)[3:]
    agg_dict = {}
    
    
    for col in summary_columns:
        agg_dict[col] = np.median
    
    df = pd_other_proj_ku_selected.groupby(group_by_column).agg(
            agg_dict
            ).reset_index()
    
    final_pd = df.drop_duplicates()
    final_pd = final_pd.rename(columns={'Developer_Name':'dev_name'})
    
    dev_list = list(set(pd_other_proj_ku_selected['Developer_Name']))
    
    pd_other_proj_ku_selected = pd_other_proj_ku_selected.loc[:, (pd_other_proj_ku_selected != 0).any(axis=0)]
    col = list(pd_other_proj_ku_selected.columns)
    ku_list = col[3:31]
    
    from scipy.stats import kurtosis, skew 
    
    result = []
    
    for d in dev_list:
        pp = pd_other_proj_ku_selected[pd_other_proj_ku_selected['Developer_Name'] == d]
        pp_ku = pp[ku_list]
        sk = pp_ku.skew()
        result.append(list(sk))
   
    pd_result = pd.DataFrame(result)
    pd_result.columns = ku_list
    pd_result = pd_result.dropna()
    
    
    
    
def main():
    start_time = time.time()

    #merge_ku_profiles()
    #map_developer()
    create_developer_feature_with_ltc()
    
    print("Full Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
    print("Program finishes successfully!")
    
if __name__ == "__main__":
    main()