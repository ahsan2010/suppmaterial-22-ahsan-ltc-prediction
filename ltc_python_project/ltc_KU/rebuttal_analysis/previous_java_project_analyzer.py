#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:25:34 2025

@author: ahsan
"""

import pandas as pd
import numpy as np


data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'
year_value = 1
path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
pd_data = pd.read_csv(path)


user_project_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/user_git_repos/json_to_csv/user_project_list.csv'
pd_user_project_data = pd.read_csv(user_project_file)


initial_java_project_users = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/user_git_repos/json_to_csv/user_project_list.csv'
ltc_data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/'
output_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/selected_other_projects.csv'


def load_ltc_data():
    pd_list = []
    for year in [1,2,3]:
        file_path = "{}full_result_updated_label_ltc{}.csv".format(ltc_data_dir, year)
        data = pd.read_csv(file_path)
        data["ltc_year"] = year
        pd_list.append(data)
    pd_full = pd.concat(pd_list)
    return pd_full

def select_other_project_users():
    gname='https://api.github.com/repos/'
    pd_ltc = load_ltc_data()
    pd_initial_project = pd.read_csv(initial_java_project_users)
    login_commit_data = pd_ltc[["login_x","FirstJavaCommit", "dev_name"]]
    login_commit_data.columns = ['user_login','first_java_commit', 'dev_name']
    
    pd_initial_project = pd_initial_project.drop_duplicates()
    login_commit_data = login_commit_data.drop_duplicates()
    login_commit_data= login_commit_data.drop_duplicates(subset='user_login', keep="first")
    
    merged_data = pd.merge(pd_initial_project, login_commit_data, on = ['user_login'], how = 'left')
    merged_data['created_at'] = pd.to_datetime(merged_data['created_at'], utc=True)
    merged_data['first_java_commit'] = pd.to_datetime(merged_data['first_java_commit'], utc=True)
    
    merged_data['project_before_first_commit'] = merged_data['first_java_commit'] > merged_data['created_at']

    merged_data['diff_years'] = (merged_data['first_java_commit'] - merged_data['created_at']).dt.days / 365.25
    
    merged_data = merged_data[(merged_data['diff_years'] >= 0) & (merged_data['diff_years'] <=2)]
    
    merged_data_selected = merged_data[merged_data['project_before_first_commit']]
    
    studied_developer = pd_data['dev_name'].tolist()
    
    merged_data_selected = merged_data_selected[merged_data_selected['dev_name'].isin(studied_developer)]
    
    merged_data_selected['proj'] = [x[len(gname):].replace('/','-') for x in merged_data_selected['repo_url']]
        
    
    merge_parent_proj_f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/mrege_parent_other_proj.csv'
    pd_merge_parent_proj = pd.read_csv(merge_parent_proj_f)    
    pd_merge_parent_proj = pd.merge(pd_merge_parent_proj, merged_data_selected[['user_login','dev_name','repo_url']], on = ['user_login','repo_url'], how = 'inner')
    
    dev_name_list = list(set(pd_merge_parent_proj.dev_name.tolist()))
    
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/select_commits_project_stat.csv'
    pd_other_project_data = pd.read_csv(f)
    
    selected_java_projects = set(pd_other_project_data.ProjectNmae.tolist())
    
    
    result = []
    
    java_file_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/other_project_java.csv'
    pd_java_file = pd.read_csv(java_file_path)
    file_map_result = pd_java_file.set_index("repo")["java_files"].to_dict()
    
    java_projects_studied = merged_data_selected[merged_data_selected['language'] == 'Java']['proj'].tolist()
    
    
    java_language_projects = []
    java_component_projects = []
    
    for d in dev_name_list:
        
        merg_sel_data_d = merged_data_selected[merged_data_selected['dev_name'] == d]
        merg_sel_data_d = merg_sel_data_d.dropna(subset=['language'])
        merg_sel_data_d['proj'] = [x[len(gname):].replace('/','_') for x in merg_sel_data_d['repo_url']]
        remv_proj = (pd_ltc[pd_ltc['dev_name'] == d]['Project_Name']).tolist()
    
        par_proj = pd_merge_parent_proj[pd_merge_parent_proj['dev_name']==d]
        par_proj['parent_repo'] = [x.replace('/','-') for x in par_proj['parent_repo']]
        par_proj['proj'] = [x[len(gname):].replace('/','-') for x in par_proj['repo_url']]
        
        data_map = par_proj.set_index("proj")["parent_repo"].to_dict()
        
        num_java_projects = np.sum(merg_sel_data_d['language']=='Java')
        non_java_projects = merg_sel_data_d[~(merg_sel_data_d['language']=='Java')]['proj'].tolist()
        
        java_language_projects.extend(merg_sel_data_d[(merg_sel_data_d['language']=='Java')]['proj'].tolist())
        
        project_with_java_files = num_java_projects
        for p in non_java_projects:
            if p in file_map_result:
                num_file = file_map_result[p]
                if num_file >= 2:
                    project_with_java_files = project_with_java_files + 1
                    java_component_projects.append(p)
        
        ratio = 100.0*num_java_projects/(project_with_java_files)
        
        result.append([d, num_java_projects, project_with_java_files,ratio])
        
    pd_result = pd.DataFrame(result)
    
    java_language_projects_set = set(java_language_projects)
    java_component_projects_set = set(java_component_projects)
    
    print(len(java_language_projects_set))
    print(len(java_component_projects_set))
    
    #pd_result.columns=['dev_name','java_projects','non_java_projects','ratio']
    #pd_result.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/reb_java_analysis.csv',index = False)
if __name__ == "__main__":
    select_other_project_users()
    print('Successfully completed')