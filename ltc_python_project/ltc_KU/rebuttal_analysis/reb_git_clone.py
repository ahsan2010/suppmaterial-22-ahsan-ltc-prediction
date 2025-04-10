#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:37:08 2025

@author: ahsan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:45:00 2024

@author: ahsan
"""
from git import RemoteProgress
import pandas as pd
import git
import requests
import time
import os
import datetime
import json
import glob
import copy
import csv
import os.path
import pandas as pd
import sys
import urllib.parse
import pathlib
output_json_path = '/home/local/SAIL/ahsan/LTC_Project/Contributor_Information/'
full_contrib_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_1_java_with_gt_login_name.csv'

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
    
    return merged_data_selected

class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            print(message)


def clone_analysis(from_ind, to_ind):
    merged_data_selected = select_other_project_users()
    repo_url_list = list(set(merged_data_selected.repo_url))
    
    gname = 'https://api.github.com/repos/'
    
    total = to_ind - from_ind
    working = 0
    
    for i in range(from_ind, to_ind):
        working = working + 1
        r  = repo_url_list[i]
        
        print('Working{}:  {}/{}'.format(r, working, total))
        owner_repo = r[len(gname):]
        owner_repo_dot = owner_repo.replace('/','.')
        clone_from = 'https://github.com/{}.git'.format(owner_repo)
        clone_to = '/home/local/SAIL/ahsan/LTC_Project/other_projects_rebuttals/{}'.format(owner_repo.replace('/','_'))
        
        try:
            git.Repo.clone_from(clone_from, clone_to, progress = CloneProgress())
        except Exception as ep:
            print(ep)
        except:
            print("Error")

def calculate_java_files():
    dir = '/home/local/SAIL/ahsan/LTC_Project/other_projects_rebuttals'
    merged_data_selected = select_other_project_users()
    repo_url_list = list(set(merged_data_selected.repo_url))
    gname = 'https://api.github.com/repos/'
    working = 0
    total = len(repo_url_list)
    
    full_results = []
    
    for i in range(len(repo_url_list)):
        working = working + 1
        r  = repo_url_list[i]
        
        print('Working{}:  {}/{}'.format(r, working, total))
        owner_repo = r[len(gname):]
        owner_repo_hyphen = owner_repo.replace('/','_')
        repo_path = '{}/{}'.format(dir,owner_repo_hyphen)
        allfiles = [f for f in pathlib.Path(repo_path).rglob("*.java")]
        
        r = [r, owner_repo_hyphen, len(allfiles)]
        full_results.append(r)
    
    pd_full_result = pd.DataFrame(full_results)
    output_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/other_project_java.csv'
    pd_full_result.to_csv(output_path, index = False)
    pd_full_result.columns = ['repo_url','repo','java_files']

def start_crawl():
    from_ind = int(sys.argv[1])
    to_ind = int(sys.argv[2])
    clone_analysis(from_ind,to_ind)

if __name__ == "__main__":
    calculate_java_files()
    print('Successfully completed')
 