#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:45:49 2025

@author: ahsan
"""
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
import urllib.parse

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
    
def request(url):
    git_token = "GITHUB_KEY"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res


def crawn_java_files_stat():
    merged_data_selected = select_other_project_users()
    repo_url_list = list(set(merged_data_selected.repo_url))
    EXTENSION= "java"
    gname = 'https://api.github.com/repos/'
    
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/project_java_file_count_Feb_8.csv'
    target = open(f,"a")
    target.write("repo_url,own_repo,java_file_count\n")
    target.flush()
    total = len(repo_url_list)
    working = 0
    for r in repo_url_list:
        working = working + 1
        
        owner_repo = r[len(gname):]
        q = "https://api.github.com/search/code?q=repo:{} +extension:{}".format(owner_repo, EXTENSION)
        response = request(q)
        time.sleep(1)
        
        if response.status_code == 200:
            data = response.json()
            java_file_count = data.get('total_count', 0)
            
            target.write("{},{},{}\n".format(r,owner_repo,java_file_count))
            target.flush()
        
        print("Working: {} {} : {}/{}".format(r, response.status_code, working,total))
        
        if working > 15:
            break
    target.close()
    
    
if __name__ == "__main__":
    crawn_java_files_stat()
    print('Successfully completed')