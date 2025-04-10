#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:59:51 2024

@author: ahsan
"""

import pandas as pd
import os

def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x

def analyze_commit_data():
    project_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['html_url']]
    proj_list = list(pd_proj_data['repository'])
    total = 0
    
    commit_merge_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_merge_data/'

    total_commits_list = [] 
    for proj in proj_list:
        total = total + 1
        print("Working {}".format(total))
        file_name = os.path.join(commit_merge_dir,'{}_full_commit_data.csv'.format(proj))
        data = pd.read_csv(file_name)
        total_commits_list.extend(data[data['no_modified_java_file'] > 0]['commit_id'])
        
    print("Total Commit {}".format(len(total_commits_list)))
    
    total_pr_list = []
    pr_directory = '/home/local/SAIL/ahsan/LTC_Project/pull-request/pr-data/csv-pr-data/'
    for proj in proj_list:
        total = total + 1
        print("Working {}".format(total))
        file_name = os.path.join(pr_directory,'pr_{}.csv'.format(proj))
        data = pd.read_csv(file_name)
        total_pr_list.extend(data['PR_ID'])
    
    print("Total PR List {}".format(len(total_pr_list)))
