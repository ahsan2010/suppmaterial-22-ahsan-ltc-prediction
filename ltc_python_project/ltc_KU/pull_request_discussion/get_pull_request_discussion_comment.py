# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:18:27 2021

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
import pandas as pd
import sys

csv.field_size_limit(sys.maxsize)

def request(url):
    git_token = "YOUR_GITHUB_KEY"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res

big_project_remove = ['https://api.github.com/repos/elastic/elasticsearch',
                      'https://api.github.com/repos/spring-projects/spring-boot',
                      'https://api.github.com/repos/facebook/react-native']

def get_pull_req_comments():
    json_folder = '/home/local/SAIL/ahsan/LTC_Project/pr-discussion-comments/'
    csv_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/pull-request/pr-data/csv-pr-data/"
    project_file_location = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    github_string = "https://api.github.com/repos/"
   
    ltc1_data_file = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/full_result_updated_label_ltc1.csv"
    data_ltc = pd.read_csv(ltc1_data_file)
    data_ltc['FirstJavaCommit'] = pd.to_datetime(data_ltc['FirstJavaCommit'])
    project_list = list(data_ltc['Project_Name'].unique())
    
    project_data = pd.read_csv(project_file_location)
    project_url_list = project_data['url'].to_list()
    total = 0
    for project_url in project_url_list:
        if project_url in big_project_remove:
                continue
        total = total + 1
        project = project_url[len(github_string):]
        project = project.replace("/","-")
        
        d = data_ltc[data_ltc['Project_Name'] == project]
        if d.shape[0] == 0:
            continue
        last_date = d['FirstJavaCommit'].max() + pd.DateOffset(months=1)
        first_date = d['FirstJavaCommit'].min() - pd.DateOffset(months=1)
        
        start = time.time()
        total_req = 0
        pull_request_file = '{}pr_{}.csv'.format(csv_dir, project)
        with open(pull_request_file) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                proj_name = row['Proj_Name']
                pr_url = row['PR_Url']
                pr_number = row['PR_Number']
                pr_creation = row['PR_Created_At']
                pr_creation_date = datetime.datetime.strptime(pr_creation, "%Y-%m-%dT%H:%M:%SZ")
                
                try:
                
                    #if pr_creation_date.replace(tzinfo=None) > last_date.replace(tzinfo=None) or pr_creation_date.replace(tzinfo=None) < first_date.replace(tzinfo=None):
                    #    print('Pr is created after the last date..')
                    #    continue
                    
                    json_output_dir = '{}{}/'.format(json_folder, proj_name)
                    
                    if not os.path.exists(json_output_dir):
                        os.makedirs(json_output_dir)
                    
                    total_req = total_req + 1
                    #print(json_output_dir)
                    
                    if os.path.isfile("{}pull_request_commennts_{}_{}.json".format(json_output_dir,proj_name, pr_number)):
                        print("Pull Request from '{}' repository were already collected. Continuing from next repository".format(pr_url))
                        continue
        
                    print("[START] ", proj_name)
                    query = "{}/issues/{}/comments?state=all&per_page=100".format(project_url,pr_number)
                    res = request(query)
                    time.sleep(2)
        
                    pull_request_reviewer_list = list()
        
                    try:
                        respos = res.json()
                        while 'next' in res.links.keys():
                            res = request(res.links['next']['url'])
                            time.sleep(2)
                        
                            try:
                                j = res.json()
                                respos.extend(j)
                            except Exception:
                                print("Repository '{}--{}' has a bad JSON format. Continuing from next repository".format(proj_name, pr_url))
                                continue
                        pull_request_reviewer_list.append(respos)
                    except Exception:
                        print("Repository '{}' has a bad JSON format. Continuing from next repository".format(query))
                        continue
        
                    i = {"project": proj_name,
                         "items": pull_request_reviewer_list}
                    with open("{}pull_request_commennts_{}_{}.json".format(json_output_dir,proj_name,pr_number), "w") as f:
                        json.dump(i, f)
                
                except Exception:
                    print('Problem in the row')
                    
                print("[END] {}--{}".format(proj_name, query))
            end = time.time()
            print(f"Runtime of the program {project} is {end - start}")
            

if __name__ == "__main__":
    start = time.time()
    get_pull_req_comments()
    end = time.time()
    print(f"Runtime of the program is {end - start}")

