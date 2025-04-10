#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:52:36 2021

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

project_list = ['apache_lucene', 'apache_wicket', 'apache_hbase', 'apache_groovy', 'apache_hive', 'apache_activemq',
'apache_storm','apache_stratos', 'elastic_elasticsearch', 'jruby_jruby', 'caskdata_cdap']


def request(url):
    git_token = "c7df13288605b85568f1044c9e3896cd6ecfcca2"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res

def get_pull_req_reviewers():
    pull_request_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request/'
    json_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/'
    
    for project in project_list:
        start = time.time()
        total_req = 0
        pull_request_file = '{}pr_reports_{}.csv'.format(pull_request_folder, project)
        with open(pull_request_file) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                git_repo_name = row['Git_Repo_Name']
                git_proj_name = row['Git_Proj_Name']
                proj_name = row['Proj_Name']
                pr_url = row['PR_Url']
                pr_number = row['PR_Number']
                
                json_output_dir = '{}{}/'.format(json_folder, proj_name)
                
                if not os.path.exists(json_output_dir):
                    os.makedirs(json_output_dir)
                
                total_req = total_req + 1
                #print(json_output_dir)
                
                if os.path.isfile("{}pull_request_reviews_{}_{}.json".format(json_output_dir,proj_name, pr_number)):
                    print("Pull Request from '{}' repository were already collected. Continuing from next repository".format(pr_url))
                    continue
    
                print("[START] ", proj_name)
                
                res = request("{}/reviews?state=all&per_page=100".format(pr_url))
                time.sleep(2)
    
                pull_request_reviewer_list = list()
    
                try:
                    j = res.json()
                    pull_request_reviewer_list = pull_request_reviewer_list + j
                except Exception:
                    print("Repository '{}' has a bad JSON format. Continuing from next repository".format(proj_name))
                    continue
                
                while 'next' in res.links.keys():
                    res = request(res.links['next']['url'])
                    time.sleep(3)
                    
                    try:
                        j = res.json()
                        pull_request_reviewer_list = pull_request_reviewer_list + j
                       
                    except Exception:
                        print("Repository '{}--{}' has a bad JSON format. Continuing from next repository".format(proj_name, pr_url))
                        continue
                    
    
                i = {"project": proj_name,
                     "items": pull_request_reviewer_list}
                with open("{}pull_request_reviews_{}_{}.json".format(json_output_dir,proj_name,pr_number), "w") as f:
                    json.dump(i, f)
                    
                print("[END] {}--{}".format(proj_name, pr_url))
            end = time.time()
            print(f"Runtime of the program {project} is {end - start}")

if __name__ == "__main__":
    start = time.time()
    get_pull_req_reviewers()
    end = time.time()
    print(f"Runtime of the program is {end - start}")

