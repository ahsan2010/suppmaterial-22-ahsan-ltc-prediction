#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:29:56 2021

@author: ahsan
"""

import json
import pandas as pd
import csv

project_list = ['apache_lucene', 'apache_wicket', 'apache_hbase', 'apache_groovy', 'apache_hive', 'apache_activemq',
'apache_storm','apache_stratos', 'elastic_elasticsearch', 'jruby_jruby', 'caskdata_cdap']

pr_column_list = ['Proj_Name',
                         'Git_Repo_Name',
                         'Git_Proj_Name',
                         'PR_Url',
                         'PR_Number',
                         'Reviewer_Login',
                         'Reviewer_Id',
                         'Reviewer_Html',
                         'Reviewer_Type',
                         'Review_Submission',
                         'Commit_Id']

project_file_location = '/scratch/ahsan/Java_Exam_Work/Result/pull_request/studied_project.csv'    
json_reviewer_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/'
json_to_csv_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/'
pull_request_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request/'

def convert_json_git_issue_to_csv(project):
    pull_request_file = '{}pr_reports_{}.csv'.format(pull_request_folder, project)
    reviewer_list = []
    with open(pull_request_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                git_repo_name = row['Git_Repo_Name']
                git_proj_name = row['Git_Proj_Name']
                proj_name = row['Proj_Name']
                pr_url = row['PR_Url']
                pr_number = row['PR_Number']        
                
                json_output_dir = '{}{}/'.format(json_reviewer_folder, proj_name)
                reviewer_json_file = "{}pull_request_reviews_{}_{}.json".format(json_output_dir,proj_name, pr_number)
                #print(reviewer_json_file)
                with open(reviewer_json_file,encoding="utf8") as jsonfile:
                    json_data = json.load(jsonfile)
                    item_num = 0
                    for item in json_data['items']:
                        item_num = item_num + 1
                        submitted_at = item['submitted_at']
                        commit_id = item['commit_id']
                        
                        if item['user'] is not None:
                            reviewer_login = item['user']['login']
                            reviewer_id = item['user']['id']
                            reviewer_html_url = item['user']['html_url']
                            reviewer_type = item['user']['type']
                        
                        reviewer_info = [
                               proj_name,
                               git_repo_name,
                               git_proj_name,
                               pr_url,
                               pr_number,
                               reviewer_login,
                               reviewer_id,
                               reviewer_html_url,
                               reviewer_type,
                               submitted_at,
                               commit_id
                                ]
                        
                        reviewer_list.append(reviewer_info)
            except Exception as e:
                print(str(e))
        
        print(f'Reviewer Length {len(reviewer_list)}')
        output_file = '{}{}_review.csv'.format(json_to_csv_folder, proj_name)
        pd_data_frame = pd.DataFrame(reviewer_list)
        if pd_data_frame.shape[0] == 0:
            pd_data_frame = pd.DataFrame(columns = pr_column_list)
        else:    
            pd_data_frame.columns = pr_column_list
        pd_data_frame.to_csv(output_file, index = False)

def merge_comments_pull_request_data():
    merged_reviewer_pull_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/merged_pull_reviewer_csv_files/'
    for project in project_list:
        print(project)
        #open pull data
        pull_request_file = '{}pr_reports_{}.csv'.format(pull_request_folder, project)
        pull_request_pd = pd.read_csv(pull_request_file)
        #open comment data
        reviewer_file = '{}{}_review.csv'.format(json_to_csv_folder, project)
        reviewer_pd = pd.read_csv(reviewer_file)
        
        reviewer_merge_pull_request = reviewer_pd.merge(pull_request_pd, on = ['Proj_Name',
                         'Git_Repo_Name',
                         'Git_Proj_Name',
                         'PR_Url',
                         'PR_Number'], how = 'left')
        
        
        reviewer_merge_pull_request.to_csv('{}{}_merged_pull_reviewers.csv'.format(merged_reviewer_pull_folder, project), index = False)
  
    
if __name__ == "__main__":
    for project in project_list:
        print(project)
        convert_json_git_issue_to_csv(project)
        '''
        try:
            print(project)
            convert_json_git_issue_to_csv(project)
        except Exception as e:
            print('Failed to upload to ftp: '+ str(e) + ' ' + project)
            '''
    print('Program finishes successfully')