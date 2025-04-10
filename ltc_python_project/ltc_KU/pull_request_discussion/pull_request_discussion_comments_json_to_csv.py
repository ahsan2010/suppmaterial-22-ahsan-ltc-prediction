# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 10:45:59 2021

@author: ahsan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:29:56 2021

@author: ahsan
"""

import json
import pandas as pd
import csv

#project_list = ['apache_lucene', 'apache_wicket', 'apache_hbase', 'apache_groovy', 'apache_hive', 'apache_activemq',
#'apache_storm','apache_stratos', 'elastic_elasticsearch', 'jruby_jruby', 'caskdata_cdap']

project_list = ['apache_hbase', 'apache_hive',
'apache_storm','apache_stratos','elastic_elasticsearch']


#project_list = ['apache_hbase']


pr_column_list = [       'Proj_Name',
                         'Git_Repo_Name',
                         'PR_Url',
                         'PR_Number',
                         'Comment_Body',
                         'Comment_Created_At',
                         'Commenter_Login',
                         'Commenter_Id',
                         'Commenter_Html',
                         'Commenter_Type',
                         'Commenter_Commit_Id']

github_string = "https://api.github.com/repos/"
project_file_location = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
json_folder = '/home/local/SAIL/ahsan/LTC_Project/pr-discussion-comments/'
json_to_csv_folder = '/home/local/SAIL/ahsan/LTC_Project/pr-discussion-comments-csv/'
pull_request_folder = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/pull-request/pr-data/csv-pr-data/'

def convert_json_git_issue_to_csv(project):
    pull_request_file = '{}pr_{}.csv'.format(pull_request_folder, project)
    commenter_list = []
    with open(pull_request_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                git_repo_name = row['Git_Repo_Name']
                proj_name = row['Proj_Name']
                pr_url = row['PR_Url']
                pr_number = row['PR_Number']    
                pr_creation = row['PR_Created_At']
                
                json_output_dir = '{}{}/'.format(json_folder, proj_name)
                reviewer_json_file = "{}pull_request_commennts_{}_{}.json".format(json_output_dir,proj_name, pr_number)
                #print(reviewer_json_file)
                with open(reviewer_json_file,encoding="utf8") as jsonfile:
                    json_data = json.load(jsonfile)
                    item_num = 0
                    
                    for item in json_data['items'][0]:
                        item_num = item_num + 1
                        
                        created_at = item['created_at']
                        #commit_id = item['commit_id']
                        body = item['body']
                        
                        if item['user'] is not None:
                            commenter_login = item['user']['login']
                            commenter_id = item['user']['id']
                            commenter_html_url = item['user']['html_url']
                            commenter_type = item['user']['type']
                        
                        commenter_info = [
                               proj_name,
                               git_repo_name,
                               pr_url,
                               pr_number,
                               body,
                               created_at,
                               commenter_login,
                               commenter_id,
                               commenter_html_url,
                               commenter_type,
                               None
                                ]
                        
                        commenter_list.append(commenter_info)
            except Exception as e:
                print(str(e))
        
        print(f'Comment data Length {len(commenter_list)}')
        output_file = '{}{}_comments.csv'.format(json_to_csv_folder, proj_name)
        pd_data_frame = pd.DataFrame(commenter_list)
        if pd_data_frame.shape[0] == 0:
            pd_data_frame = pd.DataFrame(columns = pr_column_list)
        else:    
            pd_data_frame.columns = pr_column_list
        pd_data_frame.to_csv(output_file, index = False)


def merge_comments_pull_request_data():
    merged_comment_pull_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/merged_pull_comment_csv_files/'
    commit_path = '/scratch/ahsan/Java_Exam_Work/Data/commit_merge_data/'
    for project in project_list:
        print(project)
        #commit data
        full_commit_file = '{}{}_full_commit_data.csv'.format(commit_path, project)
        commit_pd = pd.read_csv(full_commit_file)
        #open pull data
        pull_request_file = '{}pr_reports_{}.csv'.format(pull_request_folder, project)
        pull_request_pd = pd.read_csv(pull_request_file)
        pull_request_pd = pull_request_pd.rename (columns = {"PR_Merge_Commit_SHA":"commit_id"})
        #open comment data
        comment_file = '{}{}_comments.csv'.format(json_to_csv_folder, project)
        comment_pd = pd.read_csv(comment_file)
        
        comment_merge_pull_request = comment_pd.merge(pull_request_pd, on = ['Proj_Name',
                         'Git_Repo_Name',
                         'Git_Proj_Name',
                         'PR_Url',
                         'PR_Number'], how = 'left')
        
        
        comment_merge_pull_request_with_commit_info = comment_merge_pull_request.merge(commit_pd, on = ['commit_id'], how = 'inner')
        missing_data = comment_merge_pull_request_with_commit_info[comment_merge_pull_request_with_commit_info['project_name'].isnull()]
        
        
        comment_merge_pull_request.to_csv('{}{}_merged_pull_comments.csv'.format(merged_comment_pull_folder, project), index = False)
        
if __name__ == "__main__":
    project_data = pd.read_csv(project_file_location)
    project_url_list = project_data['url'].to_list()
    
    for project_url in project_url_list:
        project = project_url[len(github_string):]
        project = project.replace("/","-")
        print(project)
        
        #convert_json_git_issue_to_csv(project)
        
        try:
            print(project)
            convert_json_git_issue_to_csv(project)
        except Exception as e:
            print('Failed to upload to ftp: '+ str(e) + ' ' + project)
            
    print('Program finishes successfully')